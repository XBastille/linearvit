import os
import sys
import argparse
import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.linear_vit import LinearViT, SimCLRProjectionHead
from src.data.dataset import CMSContrastiveDataset
from src.training.utils import (
    CosineWarmupScheduler,
    EarlyStopping,
    AverageMeter,
    load_config,
    save_checkpoint,
    count_parameters,
)

LATEST_PATH = "weights/pretrain_simclr_latest.pth"
BEST_PATH = "weights/pretrained_backbone.pt"


def nt_xent_loss(z1, z2, temperature=0.1):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    For a batch of N pairs, computes contrastive loss where
    positive pairs are (z1[i], z2[i]) and all other combinations
    are negative pairs.

    Args:
        z1: (N, D) L2-normalized projections from view 1
        z2: (N, D) L2-normalized projections from view 2
        temperature: scaling temperature

    Returns:
        scalar loss
    """

    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    # Cosine similarity matrix: (2N, 2N)
    sim = torch.mm(z, z.t()) / temperature
    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * N, device=sim.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)
    # Positive pairs: (i, i+N) and (i+N, i)
    pos_idx_1 = torch.arange(N, device=sim.device) + N  # view1 -> view2
    pos_idx_2 = torch.arange(N, device=sim.device)  # view2 -> view1
    pos_idx = torch.cat([pos_idx_1, pos_idx_2])  # (2N,)
    labels = pos_idx.long()
    loss = F.cross_entropy(sim, labels)

    return loss


def pretrain_simclr(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    cfg_data = config["data"]
    cfg_model = config["model"]
    cfg_train = config["training"]
    seed = cfg_data.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    temperature = cfg_train.get("temperature", 0.1)
    dataset = CMSContrastiveDataset(cfg_data["unlabelled_path"])
    # 95% train, 5% val (for monitoring only)
    n = len(dataset)
    n_val = max(1, int(0.05 * n))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_data["batch_size"],
        shuffle=True,
        num_workers=cfg_data.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_data["batch_size"],
        shuffle=False,
        num_workers=cfg_data.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    sample = dataset[0]
    in_channels = sample[0].shape[0]
    img_size = sample[0].shape[1]
    embed_dim = cfg_model.get("embed_dim", 256)
    proj_dim = cfg_model.get("projection_dim", 128)
    print(f"Image: channels={in_channels}, size={img_size}", flush=True)
    print(f"SimCLR: projection_dim={proj_dim}, temperature={temperature}", flush=True)

    # Encoder (backbone)
    model = LinearViT(
        in_channels=in_channels,
        img_size=img_size,
        patch_size=cfg_model.get("patch_size", 5),
        embed_dim=embed_dim,
        depth=cfg_model.get("depth", 6),
        num_heads=cfg_model.get("num_heads", 4),
        mlp_ratio=cfg_model.get("mlp_ratio", 4.0),
        num_classes=2,  
        drop_rate=cfg_model.get("dropout", 0.1),
        drop_path_rate=cfg_model.get("drop_path_rate", 0.1),
        lcm_kernel=cfg_model.get("lcm_kernel", 7),
    ).to(device)

    # Projection head (discarded after pretraining)
    proj_head = SimCLRProjectionHead(
        embed_dim=embed_dim,
        hidden_dim=embed_dim,
        proj_dim=proj_dim,
    ).to(device)

    print(f"Backbone params: {count_parameters(model):,}", flush=True)
    print(f"Projection head params: {count_parameters(proj_head):,}", flush=True)
    params = list(model.parameters()) + list(proj_head.parameters())

    optimizer = torch.optim.AdamW(
        params, lr=cfg_train["lr"], weight_decay=cfg_train.get("weight_decay", 0.05)
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg_train.get("warmup_epochs", 5),
        total_epochs=cfg_train["epochs"],
    )

    patience = cfg_train.get("patience", 15)
    early_stopping = EarlyStopping(patience=patience, mode="min")
    os.makedirs("weights", exist_ok=True)
    best_val_loss = float("inf")
    epochs = cfg_train["epochs"]
    start_epoch = 0

    if os.path.exists(LATEST_PATH):
        print(f"Resuming from {LATEST_PATH}...", flush=True)
        ckpt = torch.load(LATEST_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["encoder_state_dict"])
        proj_head.load_state_dict(ckpt["proj_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        early_stopping.best = ckpt.get("es_best", None)
        early_stopping.counter = ckpt.get("es_counter", 0)

        for _ in range(start_epoch):
            scheduler.step()

        print(
            f"Resumed at epoch {start_epoch + 1}, best_val_loss={best_val_loss:.6f}",
            flush=True,
        )

    else:
        print("Starting fresh (no latest checkpoint found)", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        proj_head.train()
        loss_meter = AverageMeter()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [train]",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            view1, view2 = batch[0].to(device), batch[1].to(device)

            # Forward both views through backbone + projection
            z1 = model.forward_contrastive(view1, proj_head)
            z2 = model.forward_contrastive(view2, proj_head)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            loss_meter.update(loss.item(), view1.shape[0])
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        scheduler.step()

        model.eval()
        proj_head.eval()
        val_meter = AverageMeter()

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{epochs} [val]",
                leave=False,
                dynamic_ncols=True,
            ):
                view1, view2 = batch[0].to(device), batch[1].to(device)
                z1 = model.forward_contrastive(view1, proj_head)
                z2 = model.forward_contrastive(view2, proj_head)
                loss = nt_xent_loss(z1, z2, temperature=temperature)
                val_meter.update(loss.item(), view1.shape[0])

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_loss={loss_meter.avg:.4f} "
            f"val_loss={val_meter.avg:.4f} "
            f"lr={lr:.2e}",
            flush=True,
        )

        improved = early_stopping.step(val_meter.avg)
        if val_meter.avg < best_val_loss:
            best_val_loss = val_meter.avg

        if improved:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "encoder_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                },
                BEST_PATH,
            )
            
            print(f"  -> New best! Saved backbone to {BEST_PATH}", flush=True)

        save_checkpoint(
            {
                "epoch": epoch,
                "encoder_state_dict": model.state_dict(),
                "proj_head_state_dict": proj_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_loss": val_meter.avg,
                "es_best": early_stopping.best,
                "es_counter": early_stopping.counter,
                "config": config,
            },
            LATEST_PATH,
        )

        if early_stopping.should_stop:
            print(
                f"Early stopping at epoch {epoch + 1} (patience={patience})", flush=True
            )
            break

    print(
        f"SimCLR pretraining complete. Best val loss: {best_val_loss:.4f}", flush=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain_simclr.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    pretrain_simclr(config)


if __name__ == "__main__":
    main()
