import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.linear_vit import LinearViT, PretrainDecoder
from src.data.dataset import CMSUnlabelledDataset
from src.training.utils import (
    CosineWarmupScheduler, EarlyStopping, AverageMeter,
    load_config, save_checkpoint, count_parameters
)

LATEST_PATH = "weights/pretrain_latest.pth"
BEST_PATH = "weights/pretrained_backbone.pt"


def patchify(imgs, patch_size):
    """Convert images to patch-level representation for loss computation.
    imgs: (B, C, H, W) -> patches: (B, N, patch_size**2 * C)
    """
    B, C, H, W = imgs.shape
    P = patch_size
    h, w = H // P, W // P
    x = imgs.reshape(B, C, h, P, w, P)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, P, P, C)
    x = x.reshape(B, h * w, P * P * C)
    return x


def pretrain(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    cfg_data = config["data"]
    cfg_model = config["model"]
    cfg_train = config["training"]

    mask_ratio = cfg_train.get("mask_ratio", 0.75)

    dataset = CMSUnlabelledDataset(
        cfg_data["unlabelled_path"],
        noise_std=0.0,  
        augment=True,
    )

    n = len(dataset)
    n_val = max(1, int(0.05 * n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=cfg_data["batch_size"],
        shuffle=True, num_workers=cfg_data.get("num_workers", 4),
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg_data["batch_size"],
        shuffle=False, num_workers=cfg_data.get("num_workers", 4),
        pin_memory=True
    )

    sample = dataset[0]
    in_channels = sample[0].shape[0]
    img_size = sample[0].shape[1]
    patch_size = cfg_model.get("patch_size", 5)
    num_patches = (img_size // patch_size) ** 2
    embed_dim = cfg_model.get("embed_dim", 256)

    print(f"Image: channels={in_channels}, size={img_size}", flush=True)
    print(f"Patches: {num_patches} ({img_size//patch_size}x{img_size//patch_size}), mask_ratio={mask_ratio}", flush=True)

    # Encoder
    model = LinearViT(
        in_channels=in_channels,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=cfg_model.get("depth", 6),
        num_heads=cfg_model.get("num_heads", 4),
        mlp_ratio=cfg_model.get("mlp_ratio", 4.0),
        num_classes=2, 
        drop_rate=cfg_model.get("dropout", 0.1),
        drop_path_rate=cfg_model.get("drop_path_rate", 0.1),
        lcm_kernel=cfg_model.get("lcm_kernel", 7),
    ).to(device)

    # Decoder
    decoder_embed_dim = cfg_model.get("decoder_embed_dim", 128)
    decoder_depth = cfg_model.get("decoder_depth", 2)
    decoder_num_heads = cfg_model.get("decoder_num_heads", 4)

    decoder = PretrainDecoder(
        embed_dim=embed_dim,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        patch_size=patch_size,
        in_channels=in_channels,
        num_patches=num_patches,
    ).to(device)

    print(f"Encoder params: {count_parameters(model):,}", flush=True)
    print(f"Decoder params: {count_parameters(decoder):,}", flush=True)

    params = list(model.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg_train["lr"],
        weight_decay=cfg_train.get("weight_decay", 0.05)
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg_train.get("warmup_epochs", 5),
        total_epochs=cfg_train["epochs"]
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
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        early_stopping.best = ckpt.get("es_best", None)
        early_stopping.counter = ckpt.get("es_counter", 0)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed at epoch {start_epoch + 1}, best_val_loss={best_val_loss:.6f}", flush=True)
    else:
        print("Starting fresh (no latest checkpoint found)", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        decoder.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            imgs = batch[0].to(device)  # (B, C, H, W)

            vis_tokens, mask, ids_restore, H, W = model.forward_mae(imgs, mask_ratio=mask_ratio)

            recon = decoder(vis_tokens, ids_restore, H, W)

            target = imgs[:, :, :recon.shape[2], :recon.shape[3]]
            target_patches = patchify(target, patch_size)  # (B, N, P*P*C)
            recon_patches = patchify(recon, patch_size)     # (B, N, P*P*C)     

            per_patch_loss = ((recon_patches - target_patches) ** 2).mean(dim=-1)  # (B, N)
            loss = (per_patch_loss * mask.float()).sum() / mask.float().sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            loss_meter.update(loss.item(), imgs.shape[0])
            pbar.set_postfix(loss=f"{loss_meter.avg:.6f}")

        scheduler.step()

        model.eval()
        decoder.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]",
                              leave=False, dynamic_ncols=True):
                imgs = batch[0].to(device)

                vis_tokens, mask, ids_restore, H, W = model.forward_mae(imgs, mask_ratio=mask_ratio)
                recon = decoder(vis_tokens, ids_restore, H, W)

                target = imgs[:, :, :recon.shape[2], :recon.shape[3]]
                target_patches = patchify(target, patch_size)
                recon_patches = patchify(recon, patch_size)

                per_patch_loss = ((recon_patches - target_patches) ** 2).mean(dim=-1)
                loss = (per_patch_loss * mask.float()).sum() / mask.float().sum()
                val_meter.update(loss.item(), imgs.shape[0])

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"train_loss={loss_meter.avg:.6f} "
              f"val_loss={val_meter.avg:.6f} "
              f"lr={lr:.2e}", flush=True)

        improved = early_stopping.step(val_meter.avg)
        if val_meter.avg < best_val_loss:
            best_val_loss = val_meter.avg
        if improved:
            save_checkpoint({
                "epoch": epoch,
                "encoder_state_dict": model.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, BEST_PATH)

        save_checkpoint({
            "epoch": epoch,
            "encoder_state_dict": model.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "val_loss": val_meter.avg,
            "es_best": early_stopping.best,
            "es_counter": early_stopping.counter,
            "config": config,
        }, LATEST_PATH)

        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})", flush=True)
            break

    print(f"MAE pretraining complete. Best val loss: {best_val_loss:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    pretrain(config)


if __name__ == "__main__":
    main()
