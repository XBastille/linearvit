import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.linear_vit import LinearViT
from src.data.dataset import CMSLabelledDataset, create_splits, inspect_h5
from src.training.utils import (
    CosineWarmupScheduler, EarlyStopping, AverageMeter,
    load_config, save_checkpoint, count_parameters
)


def finetune(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    cfg_data = config["data"]
    cfg_model = config["model"]
    cfg_train = config["training"]

    print("--- Dataset inspection ---", flush=True)
    inspect_h5(cfg_data["labelled_path"])

    splits = create_splits(
        cfg_data["labelled_path"],
        seed=cfg_data.get("seed", 42)
    )

    augment = cfg_data.get("augment", False)
    train_ds = CMSLabelledDataset(cfg_data["labelled_path"], splits["train"], augment=augment)
    val_ds = CMSLabelledDataset(cfg_data["labelled_path"], splits["val"], augment=False)
    test_ds = CMSLabelledDataset(cfg_data["labelled_path"], splits["test"], augment=False)

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
    test_loader = DataLoader(
        test_ds, batch_size=cfg_data["batch_size"],
        shuffle=False, num_workers=cfg_data.get("num_workers", 4),
        pin_memory=True
    )

    img, cls_label, mass_label = train_ds[0]
    in_channels = img.shape[0]
    img_size = img.shape[1]
    print(f"Image: channels={in_channels}, size={img_size}", flush=True)
    print(f"Sample class: {cls_label}, mass: {mass_label.item():.4f}", flush=True)

    with h5py.File(cfg_data["labelled_path"], "r") as f:
        if train_ds.cls_key is not None:
            all_labels = f[train_ds.cls_key][:].flatten()
            num_classes = len(np.unique(all_labels))
        else:
            num_classes = 2

        if train_ds.mass_key is not None:
            all_mass = f[train_ds.mass_key][:].flatten()
            train_mass = all_mass[splits["train"]]
            mass_mean = float(train_mass.mean())
            mass_std = float(train_mass.std())
            if mass_std < 1e-6:
                mass_std = 1.0
        else:
            mass_mean = 0.0
            mass_std = 1.0

    print(f"Number of classes: {num_classes}", flush=True)
    print(f"Mass normalization: mean={mass_mean:.2f}, std={mass_std:.2f}", flush=True)

    model = LinearViT(
        in_channels=in_channels,
        img_size=img_size,
        patch_size=cfg_model.get("patch_size", 5),
        embed_dim=cfg_model.get("embed_dim", 256),
        depth=cfg_model.get("depth", 6),
        num_heads=cfg_model.get("num_heads", 4),
        mlp_ratio=cfg_model.get("mlp_ratio", 4.0),
        num_classes=num_classes,
        drop_rate=cfg_model.get("dropout", 0.1),
        drop_path_rate=cfg_model.get("drop_path_rate", 0.1),
        lcm_kernel=cfg_model.get("lcm_kernel", 7),
    ).to(device)

    pretrained_path = config.get("pretrained_weights", None)
    is_pretrained = pretrained_path is not None and os.path.exists(str(pretrained_path))
    tag = "pretrained" if is_pretrained else "scratch"

    LATEST_PATH = f"weights/finetune_latest_{tag}.pth"
    BEST_PATH = f"weights/linear_vit_{tag}_finetuned.pt"

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "cls_head" in name or "reg_head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    backbone_lr = cfg_train.get("backbone_lr", cfg_train.get("lr", 1e-4))
    head_lr = cfg_train.get("head_lr", cfg_train.get("lr", 1e-4))

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=cfg_train.get("weight_decay", 0.01))

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg_train.get("warmup_epochs", 3),
        total_epochs=cfg_train["epochs"]
    )

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    patience = cfg_train.get("patience", 15)
    early_stopping = EarlyStopping(patience=patience, mode="min")

    lambda_cls = cfg_train.get("lambda_cls", 1.0)
    lambda_reg = cfg_train.get("lambda_reg", 1.0)

    os.makedirs("weights", exist_ok=True)
    epochs = cfg_train["epochs"]
    start_epoch = 0
    best_val_loss = float("inf")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_mae": [], "val_mae": []}

    if os.path.exists(LATEST_PATH):
        print(f"Resuming from {LATEST_PATH}...", flush=True)
        ckpt = torch.load(LATEST_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", history)
        early_stopping.best = ckpt.get("es_best", None)
        early_stopping.counter = ckpt.get("es_counter", 0)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed at epoch {start_epoch + 1}, best_val_loss={best_val_loss:.6f}", flush=True)
    else:
        if is_pretrained:
            ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
            encoder_sd = ckpt.get("encoder_state_dict", ckpt)
            model_sd = model.state_dict()
            filtered = {k: v for k, v in encoder_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
            model_sd.update(filtered)
            model.load_state_dict(model_sd)
            print(f"Loaded pretrained weights from {pretrained_path} ({len(filtered)}/{len(model_sd)} keys)", flush=True)
        else:
            print("Training from scratch (no pretrained weights)", flush=True)

    print(f"Model params: {count_parameters(model):,}", flush=True)
    print(f"Mode: {tag} | backbone_lr={backbone_lr:.2e}, head_lr={head_lr:.2e}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        correct = 0
        total = 0
        mae_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]",
                    leave=False, dynamic_ncols=True)
        for imgs, cls_labels, mass_labels in pbar:
            imgs = imgs.to(device)
            cls_labels = cls_labels.to(device)
            mass_labels = mass_labels.to(device).unsqueeze(1)

            cls_logits, mass_pred = model(imgs)

            mass_norm = (mass_labels - mass_mean) / mass_std

            cls_loss = cls_criterion(cls_logits, cls_labels)
            reg_loss = reg_criterion(mass_pred, mass_norm)
            loss = lambda_cls * cls_loss + lambda_reg * reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_meter.update(loss.item(), imgs.shape[0])
            cls_loss_meter.update(cls_loss.item(), imgs.shape[0])
            reg_loss_meter.update(reg_loss.item(), imgs.shape[0])

            preds = cls_logits.argmax(dim=1)
            correct += (preds == cls_labels).sum().item()
            total += imgs.shape[0]
            mae_meter.update(
                ((mass_pred * mass_std + mass_mean) - mass_labels).abs().mean().item(), imgs.shape[0]
            )
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{correct/total:.4f}")

        scheduler.step()
        train_acc = correct / total
        train_mae = mae_meter.avg

        model.eval()
        val_loss_meter = AverageMeter()
        val_correct = 0
        val_total = 0
        val_mae_meter = AverageMeter()

        with torch.no_grad():
            for imgs, cls_labels, mass_labels in tqdm(val_loader,
                    desc=f"Epoch {epoch+1}/{epochs} [val]",
                    leave=False, dynamic_ncols=True):
                imgs = imgs.to(device)
                cls_labels = cls_labels.to(device)
                mass_labels = mass_labels.to(device).unsqueeze(1)

                cls_logits, mass_pred = model(imgs)

                mass_norm = (mass_labels - mass_mean) / mass_std

                cls_loss = cls_criterion(cls_logits, cls_labels)
                reg_loss = reg_criterion(mass_pred, mass_norm)
                loss = lambda_cls * cls_loss + lambda_reg * reg_loss

                val_loss_meter.update(loss.item(), imgs.shape[0])
                preds = cls_logits.argmax(dim=1)
                val_correct += (preds == cls_labels).sum().item()
                val_total += imgs.shape[0]
                val_mae_meter.update(
                    ((mass_pred * mass_std + mass_mean) - mass_labels).abs().mean().item(), imgs.shape[0]
                )

        val_acc = val_correct / max(val_total, 1)
        val_mae = val_mae_meter.avg

        history["train_loss"].append(loss_meter.avg)
        history["val_loss"].append(val_loss_meter.avg)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"loss={loss_meter.avg:.4f}(cls={cls_loss_meter.avg:.4f},reg={reg_loss_meter.avg:.4f}) "
              f"acc={train_acc:.4f} mae={train_mae:.4f} | "
              f"val_loss={val_loss_meter.avg:.4f} val_acc={val_acc:.4f} val_mae={val_mae:.4f} "
              f"lr={lr:.2e}", flush=True)

        improved = early_stopping.step(val_loss_meter.avg)
        if val_loss_meter.avg < best_val_loss:
            best_val_loss = val_loss_meter.avg
        if improved:
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss_meter.avg,
                "val_acc": val_acc,
                "val_mae": val_mae,
                "config": config,
                "history": history,
                "num_classes": num_classes,
                "mass_mean": mass_mean,
                "mass_std": mass_std,
            }, BEST_PATH)

        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "val_loss": val_loss_meter.avg,
            "es_best": early_stopping.best,
            "es_counter": early_stopping.counter,
            "history": history,
            "config": config,
            "num_classes": num_classes,
            "mass_mean": mass_mean,
            "mass_std": mass_std,
        }, LATEST_PATH)

        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})", flush=True)
            break

    print("\n--- Test Set Evaluation ---", flush=True)
    ckpt = torch.load(BEST_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_results = evaluate_model(model, test_loader, device, cls_criterion, reg_criterion,
                                  lambda_cls, lambda_reg, mass_mean, mass_std)
    print(f"[{tag.upper()}] Test Acc: {test_results['accuracy']:.4f} | "
          f"Test MAE: {test_results['mae']:.4f} | "
          f"Test MSE: {test_results['mse']:.4f} | "
          f"Test Loss: {test_results['loss']:.4f}", flush=True)

    ckpt["test_results"] = test_results
    ckpt["history"] = history
    torch.save(ckpt, BEST_PATH)

    return test_results, history


def evaluate_model(model, loader, device, cls_criterion, reg_criterion,
                   lambda_cls=1.0, lambda_reg=1.0, mass_mean=0.0, mass_std=1.0):
    """Evaluate model on a dataloader. Denormalizes mass predictions."""
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []
    all_mass_pred = []
    all_mass_true = []

    with torch.no_grad():
        for imgs, cls_labels, mass_labels in tqdm(loader, desc="Evaluating",
                                                   leave=False, dynamic_ncols=True):
            imgs = imgs.to(device)
            cls_labels = cls_labels.to(device)
            mass_labels = mass_labels.to(device).unsqueeze(1)

            cls_logits, mass_pred = model(imgs)

            mass_norm = (mass_labels - mass_mean) / mass_std
            cls_loss = cls_criterion(cls_logits, cls_labels)
            reg_loss = reg_criterion(mass_pred, mass_norm)
            loss = lambda_cls * cls_loss + lambda_reg * reg_loss

            loss_meter.update(loss.item(), imgs.shape[0])
            preds = cls_logits.argmax(dim=1)
            correct += (preds == cls_labels).sum().item()
            total += imgs.shape[0]
            mass_pred_real = mass_pred * mass_std + mass_mean
            probs = torch.softmax(cls_logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(cls_labels.cpu().numpy().tolist())
            all_mass_pred.extend(mass_pred_real.cpu().numpy().flatten().tolist())
            all_mass_true.extend(mass_labels.cpu().numpy().flatten().tolist())

    all_mass_pred = np.array(all_mass_pred)
    all_mass_true = np.array(all_mass_true)

    return {
        "loss": loss_meter.avg,
        "accuracy": correct / max(total, 1),
        "mae": float(np.mean(np.abs(all_mass_pred - all_mass_true))),
        "mse": float(np.mean((all_mass_pred - all_mass_true) ** 2)),
        "preds": all_preds,
        "probs": all_probs,
        "labels": all_labels,
        "mass_pred": all_mass_pred.tolist(),
        "mass_true": all_mass_true.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    finetune(config)


if __name__ == "__main__":
    main()
