import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def load_results(path):
    """Load checkpoint with test results."""
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def print_comparison_table(pretrained_results, scratch_results):
    """Print a formatted comparison table."""
    header = f"{'Metric':<25} {'Pretrained':<15} {'Scratch':<15}"
    sep = "-" * 55
    print(sep)
    print(header)
    print(sep)

    metrics = [
        ("Test Accuracy", "accuracy"),
        ("Test MAE (mass)", "mae"),
        ("Test MSE (mass)", "mse"),
        ("Test Loss", "loss"),
    ]

    for label, key in metrics:
        pt_val = pretrained_results.get(key, "N/A") if pretrained_results else "N/A"
        sc_val =  scratch_results.get(key, "N/A") if scratch_results else "N/A"

        if isinstance(pt_val, float):
            pt_str = f"{pt_val:.6f}"
        else:
            pt_str = str(pt_val)

        if isinstance(sc_val, float):
            sc_str = f"{sc_val:.6f}"
        else:
            sc_str = str(sc_val)

        print(f"{label:<25} {pt_str:<15} {sc_str:<15}")

    print(sep)


def plot_training_curves(histories, save_dir="reports"):
    """Plot training/validation curves for pretrained and scratch models."""
    matplotlib.use("Agg")

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for label, hist in histories.items():
        if hist is None:
            continue
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], label=f"{label} (train)")
        axes[0].plot(epochs, hist["val_loss"], "--", label=f"{label} (val)")

        axes[1].plot(epochs, hist["train_acc"], label=f"{label} (train)")
        axes[1].plot(epochs, hist["val_acc"], "--", label=f"{label} (val)")

        axes[2].plot(epochs, hist["train_mae"], label=f"{label} (train)")
        axes[2].plot(epochs, hist["val_mae"], "--", label=f"{label} (val)")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Mass MAE")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(labels, preds, num_classes, title, save_path):
    """Plot confusion matrix heatmap."""
    matplotlib.use("Agg")
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_mass_scatter(mass_true, mass_pred, title, save_path):
    """Plot predicted vs true mass scatter."""
    matplotlib.use("Agg")
    mass_true = np.array(mass_true)
    mass_pred = np.array(mass_pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(mass_true, mass_pred, alpha=0.3, s=5)

    lims = [min(mass_true.min(), mass_pred.min()),
            max(mass_true.max(), mass_pred.max())]

    ax.plot(lims, lims, "r--", linewidth=1, label="Ideal")
    ax.set_xlabel("True Mass")
    ax.set_ylabel("Predicted Mass")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_auc(labels, preds_proba, title, save_path):
    """Plot ROC curve and compute AUC."""
    matplotlib.use("Agg")
    labels = np.array(labels)
    preds_proba = np.array(preds_proba)

    if preds_proba.ndim == 1 or preds_proba.shape[1] == 2:
        if preds_proba.ndim == 2:
            preds_proba = preds_proba[:, 1]
        fpr, tpr, _ = roc_curve(labels, preds_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "r--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path} (AUC={roc_auc:.4f})")
        return roc_auc
    return None


def run_evaluation():
    """Load saved results and generate all evaluation outputs."""
    os.makedirs("reports", exist_ok=True)

    pt_ckpt = load_results("weights/linear_vit_pretrained_finetuned.pt")
    sc_ckpt = load_results("weights/linear_vit_scratch_finetuned.pt")

    pt_test = pt_ckpt.get("test_results") if pt_ckpt else None
    sc_test = sc_ckpt.get("test_results") if sc_ckpt else None

    print("\n=== Model Comparison ===")
    print_comparison_table(pt_test, sc_test)

    histories = {}
    if pt_ckpt and "history" in pt_ckpt:
        histories["Pretrained"] = pt_ckpt["history"]
    if sc_ckpt and "history" in sc_ckpt:
        histories["Scratch"] = sc_ckpt["history"]

    if histories:
        plot_training_curves(histories)

    num_classes = 2
    if pt_ckpt:
        num_classes = pt_ckpt.get("num_classes", 2)
    elif sc_ckpt:
        num_classes = sc_ckpt.get("num_classes", 2)

    for tag, results in [("pretrained", pt_test), ("scratch", sc_test)]:
        if results is None:
            continue

        plot_confusion_matrix(
            results["labels"], results["preds"], num_classes,
            f"Confusion Matrix ({tag})",
            f"reports/confusion_matrix_{tag}.png"
        )
        plot_mass_scatter(
            results["mass_true"], results["mass_pred"],
            f"Predicted vs True Mass ({tag})",
            f"reports/mass_scatter_{tag}.png"
        )

    print("\nEvaluation complete. Reports saved to reports/")


if __name__ == "__main__":
    run_evaluation()
