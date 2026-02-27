import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

NORM_SCALE = 1.0


def inspect_h5(path):
    """Print the structure and shapes of an HDF5 file."""
    with h5py.File(path, "r") as f:
        print(f"HDF5 file: {path}")
        for key in f.keys():
            ds = f[key]
            print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")
            if len(ds.shape) > 0 and ds.shape[0] > 0:
                sample = ds[0]
                if hasattr(sample, "shape"):
                    print(
                        f"    sample shape: {sample.shape}, min={np.min(sample):.4f}, max={np.max(sample):.4f}"
                    )
                else:
                    print(f"    sample value: {sample}")


def create_splits(labelled_path, seed=42, test_ratio=0.2, val_ratio=0.1):
    """
    Create stratified train/val/test splits from the labelled dataset.

    Returns dict with keys 'train', 'val', 'test' mapping to index arrays.
    Split: 80% train+val, 20% test. Within train+val: 90% train, 10% val.
    """
    with h5py.File(labelled_path, "r") as f:
        keys = list(f.keys())
        img_key = None
        for candidate in ["jet", "X", "images", "data", "x", "all_data", "X_jets"]:
            if candidate in keys:
                img_key = candidate
                break
        if img_key is None:
            for k in keys:
                if len(f[k].shape) >= 3:
                    img_key = k
                    break
        if img_key is None:
            img_key = keys[0]

        cls_key = None
        for candidate in ["Y", "y", "labels", "label", "class", "class_id", "target"]:
            if candidate in keys:
                cls_key = candidate
                break

        n_samples = f[img_key].shape[0]

        if cls_key is not None:
            labels = f[cls_key][:].flatten()
        else:
            labels = np.zeros(n_samples, dtype=int)
            print(f"[WARN] No label key found in {keys}. Using dummy labels for split.")

    indices = np.arange(n_samples)
    if labels.dtype in [np.float32, np.float64]:
        unique = np.unique(labels)
        if len(unique) > 10:
            labels = np.digitize(labels, np.percentile(labels, [25, 50, 75]))
        else:
            labels = labels.astype(int)

    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, stratify=labels, random_state=seed
    )
    trainval_labels = labels[trainval_idx]
    relative_val = val_ratio / (1 - test_ratio)

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=relative_val,
        stratify=trainval_labels,
        random_state=seed,
    )

    print(
        f"Splits -- train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _process_image(img):
    """Convert raw HDF5 image to (C, H, W) float32 tensor, normalized by global constant."""
    img = img.astype(np.float32)

    # Handle channel dimension: (H, W, C) -> (C, H, W) or (H, W) -> (1, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    elif img.ndim == 3:
        # CMS data is (H, W, C) with C=8
        if img.shape[2] < img.shape[0]:
            img = np.transpose(img, (2, 0, 1))

    img = torch.from_numpy(img.copy()) / NORM_SCALE
    return img


def _augment_image(img):
    """Apply random augmentation: flips and cyclic φ-translation.
    CMS detector images in (η, φ) space: φ is periodic (cylindrical detector).
    img: (C, H, W) tensor.
    """
    if torch.rand(1).item() > 0.5:
        img = img.flip(-1)  # φ-flip (azimuthal symmetry)

    if torch.rand(1).item() > 0.5:
        img = img.flip(-2)  # η-flip (forward-backward symmetry)
    # Cyclic roll along φ-axis (W dimension)
    shift = torch.randint(0, img.shape[-1], (1,)).item()
    if shift > 0:
        img = torch.roll(img, shifts=shift, dims=-1)
    return img


class CMSUnlabelledDataset(Dataset):
    """
    Dataset for pretraining on unlabelled CMS HDF5 data.
    Returns images only (no labels). For MAE: returns (image,).
    """

    def __init__(self, h5_path, noise_std=0.0, indices=None, augment=False):
        self.h5_path = h5_path
        self.noise_std = noise_std
        self.augment = augment

        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            self.img_key = None
            for candidate in ["jet", "X", "images", "data", "x", "all_data", "X_jets"]:
                if candidate in keys:
                    self.img_key = candidate
                    break
            if self.img_key is None:
                for k in keys:
                    if len(f[k].shape) >= 3:
                        self.img_key = k
                        break
            if self.img_key is None:
                self.img_key = keys[0]

            total = f[self.img_key].shape[0]

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(total)

        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        real_idx = int(self.indices[idx])
        img = self._file[self.img_key][real_idx]
        img = _process_image(img)

        if self.augment:
            img = _augment_image(img)

        # For denoising pretraining: return (noisy, clean)
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            noisy_img = img + noise
            return noisy_img, img

        # For MAE pretraining: return (image,)
        return (img,)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


class CMSLabelledDataset(Dataset):
    """
    Dataset for finetuning on labelled CMS HDF5 data.
    Returns (image, class_label, mass_label).
    """

    def __init__(self, h5_path, indices=None, augment=False):
        self.h5_path = h5_path
        self.augment = augment

        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            self.img_key = None
            for candidate in ["jet", "X", "images", "data", "x", "all_data", "X_jets"]:
                if candidate in keys:
                    self.img_key = candidate
                    break

            if self.img_key is None:
                for k in keys:
                    if len(f[k].shape) >= 3:
                        self.img_key = k
                        break
            if self.img_key is None:
                self.img_key = keys[0]

            self.cls_key = None
            for candidate in [
                "Y",
                "y",
                "labels",
                "label",
                "class",
                "class_id",
                "target",
            ]:
                if candidate in keys:
                    self.cls_key = candidate
                    break

            self.mass_key = None
            for candidate in ["m", "mass", "mass_label", "Mass"]:
                if candidate in keys:
                    self.mass_key = candidate
                    break

            if self.cls_key is None or self.mass_key is None:
                print(f"[INFO] Available keys: {keys}")
                print(
                    f"[INFO] img_key={self.img_key}, cls_key={self.cls_key}, mass_key={self.mass_key}"
                )
                for k in keys:
                    if k != self.img_key:
                        shape = f[k].shape
                        print(f"  {k}: shape={shape}, dtype={f[k].dtype}")

            total = f[self.img_key].shape[0]

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(total)

        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        real_idx = int(self.indices[idx])
        img = self._file[self.img_key][real_idx]
        img = _process_image(img)

        if self.augment:
            img = _augment_image(img)

        # Class label (Y is shape (1,) so flatten)
        if self.cls_key is not None:
            val = self._file[self.cls_key][real_idx]
            cls_label = int(val.flatten()[0]) if hasattr(val, "flatten") else int(val)
        else:
            cls_label = 0

        # Mass label (m is shape (1,) so flatten)
        if self.mass_key is not None:
            val = self._file[self.mass_key][real_idx]

            mass_label = (
                float(val.flatten()[0]) if hasattr(val, "flatten") else float(val)
            )

        else:
            mass_label = 0.0

        return img, cls_label, torch.tensor(mass_label, dtype=torch.float32)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def _contrastive_augment(img):
    """Physics-aware augmentation for contrastive learning on CMS images.
    Stronger than _augment_image to create diverse views for SimCLR.
    img: (C, H, W) tensor.
    """
    if torch.rand(1).item() > 0.5:
        img = img.flip(-1)

    if torch.rand(1).item() > 0.5:
        img = img.flip(-2)
    # Cyclic roll along φ-axis (W dimension): CMS detector is cylindrical,
    # so φ is periodic. This is a physics-preserving translation in azimuth.
    shift = torch.randint(0, img.shape[-1], (1,)).item()
    if shift > 0:
        img = torch.roll(img, shifts=shift, dims=-1)

    # Channel dropout: randomly zero 1-2 channels (out of 8)
    C = img.shape[0]
    if C > 2:
        n_drop = torch.randint(1, min(3, C), (1,)).item()
        drop_idx = torch.randperm(C)[:n_drop]
        img = img.clone()
        img[drop_idx] = 0.0

    if torch.rand(1).item() > 0.3:
        mask = img > 0
        noise = torch.randn_like(img) * 0.02
        img = img + noise * mask.float()
        img = img.clamp(0.0, 1.0)

    return img


class CMSContrastiveDataset(Dataset):
    """
    Dataset for SimCLR contrastive pretraining on unlabelled CMS data.
    Returns two independently augmented views of each image: (view1, view2).
    """

    def __init__(self, h5_path, indices=None):
        self.h5_path = h5_path

        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            self.img_key = None
            for candidate in ["jet", "X", "images", "data", "x", "all_data", "X_jets"]:
                if candidate in keys:
                    self.img_key = candidate
                    break
            if self.img_key is None:
                for k in keys:
                    if len(f[k].shape) >= 3:
                        self.img_key = k
                        break
            if self.img_key is None:
                self.img_key = keys[0]

            total = f[self.img_key].shape[0]

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(total)

        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        real_idx = int(self.indices[idx])
        img = self._file[self.img_key][real_idx]
        img = _process_image(img)
        view1 = _contrastive_augment(img)
        view2 = _contrastive_augment(img)

        return view1, view2

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
