# Linear Attention ViT for CMS End-to-End Mass Regression and Classification

ML4SCI GSoC 2026 test task: Linear-scale attention vision transformer for particle physics.

## Task

Train a linear-attention ViT on CMS detector images for:
- **Classification**: Identify particle type from collision images
- **Regression**: Predict particle mass from collision images

Compare pretrained-finetuned model vs training from scratch.

## Architecture

Based on L2ViT (Zheng et al., 2025) with ReLU-based linear attention and local concentration module.

- Linear attention: O(N*d^2) complexity instead of O(N^2*d)
- Local concentration module (LCM): depthwise convolutions to concentrate dispersive attention
- Dual heads: classification + mass regression from shared backbone

## Setup

```bash
pip install -r requirements.txt
```

## Download Data

```bash
python download_data.py                  # both datasets
python download_data.py --labelled-only  # labelled only (~5GB)
python download_data.py --unlabelled-only  # unlabelled only (~30GB)
```

## Training Pipeline

### 1. Pretrain (denoising autoencoder on unlabelled data)

```bash
python -m src.training.pretrain --config configs/pretrain.yaml
```

### 2. Finetune (pretrained backbone + dual heads on labelled data)

```bash
python -m src.training.finetune --config configs/finetune.yaml
```

### 3. Train from scratch (same architecture, random init)

```bash
python -m src.training.finetune --config configs/scratch.yaml
```

### 4. Evaluate and compare

```bash
python -m src.training.evaluate
```

## Project Structure

```
linearvit/
  download_data.py             # dataset download script
  requirements.txt             # dependencies
  configs/
    pretrain.yaml              # pretraining hyperparameters
    finetune.yaml              # finetuning hyperparameters
    scratch.yaml               # scratch training hyperparameters
  src/
    data/
      dataset.py               # HDF5 dataset classes + splits
    models/
      linear_vit.py            # linear-attention ViT architecture
    training/
      pretrain.py              # pretraining loop
      finetune.py              # finetuning loop (dual head)
      evaluate.py              # metrics and visualization
      utils.py                 # scheduler, early stopping, helpers
  weights/                     # saved checkpoints
  data/                        # downloaded datasets
  reports/                     # evaluation plots
```

## References

- [L2ViT: The Linear Attention Resurrection in Vision Transformer](https://arxiv.org/abs/2501.16182)
- [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681)
- [ML4SCI GSoC E2E Project](https://ml4sci.org/gsoc/2026/proposal_E2E5.html)
