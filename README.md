# CMP722 — Self-Supervised Pretraining vs. Fine-Tuning vs. Training from Scratch

ResNet-18 on CIFAR-10, comparing three initializations under 1% / 10% / 100% label budgets:

| Initialization | Pretraining data | Labels used in pretraining |
|---|---|---|
| **Scratch** (baseline) | – | – |
| **SimCLR** (self-supervised, contrastive) | CIFAR-10 train images (50k) | none |
| **ImageNet** (transfer learning) | ImageNet-1k (1.28M) | yes (ImageNet labels) |

Each initialization is evaluated with a **linear probe** (frozen backbone) and with **full fine-tuning**.
Metrics: top-1 accuracy, macro-F1 and confusion matrices on the 10k CIFAR-10 test set.

## Results (CIFAR-10 test set, accuracy / macro-F1 %)
1% and 10%: mean ± population std over 3 seeds (different labelled subsets); 100%: one run.

| Method | 1% labels Acc / F1 | 10% labels Acc / F1 | 100% labels Acc / F1 |
|---|---|---|---|
| Scratch (baseline) | 40.58 ± 1.0 / 40.45 | 77.34 ± 0.2 / 77.30 | 93.04 / 93.03 |
| Scratch, ImageNet arch @128px | – | – | 92.32 / 92.31 |
| Random init + linear probe | 27.86 ± 0.9 / 27.19 | 36.85 ± 0.4 / 36.46 | 43.01 / 42.75 |
| SimCLR + linear probe | 78.78 ± 0.4 / 78.84 | 84.70 ± 0.2 / 84.72 | 87.25 / 87.27 |
| SimCLR + fine-tune | 77.59 ± 0.1 / 77.63 | 87.20 ± 0.4 / 87.16 | 94.05 / 94.04 |
| ImageNet + linear probe | 76.26 ± 0.2 / 76.11 | 83.14 ± 0.1 / 83.07 | 86.46 / 86.44 |
| ImageNet + fine-tune | 80.42 ± 0.2 / 80.31 | 91.08 ± 0.2 / 91.03 | 95.88 / 95.88 |

SimCLR (200 epochs, no labels) reaches 85.3% kNN accuracy. Full analysis: [report/report.pdf](report/report.pdf).

## Layout
```
src/common.py           data on GPU, batched GPU augmentations, ResNet-18 variants, metrics
src/prepare_data.py     downloads CIFAR-10 and caches it as .npz
src/pretrain_simclr.py  SimCLR (NT-Xent) pretraining, kNN monitor
src/downstream.py       fine-tune / linear-probe / scratch training on a label fraction
src/make_figures.py     results table + figures for the report
scripts/run_all.sh      reproduces every experiment
results/                per-run JSON (accuracy, macro-F1, per-class F1, confusion matrix)
report/                 LaTeX report + figures
```

## Reproduce
Requires PyTorch ≥ 2.1 with CUDA, torchvision, numpy, matplotlib, pyarrow, Pillow (`requirements.txt`).
```bash
pip install -r requirements.txt
bash scripts/run_all.sh          # ~2.5 h on a single RTX 5060 Laptop GPU (8 GB)
# or stage by stage: data | pretrain | baselines | simclr | figures
```
Single runs:
```bash
cd src
python pretrain_simclr.py --epochs 200
python downstream.py --init simclr   --protocol finetune --fraction 0.01 --seed 0
python downstream.py --init imagenet --protocol linear   --fraction 0.1
python downstream.py --init scratch  --fraction 1.0
```

## Implementation notes
- The full dataset lives on the GPU as uint8; SimCLR augmentations (random resized crop + flip,
  colour jitter incl. hue, random grayscale) are implemented as batched GPU ops with independent
  per-sample parameters, so a laptop GPU is not bottlenecked by CPU data loading.
- The ImageNet model keeps its original 7×7/2 stem + max-pool and is fed 128×128 upsampled
  images, so its post-stem feature map (32×32) matches the CIFAR-stem ResNet-18 used for scratch and
  SimCLR. `imagenet_random` (same architecture & resolution, random init) controls for this choice.
