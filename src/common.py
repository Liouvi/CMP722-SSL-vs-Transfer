"""Shared utilities: data on GPU, batched GPU augmentations, models, metrics.

The whole CIFAR-10 split (50k x 3 x 32 x 32 uint8 = 150 MB) is kept on the GPU,
and augmentations are applied per-sample in a batched way on the GPU. This
removes the CPU data-loading bottleneck and makes SimCLR pretraining feasible
on a single laptop GPU.
"""
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_cifar10(device, train: bool):
    """Return (images uint8 [N,3,32,32] on device, labels int64 [N] on device)."""
    d = np.load(os.path.join(DATA_DIR, f"cifar10_{'train' if train else 'test'}.npz"))
    x = torch.from_numpy(d["x"]).permute(0, 3, 1, 2).contiguous()  # N,3,32,32
    y = torch.from_numpy(d["y"]).long()
    return x.to(device), y.to(device)


def stratified_subset(labels: torch.Tensor, fraction: float, seed: int) -> torch.Tensor:
    """Class-balanced index subset containing `fraction` of the labels."""
    if fraction >= 1.0:
        return torch.arange(len(labels), device=labels.device)
    g = torch.Generator().manual_seed(seed)
    y = labels.cpu()
    idx = []
    for c in range(10):
        cls = (y == c).nonzero().squeeze(1)
        k = max(1, int(round(len(cls) * fraction)))
        idx.append(cls[torch.randperm(len(cls), generator=g)[:k]])
    return torch.cat(idx).to(labels.device)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------- #
# Batched GPU augmentations (independent random parameters per sample)
# --------------------------------------------------------------------------- #
def _affine_sample(x, theta, out_size):
    grid = F.affine_grid(theta, (x.shape[0], 3, out_size, out_size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def random_resized_crop_flip(x, out_size, scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)):
    """Per-sample RandomResizedCrop + horizontal flip via one affine grid_sample.
    x: float [B,3,H,W] in [0,1]."""
    b, dev = x.shape[0], x.device
    area = torch.empty(b, device=dev).uniform_(*scale)
    logr = torch.empty(b, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    r = torch.exp(logr)
    w = torch.sqrt(area * r).clamp(max=1.0)  # crop width as fraction of image
    h = torch.sqrt(area / r).clamp(max=1.0)
    cx = (torch.rand(b, device=dev) * 2 - 1) * (1 - w)  # normalized centre in [-1,1]
    cy = (torch.rand(b, device=dev) * 2 - 1) * (1 - h)
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    return _affine_sample(x, theta, out_size)


def random_shift_flip(x, out_size, max_shift=4):
    """Standard CIFAR augmentation: zero-pad by `max_shift` px, random crop, h-flip.
    Implemented as an integer-pixel translation (+ optional resize to out_size)."""
    b, dev, s = x.shape[0], x.device, x.shape[-1]
    shift = torch.randint(-max_shift, max_shift + 1, (b, 2), device=dev).float() * (2.0 / s)
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = flip
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = shift
    return _affine_sample(x, theta, out_size)


_GRAY_W = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
# RGB <-> YIQ, used for hue rotation
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114],
                         [0.596, -0.274, -0.322],
                         [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


def _gray(x):
    return (x * _GRAY_W.to(x.device)).sum(1, keepdim=True)


def color_jitter(x, s=0.5, p=0.8):
    """SimCLR colour distortion: brightness/contrast/saturation 0.8s, hue 0.2s,
    applied with probability p (fixed order b->c->s->h)."""
    b, dev = x.shape[0], x.device
    apply = (torch.rand(b, 1, 1, 1, device=dev) < p).float()

    def u(lo, hi):
        return torch.empty(b, 1, 1, 1, device=dev).uniform_(lo, hi)

    y = x * u(1 - 0.8 * s, 1 + 0.8 * s)  # brightness
    m = _gray(y).mean((2, 3), keepdim=True)
    y = (y - m) * u(1 - 0.8 * s, 1 + 0.8 * s) + m  # contrast
    g = _gray(y)
    y = (y - g) * u(1 - 0.8 * s, 1 + 0.8 * s) + g  # saturation
    # hue: rotate chroma (I,Q) plane by angle in [-0.2s, 0.2s] * 2pi
    ang = u(-0.2 * s, 0.2 * s).view(b) * 2 * math.pi
    cos, sin = torch.cos(ang), torch.sin(ang)
    rot = torch.zeros(b, 3, 3, device=dev)
    rot[:, 0, 0] = 1
    rot[:, 1, 1], rot[:, 1, 2] = cos, -sin
    rot[:, 2, 1], rot[:, 2, 2] = sin, cos
    m_full = _YIQ2RGB.to(dev) @ rot @ _RGB2YIQ.to(dev)  # b,3,3
    y = torch.einsum("bij,bjhw->bihw", m_full, y)
    y = y.clamp(0, 1)
    return apply * y + (1 - apply) * x


def random_grayscale(x, p=0.2):
    apply = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < p).float()
    return apply * _gray(x).expand_as(x) + (1 - apply) * x


def simclr_view(x_uint8, out_size=32):
    x = x_uint8.float() / 255.0
    x = random_resized_crop_flip(x, out_size)
    x = color_jitter(x)
    x = random_grayscale(x)
    return x


def supervised_view(x_uint8, out_size=32, augment=True):
    x = x_uint8.float() / 255.0
    if augment:
        return random_shift_flip(x, out_size)
    if out_size != x.shape[-1]:
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
    return x


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def resnet18_backbone(variant: str) -> nn.Module:
    """ResNet-18 trunk returning 512-d pooled features.

    variant='cifar'    : 3x3 stride-1 stem, no max-pool (standard for 32x32 inputs).
    variant='imagenet' : original 7x7/2 stem + max-pool, ImageNet-1k weights.
                         Used on 128x128 upsampled inputs so that the post-stem
                         feature map (32x32) matches the CIFAR variant.
    variant='imagenet_random' : ImageNet architecture, random init (control).
    """
    if variant == "imagenet":
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        net = torchvision.models.resnet18(weights=None)
        if variant == "cifar":
            net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net


def input_spec(variant: str):
    """(input resolution, normalization) for each backbone variant."""
    if variant in ("imagenet", "imagenet_random"):
        return 128, Normalize(IMNET_MEAN, IMNET_STD)
    return 32, Normalize(CIFAR_MEAN, CIFAR_STD)


class Classifier(nn.Module):
    def __init__(self, backbone, norm, num_classes=10):
        super().__init__()
        self.norm = norm
        self.backbone = backbone
        self.head = nn.Linear(512, num_classes)

    def features(self, x):
        return self.backbone(self.norm(x))

    def forward(self, x):
        return self.head(self.features(x))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(np.float64)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict(model, x_uint8, out_size, bs=1000):
    model.eval()
    preds = []
    for i in range(0, len(x_uint8), bs):
        xb = supervised_view(x_uint8[i:i + bs], out_size, augment=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds.append(model(xb.contiguous(memory_format=torch.channels_last)).argmax(1))
    return torch.cat(preds)


@torch.no_grad()
def extract_features(model, x_uint8, out_size, bs=1000):
    model.eval()
    feats = []
    for i in range(0, len(x_uint8), bs):
        xb = supervised_view(x_uint8[i:i + bs], out_size, augment=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(model.features(xb.contiguous(memory_format=torch.channels_last)).float())
    return torch.cat(feats)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
