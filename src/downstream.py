"""Downstream CIFAR-10 classification with different initializations.

--init     scratch          CIFAR ResNet-18, random init (baseline)
           simclr           CIFAR ResNet-18, SimCLR-pretrained on CIFAR-10 images
           imagenet         ImageNet ResNet-18 (torchvision weights), 128px input
           imagenet_random  ImageNet ResNet-18 architecture, random init, 128px (control)
--protocol finetune         all weights trained end-to-end
           linear           backbone frozen, only a linear classifier is trained
--fraction fraction of the 50k CIFAR-10 training labels used (class-balanced)
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (CKPT_DIR, RESULTS_DIR, Classifier, classification_metrics,
                    extract_features, input_spec, load_cifar10, predict,
                    resnet18_backbone, save_json, seed_everything, stratified_subset,
                    supervised_view)

DEFAULT_STEPS = {1.0: 6000, 0.1: 2000, 0.01: 1000}


def build_model(init, dev):
    variant = {"scratch": "cifar", "simclr": "cifar"}.get(init, init)
    backbone = resnet18_backbone(variant)
    if init == "simclr":
        sd = torch.load(os.path.join(CKPT_DIR, "simclr_resnet18.pt"), map_location="cpu")
        backbone.load_state_dict(sd["backbone"])
    size, norm = input_spec(variant)
    return Classifier(backbone, norm).to(dev).to(memory_format=torch.channels_last), size


def train_finetune(model, size, x, y, args, dev):
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim == 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, momentum=0.9, nesterov=True)
    bs = min(args.batch_size, len(x))
    warm = int(0.05 * args.steps)
    step, perm, ptr = 0, torch.randperm(len(x), device=dev), 0
    while step < args.steps:
        if ptr + bs > len(x):
            perm, ptr = torch.randperm(len(x), device=dev), 0
        idx = perm[ptr:ptr + bs]
        ptr += bs
        lr = args.lr * (step + 1) / warm if step < warm else \
            args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / (args.steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        xb = supervised_view(x[idx], size).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(xb), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 500 == 0:
            print(f"  step {step}/{args.steps} loss {loss.item():.4f}", flush=True)


def train_linear(model, size, x, y, args, dev):
    """Linear evaluation on frozen, pre-extracted (non-augmented) features."""
    ftr = extract_features(model, x, size)
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    ftr = (ftr - mu) / sd
    head = nn.Linear(512, 10).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=args.linear_wd)
    bs = min(1024, len(ftr))
    epochs = 100
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * math.ceil(len(ftr) / bs))
    for _ in range(epochs):
        perm = torch.randperm(len(ftr), device=dev)
        for i in range(0, len(ftr), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(head(ftr[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    # Fold feature standardization into the linear head.
    with torch.no_grad():
        w = head.weight / sd
        b = head.bias - (w * mu).sum(1)
        model.head.weight.copy_(w)
        model.head.bias.copy_(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, choices=["scratch", "simclr", "imagenet", "imagenet_random"])
    ap.add_argument("--protocol", default="finetune", choices=["finetune", "linear"])
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--linear_wd", type=float, default=1e-2)
    args = ap.parse_args()
    if args.steps is None:
        args.steps = DEFAULT_STEPS[args.fraction]
    if args.lr is None:  # higher LR from scratch, smaller LR when starting from pretrained weights
        args.lr = 0.1 if args.init in ("scratch", "imagenet_random") else 0.03

    seed_everything(args.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    xtr, ytr = load_cifar10(dev, train=True)
    xte, yte = load_cifar10(dev, train=False)
    sub = stratified_subset(ytr, args.fraction, seed=args.seed)
    x, y = xtr[sub], ytr[sub]

    model, size = build_model(args.init, dev)
    t0 = time.time()
    print(f"[{args.init}/{args.protocol}] fraction={args.fraction} n={len(x)} seed={args.seed}", flush=True)
    if args.protocol == "finetune":
        train_finetune(model, size, x, y, args, dev)
    else:
        train_linear(model, size, x, y, args, dev)
    train_min = (time.time() - t0) / 60

    pred = predict(model, xte, size).cpu().numpy()
    m = classification_metrics(yte.cpu().numpy(), pred)
    m.update({"init": args.init, "protocol": args.protocol, "fraction": args.fraction,
              "seed": args.seed, "n_train": len(x), "train_minutes": train_min, "args": vars(args)})
    print(f"  -> acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  ({train_min:.1f} min)", flush=True)
    name = f"{args.init}_{args.protocol}_f{args.fraction}_s{args.seed}.json"
    save_json(m, os.path.join(RESULTS_DIR, "runs", name))


if __name__ == "__main__":
    main()
