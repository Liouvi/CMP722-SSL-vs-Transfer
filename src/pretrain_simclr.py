"""SimCLR self-supervised pretraining of a CIFAR ResNet-18 on CIFAR-10 images
(labels are never used).

Chen et al., "A Simple Framework for Contrastive Learning of Visual
Representations", ICML 2020.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (CKPT_DIR, RESULTS_DIR, load_cifar10, resnet18_backbone, input_spec,
                    simclr_view, seed_everything, Classifier, extract_features)


class SimCLR(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = resnet18_backbone("cifar")
        _, self.norm = input_spec("cifar")
        self.projector = nn.Sequential(
            nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Linear(2048, proj_dim))

    def forward(self, x):
        return self.projector(self.backbone(self.norm(x)))


def nt_xent(z1, z2, temperature):
    """Normalized-temperature cross entropy over 2B views (positives = other view)."""
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


@torch.no_grad()
def knn_accuracy(backbone, norm, xtr, ytr, xte, yte, k=200, t=0.1):
    """Weighted kNN monitor (Wu et al. 2018) on frozen backbone features."""
    clf = Classifier(backbone, norm)
    ftr = F.normalize(extract_features(clf, xtr, 32), dim=1)
    fte = F.normalize(extract_features(clf, xte, 32), dim=1)
    correct = 0
    for i in range(0, len(fte), 1000):
        sim = fte[i:i + 1000] @ ftr.t()
        w, idx = sim.topk(k, dim=1)
        votes = torch.zeros(len(w), 10, device=w.device)
        votes.scatter_add_(1, ytr[idx], (w / t).exp())
        correct += (votes.argmax(1) == yte[i:i + 1000]).sum().item()
    return correct / len(fte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.5)        # for batch 512, SGD
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--knn_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, "simclr_resnet18.pt"))
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    xtr, ytr = load_cifar10(dev, train=True)   # ytr used ONLY for the kNN monitor
    xte, yte = load_cifar10(dev, train=False)

    model = SimCLR().to(dev).to(memory_format=torch.channels_last)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim == 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, momentum=0.9, nesterov=True)

    steps_per_epoch = len(xtr) // args.batch_size
    total = args.epochs * steps_per_epoch
    warm = args.warmup * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))

    log = {"args": vars(args), "epochs": []}
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        running = 0.0
        for i in range(steps_per_epoch):
            xb = xtr[perm[i * args.batch_size:(i + 1) * args.batch_size]]
            v1 = simclr_view(xb).contiguous(memory_format=torch.channels_last)
            v2 = simclr_view(xb).contiguous(memory_format=torch.channels_last)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z1, z2 = model(v1), model(v2)
            loss = nt_xent(z1, z2, args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() if i % 10 == 0 else 0.0
            step += 1
        entry = {"epoch": ep + 1, "loss": running / math.ceil(steps_per_epoch / 10),
                 "time_min": (time.time() - t0) / 60}
        if (ep + 1) % args.knn_every == 0 or ep == 0 or ep + 1 == args.epochs:
            entry["knn_acc"] = knn_accuracy(model.backbone, model.norm, xtr, ytr, xte, yte)
        log["epochs"].append(entry)
        print(json.dumps(entry), flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({"backbone": model.backbone.state_dict(), "epoch": ep + 1}, args.out)

    with open(os.path.join(RESULTS_DIR, "simclr_pretrain_log.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
