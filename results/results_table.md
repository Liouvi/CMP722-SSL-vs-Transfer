| Method | 1% labels Acc / F1 | 10% labels Acc / F1 | 100% labels Acc / F1 |
|---|---|---|---|
| Scratch (baseline) | 40.58 ± 1.0 / 40.45 | 77.34 ± 0.2 / 77.30 | 93.04 / 93.03 |
| Scratch, ImageNet arch @128px | – | – | 92.32 / 92.31 |
| Random init + linear probe | 27.86 ± 0.9 / 27.19 | 36.85 ± 0.4 / 36.46 | 43.01 / 42.75 |
| SimCLR + linear probe | 78.78 ± 0.4 / 78.84 | 84.70 ± 0.2 / 84.72 | 87.25 / 87.27 |
| SimCLR + fine-tune | 77.59 ± 0.1 / 77.63 | 87.20 ± 0.4 / 87.16 | 94.05 / 94.04 |
| ImageNet + linear probe | 76.26 ± 0.2 / 76.11 | 83.14 ± 0.1 / 83.07 | 86.46 / 86.44 |
| ImageNet + fine-tune | 80.42 ± 0.2 / 80.31 | 91.08 ± 0.2 / 91.03 | 95.88 / 95.88 |
