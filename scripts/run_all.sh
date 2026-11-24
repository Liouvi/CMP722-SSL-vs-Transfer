#!/usr/bin/env bash
# Reproduce every experiment in the report.
#   usage: bash scripts/run_all.sh [all|pretrain|baselines|simclr|figures]
# Seeds 0,1,2 for the 1% / 10% label budgets (high variance), seed 0 for 100%.
set -euo pipefail
cd "$(dirname "$0")/../src"
PY=${PY:-../.venv/bin/python}
STAGE=${1:-all}

runs_for() {  # $1=init $2=protocol
  for f in 0.01 0.1; do for s in 0 1 2; do
    $PY downstream.py --init "$1" --protocol "$2" --fraction $f --seed $s
  done; done
  $PY downstream.py --init "$1" --protocol "$2" --fraction 1.0 --seed 0
}

if [[ $STAGE == all || $STAGE == data ]]; then $PY prepare_data.py; fi

if [[ $STAGE == all || $STAGE == pretrain ]]; then
  $PY pretrain_simclr.py --epochs 200
fi

if [[ $STAGE == all || $STAGE == baselines ]]; then
  runs_for scratch finetune
  runs_for scratch linear          # random-init features: lower bound for linear probing
  runs_for imagenet linear
  runs_for imagenet finetune
  $PY downstream.py --init imagenet_random --fraction 1.0 --seed 0   # architecture/resolution control
fi

if [[ $STAGE == all || $STAGE == simclr ]]; then
  runs_for simclr linear
  runs_for simclr finetune
fi

if [[ $STAGE == all || $STAGE == figures ]]; then $PY make_figures.py --aug; fi
