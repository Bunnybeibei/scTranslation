#!/usr/bin/env bash
# ----------------------------------------------------------------------
# Train + test scDiffusionX (uses torchrun for the training stage).
#
# Usage:
#   SCT_ROOT=/path/to/datasets ./scripts/run_scdiffusionx.sh
# ----------------------------------------------------------------------
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SCT_ROOT="${SCT_ROOT:-./datasets}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

CONFIG="${CONFIG:-configs/default.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29502}"

DATASETS=("Chen_2019" "Brain")

for seed in 0 1 2 3 4; do
    for dataset in "${DATASETS[@]}"; do
        echo "==> [train] seed=${seed} dataset=${dataset}"
        torchrun \
            --nproc_per_node="${NPROC_PER_NODE}" \
            --rdzv-endpoint="${RDZV_ENDPOINT}" \
            tests/scDiffusionX.py \
            --data_name "${dataset}" \
            --mode train \
            --random_seed "${seed}" \
            --modal2 a \
            --config_path "${CONFIG}" \
            || echo "[FAIL train] scDiffusionX/${dataset}/${seed}"

        echo "==> [test]  seed=${seed} dataset=${dataset}"
        python tests/scDiffusionX.py \
            --data_name "${dataset}" \
            --mode test \
            --random_seed "${seed}" \
            --modal2 a \
            --config_path "${CONFIG}" \
            || echo "[FAIL test] scDiffusionX/${dataset}/${seed}"
    done
done

echo "==> Running evaluation"
python tests/evaluation.py --output_root ./output/statistics --skip_existing
