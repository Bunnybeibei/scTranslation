#!/usr/bin/env bash
# ----------------------------------------------------------------------
# Train + test every baseline on every dataset / seed combination.
#
# Usage:
#   SCT_ROOT=/path/to/datasets ./scripts/run_baselines.sh
#
# Environment variables:
#   SCT_ROOT          Root directory holding <dataset>/{RNA,ATAC,ADT}_data.h5ad
#   CUDA_VISIBLE_DEVICES   GPU(s) to use (default: 0)
#   MODAL2            Secondary modality: a (ATAC) or p (ADT). Default: a
#   CONFIG            YAML config path. Default: configs/default.yaml
# ----------------------------------------------------------------------
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SCT_ROOT="${SCT_ROOT:-./datasets}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

CONFIG="${CONFIG:-configs/default.yaml}"
MODAL2="${MODAL2:-a}"

DATASETS=(
    "Chen_2019"
    "Brain"
)

MODELS=(
    "BABEL"
    "scButterfly"
    "scPair"
    "JAMIE"
    "multiDGD"
)

MODES=("train" "test")

for seed in 0 1 2 3 4; do
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for mode in "${MODES[@]}"; do
                echo "==> seed=${seed} dataset=${dataset} model=${model} mode=${mode}"
                python "tests/${model}.py" \
                    --data_name "${dataset}" \
                    --mode "${mode}" \
                    --random_seed "${seed}" \
                    --modal2 "${MODAL2}" \
                    --config_path "${CONFIG}" \
                    || echo "[FAIL] ${model}/${dataset}/${seed}/${mode}"
            done
        done
    done
done

echo "==> Running evaluation"
python tests/evaluation.py --output_root ./output/statistics --skip_existing
