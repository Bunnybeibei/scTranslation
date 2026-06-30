# ScTranslation 🐰

A benchmark for cross-modality translation between single-cell modalities
(RNA ⇄ ATAC and RNA ⇄ ADT/Protein) covering six representative methods:

| Model         | Reference implementation re-packaged under | Modalities |
|---------------|--------------------------------------------|------------|
| BABEL         | `sctranslation/models/babel`               | RNA, ATAC, ADT |
| scButterfly   | `sctranslation/models/scbutterfly`         | RNA, ATAC, ADT |
| JAMIE         | `sctranslation/models/jamie`               | RNA, ATAC, ADT |
| multiDGD      | `sctranslation/models/multidgd`            | RNA, ATAC, ADT |
| scPair        | `sctranslation/models/scpair`              | RNA, ATAC, ADT |
| scDiffusionX  | `sctranslation/models/scdiffusionx`        | RNA, ATAC, ADT |

Every method is wrapped behind a uniform `Runner` interface
(`train(...)`, `test(...)`, `preprocessing_pipeline_{r,a,p}`) so the same
preprocessing + dataset + evaluation code can drive all of them.

## Repository layout

```
ScTranslation/
├── configs/                 # YAML configurations (default + HVG-length ablations)
│   ├── default.yaml
│   ├── hvg_500.yaml
│   ├── hvg_1000.yaml
│   ├── hvg_2000.yaml
│   └── hvg_4000.yaml
├── sctranslation/
│   ├── data/                # In-memory containers (scData, scDataManager)
│   ├── datasets/            # CustomDataset loader + train/val/test splitting
│   ├── transforms/          # Preprocessing pipeline (HVG, log1p, TF-IDF, ...)
│   ├── metrics/             # scTranslation_eval + ARI/AMI/PCC/MMD/LISI/AUROC
│   ├── models/              # Per-method Runner implementations
│   └── utils/               # Config loader, logging
├── tests/                   # Per-model CLI entry points
│   ├── BABEL.py
│   ├── scButterfly.py
│   ├── JAMIE.py
│   ├── multiDGD.py
│   ├── scPair.py
│   ├── scDiffusionX.py
│   └── evaluation.py        # Batch evaluation over output/statistics/
├── scripts/                 # Reference shell scripts for sweep-style runs
│   ├── run_baselines.sh
│   └── run_scdiffusionx.sh
├── requirements.txt
└── README.md
```

## Installation

The benchmark wraps six independent codebases; we recommend a fresh conda
environment with Python 3.10:

```bash
conda create -n sctranslation python=3.10 -y
conda activate sctranslation

# PyTorch (pick the build that matches your CUDA toolkit)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install -r requirements.txt
```

Some methods pull in additional native dependencies (`episcanpy`, `scib`,
`blobfile`). If any import fails, install the missing package and re-run.

## Data layout

The dataset loader (`sctranslation/datasets/base.py:CustomDataset`) expects
each dataset to live in its own subdirectory under a common root, with one
AnnData (`.h5ad`) per modality:

```
$SCT_ROOT/
└── <dataset_name>/
    ├── RNA_data.h5ad        # required: rna-seq counts (.X)
    ├── ATAC_data.h5ad       # optional: ATAC peaks (.X)
    ├── ADT_data.h5ad        # optional: ADT counts (.X)
    └── split_<seed>.pkl     # generated automatically on first run
```

Each AnnData must:

- share the same `obs_names` across modalities (paired cells), and
- carry a `cell_type` column in `obs` (used by the clustering metrics).

The five-fold split is created lazily and pickled to
`<dataset_name>/split_<random_seed>.pkl` so subsequent runs are deterministic.

Set the root via the `SCT_ROOT` environment variable or pass `--root_path`
on the command line. The default value is `./datasets`.

## Running a single model

Each method has a thin CLI wrapper under `tests/`:

```bash
export SCT_ROOT=/path/to/datasets
export PYTHONPATH=$(pwd)

# Train (writes checkpoints under output/<MODEL>/<dataset>/<seed>/)
python tests/BABEL.py --mode train --data_name Chen_2019 --modal2 a --random_seed 0

# Test (writes predictions under output/statistics/<MODEL>/<dataset>/<seed>/)
python tests/BABEL.py --mode test  --data_name Chen_2019 --modal2 a --random_seed 0
```

Common flags shared by every runner:

| Flag                | Description                                    | Default                                |
|---------------------|------------------------------------------------|----------------------------------------|
| `--mode`            | `train` or `test`                              | varies per model                       |
| `--data_name`       | Dataset subfolder name                         | example dataset name                   |
| `--root_path`       | Dataset root                                   | `$SCT_ROOT` or `./datasets`            |
| `--config_path`     | YAML config (see `configs/`)                   | `configs/default.yaml`                 |
| `--random_seed`     | Split + initialisation seed                    | `0`                                    |
| `--modal1`/`--modal2`| Modalities (`r`, `a`, `p`)                    | `r` / `a`                              |
| `--model_file`      | Where to read/write checkpoints                | `./output/<MODEL>/`                    |
| `--saved_path`      | Where to write per-cell predictions + metrics  | `./output/statistics/<MODEL>`          |
| `--data_path`       | Cached preprocessed inputs                     | `./data/<MODEL>`                       |

scDiffusionX requires multi-GPU launching for training:

```bash
torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:29502 \
    tests/scDiffusionX.py --mode train --data_name Brain --modal2 a --random_seed 0
python tests/scDiffusionX.py --mode test  --data_name Brain --modal2 a --random_seed 0
```

## Running the full sweep

The shell scripts in `scripts/` reproduce the experiments from the paper:

```bash
# Train + test every model on every dataset/seed in the list, then evaluate
SCT_ROOT=/path/to/datasets ./scripts/run_baselines.sh

# scDiffusionX uses torchrun and is invoked separately
SCT_ROOT=/path/to/datasets ./scripts/run_scdiffusionx.sh
```

Both scripts honour the following environment variables:
`CUDA_VISIBLE_DEVICES`, `SCT_ROOT`, `PYTHONPATH`, `CONFIG`, `MODAL2`,
`NPROC_PER_NODE`, `RDZV_ENDPOINT`.

## Evaluation

Each model's test stage writes `test_<task>_pred.h5ad` and
`test_<task>_truth.h5ad` under
`output/statistics/<MODEL>/<dataset>/<seed>/`, where `<task>` is one of
`a2r`, `r2a`, or `r2p`.

To compute all metrics over an existing results tree:

```bash
python tests/evaluation.py --output_root ./output/statistics --skip_existing
```

The metric suite (see `sctranslation/metrics/evaluate.py`) reports:

- **Clustering** (Leiden vs ground-truth `cell_type`): ARI, AMI, NMI, HOM
- **Expression-level**: PCC, Spearman, MSE, MAE
- **Generation quality**: MMD, iLISI
- **Classification**: AUROC, feature-wise correlation

Per-cell `pcc` is also stored on each output AnnData's `var`.

## HVG-length ablation

The ablation in the paper varies the number of highly variable genes used
for RNA preprocessing. Use the matching config:

```bash
python tests/BABEL.py --mode train --data_name Brain --modal2 a \
    --random_seed 0 \
    --config_path configs/hvg_2000.yaml \
    --model_file ./output2000/BABEL/ \
    --saved_path ./output2000/statistics/BABEL \
    --data_path  ./data2000/BABEL
```

## Reproducibility notes

- The dataset split is determined entirely by `--random_seed` and cached
  to `split_<seed>.pkl`. Delete the pickle to regenerate.
- `seed_everything` is called at the start of every CLI entry point.
- All evaluation outputs are deterministic given the same inputs.

## Citation

If you use this benchmark, please cite the accompanying paper as well as
the original publications for each baseline (BABEL, scButterfly, JAMIE,
multiDGD, scPair, scDiffusionX).
```bibtex
@misc{cheng2026sctranslationcomprehensivebenchmarksinglecell,
      title={scTranslation: A Comprehensive Benchmark for Single-Cell Multi-Omics Modality Translation}, 
      author={Jiabei Cheng and Jingbo Zhou and Jun Xia and Changkai Li and Zhen Lei and Chang Yu and Stan Z. Li},
      year={2026},
      eprint={2606.03906},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.03906}, 
}
```
