"""Walk a results directory, evaluate every translation prediction it contains.

For each folder under ``--output_root`` that contains
``test_<task>_pred.h5ad`` and the matching ``test_<task>_truth.h5ad``
(``task`` in ``{a2r, r2a, r2p}``), compute the full set of metrics and write
``evaluation_results3.csv`` into that folder.

Usage
-----
    python tests/evaluation.py --output_root ./output/statistics
    python tests/evaluation.py --output_root ./output4000/statistics
"""

import argparse
import inspect
import os
from pathlib import Path

import scanpy as sc

from sctranslation.metrics import scTranslation_eval

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec


PAIR_TASKS = (("a2r", "r2a"), ("r2p", None))


def evaluate_paired(root: Path):
    """Evaluate an (a2r, r2a) result directory."""
    required = [
        "test_a2r_pred.h5ad",
        "test_a2r_truth.h5ad",
        "test_r2a_pred.h5ad",
        "test_r2a_truth.h5ad",
    ]
    for name in required:
        if not (root / name).exists():
            print(f"[skip] {root}: missing {name}")
            return

    print(f"[eval] {root}")
    evaluator = scTranslation_eval(saved_path=root)
    for task in ("a2r", "r2a"):
        pred = sc.read_h5ad(root / f"test_{task}_pred.h5ad")
        truth = sc.read_h5ad(root / f"test_{task}_truth.h5ad")
        evaluator.add_data(pred=pred, truth=truth, name=task)
    evaluator.forward()


def evaluate_r2p(root: Path):
    """Evaluate an r2p (RNA -> Protein) result directory."""
    if not (root / "test_r2p_pred.h5ad").exists():
        return
    if not (root / "test_r2p_truth.h5ad").exists():
        print(f"[skip] {root}: missing test_r2p_truth.h5ad")
        return

    print(f"[eval] {root}")
    evaluator = scTranslation_eval(saved_path=root)
    pred = sc.read_h5ad(root / "test_r2p_pred.h5ad")
    truth = sc.read_h5ad(root / "test_r2p_truth.h5ad")
    evaluator.add_data(pred=pred, truth=truth, name="r2p")
    evaluator.forward()


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation over a statistics output directory"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output/statistics",
        help="Root containing <model>/<dataset>/<seed>/test_*.h5ad files",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip directories that already contain evaluation_results3.csv",
    )
    parser.add_argument(
        "--error_log",
        type=str,
        default="error_log.txt",
        help="Path to append errors encountered during evaluation",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.exists():
        raise FileNotFoundError(output_root)

    for root, _dirs, files in os.walk(output_root):
        root_path = Path(root)
        if args.skip_existing and (root_path / "evaluation_results3.csv").exists():
            continue
        try:
            if "test_a2r_pred.h5ad" in files:
                evaluate_paired(root_path)
            if "test_r2p_pred.h5ad" in files:
                evaluate_r2p(root_path)
        except Exception as exc:  # noqa: BLE001
            with open(args.error_log, "a") as fh:
                fh.write(f"{root_path} {exc}\n")
            print(f"[error] {root_path}: {exc}")


if __name__ == "__main__":
    main()
