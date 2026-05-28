"""CLI entry point for training/testing the BABEL baseline.

Usage
-----
    python tests/BABEL.py --mode test --data_name id1 --modal2 p

All paths default to relative locations under the project root and can be
overridden via CLI flags or the ``SCT_ROOT`` environment variable.
"""

import argparse
import os
import sys
import warnings

from lightning.pytorch import seed_everything

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.getcwd()))

from sctranslation.datasets import CustomDataset  # noqa: E402
from sctranslation.models.babel import BABELRunner  # noqa: E402
from sctranslation.utils.config import Config  # noqa: E402


PREPROCESS = {
    "r": BABELRunner.preprocessing_pipeline_r,
    "a": BABELRunner.preprocessing_pipeline_a,
    "p": BABELRunner.preprocessing_pipeline_p,
}


def run_pipeline(
    mode,
    config_path,
    root_path,
    data_name,
    model_dir,
    data_path,
    save_dir="",
    modal1="r",
    modal2="a",
    random_seed=0,
    test_batch_size=10000,
):
    """Train or test the BABEL model on the specified modalities."""
    config = Config(config_path, "BABEL")
    seed_everything(random_seed)
    dataset = CustomDataset(
        root_path, data_name, data_dir=data_path, random_seed=random_seed
    )
    transform = {
        modal1: PREPROCESS[modal1](config),
        modal2: PREPROCESS[modal2](config),
    }
    data = dataset.load_data(transform=transform, modal1=modal1, modal2=modal2)
    model = BABELRunner(
        config,
        model_dir,
        data_name=data_name,
        random_seed=random_seed,
        saved_path=save_dir,
    )

    if mode == "train":
        model.train(data.get_train(), data.get_valid(), data.get_test())
    elif mode == "test":
        model.test(
            data.get_train(),
            data.get_valid(),
            data.get_test(),
            batch_size=test_batch_size,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'test'.")


def main():
    root_default = os.environ.get("SCT_ROOT", "./datasets")
    parser = argparse.ArgumentParser(
        description="Multi-modal single-cell data translation with BABEL"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--root_path", type=str, default=root_default,
                        help="Root directory of datasets")
    parser.add_argument("--data_name", type=str, default="id1",
                        help="Dataset name (subdirectory under root_path)")
    parser.add_argument("--config_path", type=str, default="configs/default.yaml")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--model_file", type=str, default="./output/BABEL/",
                        help="Directory for model checkpoints")
    parser.add_argument("--saved_path", type=str,
                        default="./output/statistics/BABEL",
                        help="Directory for evaluation outputs")
    parser.add_argument("--data_path", type=str, default="./data/BABEL",
                        help="Directory for cached preprocessed data")
    parser.add_argument("--modal1", type=str, choices=["r", "a", "p"], default="r",
                        help="Primary modality (r: RNA, a: ATAC, p: ADT)")
    parser.add_argument("--modal2", type=str, choices=["r", "a", "p"], default="p",
                        help="Secondary modality (r: RNA, a: ATAC, p: ADT)")
    parser.add_argument("--test_batch_size", type=int, default=500,
                        help="Test-time batch size (for large datasets)")
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        config_path=args.config_path,
        root_path=args.root_path,
        data_name=args.data_name,
        model_dir=args.model_file,
        data_path=args.data_path,
        save_dir=args.saved_path,
        modal1=args.modal1,
        modal2=args.modal2,
        random_seed=args.random_seed,
        test_batch_size=args.test_batch_size,
    )


if __name__ == "__main__":
    main()
