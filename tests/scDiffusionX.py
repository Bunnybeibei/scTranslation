"""CLI entry point for training/testing the scDiffusionX baseline.

For multi-GPU training, launch with ``torchrun``::

    torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:29502 \\
        tests/scDiffusionX.py --mode train --data_name Brain --modal2 a
"""

import argparse
import os
import sys
import warnings

from lightning.pytorch import seed_everything

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.getcwd()))

from sctranslation.datasets import CustomDataset  # noqa: E402
from sctranslation.models.scdiffusionx import scDiffusionXRunner  # noqa: E402
from sctranslation.utils.config import Config  # noqa: E402


PREPROCESS = {
    "r": scDiffusionXRunner.preprocessing_pipeline_r,
    "a": scDiffusionXRunner.preprocessing_pipeline_a,
    "p": scDiffusionXRunner.preprocessing_pipeline_p,
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
):
    """Train or test the scDiffusionX model on the specified modalities."""
    config = Config(config_path, "scDiffusionX")
    seed_everything(random_seed)
    dataset = CustomDataset(
        root_path, data_name, data_dir=data_path, random_seed=random_seed
    )
    transform = {
        modal1: PREPROCESS[modal1](config),
        modal2: PREPROCESS[modal2](config),
    }
    data = dataset.load_data(transform=transform, modal1=modal1, modal2=modal2)
    model = scDiffusionXRunner(
        config,
        model_dir,
        data_name=data_name,
        random_seed=random_seed,
        saved_path=save_dir,
    )

    if mode == "train":
        model.train(data.get_train(), data.get_valid(), data.get_test())
    elif mode == "test":
        model.test(data.get_train(), data.get_valid(), data.get_test())
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'test'.")


def main():
    root_default = os.environ.get("SCT_ROOT", "./datasets")
    parser = argparse.ArgumentParser(
        description="Multi-modal single-cell data translation with scDiffusionX"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--root_path", type=str, default=root_default,
                        help="Root directory of datasets")
    parser.add_argument("--data_name", type=str, default="Chen_2019")
    parser.add_argument("--config_path", type=str, default="configs/default.yaml")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--model_file", type=str, default="./output/scDiffusionX/",
                        help="Directory for model checkpoints")
    parser.add_argument("--saved_path", type=str,
                        default="./output/statistics/scDiffusionX",
                        help="Directory for evaluation outputs")
    parser.add_argument("--data_path", type=str, default="./data/scDiffusionX",
                        help="Directory for cached preprocessed data")
    parser.add_argument("--modal1", type=str, choices=["r", "a", "p"], default="r",
                        help="Primary modality (r: RNA, a: ATAC, p: ADT)")
    parser.add_argument("--modal2", type=str, choices=["r", "a", "p"], default="a",
                        help="Secondary modality (r: RNA, a: ATAC, p: ADT)")
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
    )


if __name__ == "__main__":
    main()
