"""Training and testing entry point for BABEL.

Wraps ``wukevin/babel`` so it plugs into the benchmark's ``Runner`` interface.
The actual ``SplicedAutoEncoder`` / ``NaiveSplicedAutoEncoder`` definitions live
in :mod:`babel_modelling`; this module only orchestrates dataset construction,
``skorch`` configuration and the train/test loops.
"""

from __future__ import annotations

import itertools
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import skorch
import torch
import torch.nn as nn
from skorch.helper import predefined_split

from sctranslation.metrics import scTranslation_eval
from sctranslation.transforms import (
    ADT_CLR_Transform,
    annotate_babel,
    binary_data,
    clip_data,
    filter_features,
    filter_features_babel,
    identity,
    log1p,
    normalize_r,
    scale,
    use_hvg,
)
from sctranslation.transforms.misc import Compose
from sctranslation.utils.logger import ConfigLogger

from .activations import ClippedSoftplus, Exp
from .babel_dataloader import (
    PairedDataset,
    SingleCellDataset,
    SingleCellDatasetSplit,
    SingleCellProteinDataset,
    SplicedDataset,
    ensure_arr,
    get_encoded_dataset,
)
from .babel_modelling import (
    AssymSplicedAutoEncoder,
    NaiveSplicedAutoEncoder,
    SplicedAutoEncoderSkorchNet,
)
from .loss_functions import BCELoss, L1Loss, QuadLoss
from .models.autoencoders import Decoder


logger = logging.getLogger("BABEL")


OPTIMIZER_DICT = {
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
}
REDUCE_LR_ON_PLATEAU_PARAMS = {
    "mode": "min",
    "factor": 0.1,
    "patience": 10,
    "min_lr": 1e-6,
}


def get_array(data):
    """Return ``data`` as a dense ``ndarray`` regardless of sparsity."""
    if isinstance(data, scipy.sparse.csr_matrix):
        return data.toarray()
    return data


def get_device(i: Optional[int] = None) -> torch.device:
    """Return the i-th GPU if CUDA is available, else CPU."""
    if torch.cuda.is_available() and isinstance(i, int):
        devices = list(range(torch.cuda.device_count()))
        device_idx = devices[i]
        torch.cuda.set_device(device_idx)
        d = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(d)
        return d
    return torch.device("cpu")


def init_logger(config):
    """Configure stdout + file logging for a training run."""
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : {message}",
        style="{",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(config.logger_save_path)
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)
    for noisy in (
        "depthcharge",
        "github",
        "h5py",
        "numba",
        "pytorch_lightning",
        "torch",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BABELRunner:
    """Drive training and inference for the BABEL model.

    Parameters
    ----------
    config
        Parsed configuration object (see :mod:`sctranslation.utils.config`).
    model_filename
        Root directory holding ``<dataset>/<seed>`` checkpoints.
    data_name
        Dataset identifier; combined with ``random_seed`` to namespace runs.
    saved_path
        Root directory for evaluation artefacts.
    random_seed
        Seed used for model init / split indexing.
    """

    def __init__(
        self,
        config,
        model_filename: Optional[str] = None,
        data_name: Optional[str] = None,
        saved_path: Optional[str] = "",
        random_seed: Optional[int] = 0,
    ) -> None:
        self.config = config
        self.model_filename = Path(model_filename) / data_name / str(random_seed)
        self.saved_path = Path(saved_path) / data_name / str(random_seed)
        self.random_seed = random_seed

        self.model_filename.mkdir(parents=True, exist_ok=True)
        self.logger = ConfigLogger(self.model_filename / "log.txt")
        self.logger.info("Preparing to run BABEL on dataset: %s", data_name)
        self.logger.log_config(config)

        self.model: Optional[SplicedAutoEncoderSkorchNet] = None
        self.decoder: Optional[skorch.NeuralNet] = None

    # -- Preprocessing pipelines ------------------------------------------

    @staticmethod
    def preprocessing_pipeline_r(config):
        """RNA preprocessing: annotate, filter, normalise, log1p, (HVG), scale, clip.

        Matches the chain BABEL applies inside ``SingleCellDataset`` upstream
        (normalize_count_table -> clip) plus the benchmark-wide ``scale``.
        Crucially ``normalize_r`` populates ``data.raw`` so ``y_mode='size_norm'``
        works downstream.
        """
        transforms = [
            annotate_babel(),
            filter_features_babel("r"),
            normalize_r(),
            log1p(),
            use_hvg(n_top_genes=config.n_top_genes) if config.use_hvg_flag else identity(),
            scale(),
            clip_data(),
        ]
        return Compose(*transforms)

    @staticmethod
    def preprocessing_pipeline_a(config):
        """ATAC preprocessing: annotate, filter, binarise, (low-freq peak filter)."""
        transforms = [
            annotate_babel(),
            filter_features_babel("a"),
            binary_data(),
            filter_features(fpeaks=config.fpeaks) if config.fpeaks != 0.0 else identity(),
        ]
        return Compose(*transforms)

    @staticmethod
    def preprocessing_pipeline_p(config):
        """ADT preprocessing: CLR transform (BABEL's protein-decoder branch)."""
        transforms = [ADT_CLR_Transform()]
        return Compose(*transforms)

    # -- Dataset construction ---------------------------------------------

    def _build_rna_dataset(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
    ) -> SingleCellDataset:
        """Concatenate train + valid RNA AnnDatas into one indexed dataset."""
        n_train = train_sc.shape_r[0]
        n_valid = val_sc.shape_r[0]
        return SingleCellDataset(
            raw_adata=train_sc.ad["r"].concatenate([val_sc.ad["r"]]),
            data_split_to_idx={
                "train": list(range(n_train)),
                "valid": list(range(n_train, n_train + n_valid)),
                # No held-out test in the concatenated view; tests are run
                # separately from the val_sc payload (see ``test``).
                "test": list(range(n_train, n_train + n_valid)),
            },
            y_mode="size_norm",
        )

    def _build_modal2_dataset(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        modal2: str,
    ) -> SingleCellDataset:
        """Build the paired-modality dataset (ATAC or ADT)."""
        assert modal2 in ("a", "p"), f"Unsupported modal2: {modal2}"
        n_train = getattr(train_sc, f"shape_{modal2}")[0]
        n_valid = getattr(val_sc, f"shape_{modal2}")[0]
        raw = train_sc.ad[modal2].concatenate([val_sc.ad[modal2]])
        return SingleCellDataset(
            raw_adata=raw,
            data_split_to_idx={
                "train": list(range(n_train)),
                "valid": list(range(n_train, n_train + n_valid)),
                "test": list(range(n_train, n_train + n_valid)),
            },
            y_mode="x",
            calc_size_factors=False,
        )

    def _build_paired_dataset(
        self,
        rna_split: SingleCellDatasetSplit,
        modal2_split: SingleCellDatasetSplit,
    ) -> PairedDataset:
        return PairedDataset(rna_split, modal2_split, flat_mode=False)

    @staticmethod
    def _infer_input_dim2(modal2_dataset: SingleCellDataset, modal2: str) -> int:
        """Total feature count for the modal-2 branch."""
        if modal2 == "a":
            # Upstream BABEL splits ATAC features per chromosome and reports a
            # list. The benchmark's modelling layer accepts an int for
            # ``input_dim2``; we use the total dim and skip chrom-splitting.
            return modal2_dataset.data_raw.shape[1]
        # ADT
        return modal2_dataset.data_raw.shape[1]

    # -- Training ---------------------------------------------------------

    def train(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData,
    ) -> None:
        if "a" in train_sc.ad:
            modal2 = "a"
        elif "p" in train_sc.ad:
            modal2 = "p"
        else:
            raise ValueError("train_sc must contain either 'a' (ATAC) or 'p' (ADT)")

        rna_dataset = self._build_rna_dataset(train_sc, val_sc)
        rna_train_split = SingleCellDatasetSplit(rna_dataset, split="train")
        rna_valid_split = SingleCellDatasetSplit(rna_dataset, split="valid")

        modal2_dataset = self._build_modal2_dataset(train_sc, val_sc, modal2)
        modal2_train_split = SingleCellDatasetSplit(modal2_dataset, split="train")
        modal2_valid_split = SingleCellDatasetSplit(modal2_dataset, split="valid")

        sc_dual_train = self._build_paired_dataset(rna_train_split, modal2_train_split)
        sc_dual_valid = self._build_paired_dataset(rna_valid_split, modal2_valid_split)

        input_dim2 = self._infer_input_dim2(modal2_dataset, modal2)

        start_time = time.time()
        if modal2 == "a":
            self.logger.info("Translating between RNA and ATAC")
            self.initialize_model(rna_dataset, input_dim2, sc_dual_valid, modal2=modal2)
            self.logger.info("Starting training...")
            self.model.fit(sc_dual_train, y=None)
        else:  # modal2 == 'p'
            self.logger.info("Translating between RNA and ADT")
            self.initialize_decoder(input_dim2, sc_dual_valid)
            self.logger.info("Starting training...")
            self.decoder.fit(sc_dual_train, y=None)

        elapsed = time.time() - start_time
        self.logger.info("Training took %.2f s (%.2f min)", elapsed, elapsed / 60)

    # -- Testing ----------------------------------------------------------

    def test(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData,
        batch_size: int = 10000,
    ) -> None:
        """Streaming inference + evaluation over the held-out test split.

        Large datasets are chunked so the full prediction never lives in
        memory at once; per-batch artefacts are cached so re-runs skip done
        work.
        """
        if "a" in train_sc.ad:
            modal2 = "a"
        elif "p" in train_sc.ad:
            modal2 = "p"
        else:
            raise ValueError("train_sc must contain either 'a' (ATAC) or 'p' (ADT)")

        total_samples = val_sc.shape_r[0]
        self.logger.info("Total test samples: %d", total_samples)

        num_batches = (total_samples + batch_size - 1) // batch_size
        self.logger.info("Processing in %d batches of size %d", num_batches, batch_size)

        batch_save_dir = self.saved_path / "batch_results"
        batch_save_dir.mkdir(parents=True, exist_ok=True)

        all_r2modal2_preds = []
        all_modal22r_preds = []
        all_truth_modal2_list = []
        all_truth_rna_list = []
        all_obs_modal2_list = []
        all_obs_rna_list = []

        # Initialise the model once using a small slice so that even when every
        # batch is cached we still load the trained checkpoint.
        init_batch_size = min(batch_size, total_samples)
        val_init = _slice_sc(val_sc, init_batch_size, modal2)
        rna_init_dataset = self._build_rna_dataset(train_sc, val_init)
        modal2_init_dataset = self._build_modal2_dataset(train_sc, val_init, modal2)
        sc_dual_init = self._build_paired_dataset(
            SingleCellDatasetSplit(rna_init_dataset, split="valid"),
            SingleCellDatasetSplit(modal2_init_dataset, split="valid"),
        )
        input_dim2 = self._infer_input_dim2(modal2_init_dataset, modal2)

        if modal2 == "a":
            self.initialize_model(
                rna_init_dataset, input_dim2, sc_dual_init, modal2=modal2, train=False
            )
        else:
            self.initialize_decoder(input_dim2, sc_dual_init, train=False)
        self.logger.info("Model initialised for inference")

        start_time = time.time()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)

            batch_file_r2modal2 = batch_save_dir / f"batch_{batch_idx}_r2{modal2}_pred.npz"
            batch_file_modal22r = batch_save_dir / f"batch_{batch_idx}_{modal2}2r_pred.npz"
            batch_file_truth_modal2 = batch_save_dir / f"batch_{batch_idx}_truth_{modal2}.npz"
            batch_file_truth_rna = batch_save_dir / f"batch_{batch_idx}_truth_rna.npz"
            batch_file_obs_modal2 = batch_save_dir / f"batch_{batch_idx}_obs_{modal2}.pkl"
            batch_file_obs_rna = batch_save_dir / f"batch_{batch_idx}_obs_rna.pkl"

            basic_files_exist = (
                batch_file_r2modal2.exists()
                and batch_file_truth_modal2.exists()
                and batch_file_obs_modal2.exists()
            )
            if modal2 == "p":
                batch_files_complete = basic_files_exist and (
                    batch_file_modal22r.exists()
                    and batch_file_truth_rna.exists()
                    and batch_file_obs_rna.exists()
                )
            else:
                batch_files_complete = basic_files_exist

            if batch_files_complete:
                self.logger.info(
                    "Batch %d/%d already processed, loading from disk", batch_idx + 1, num_batches
                )
                all_r2modal2_preds.append(_load_array(batch_file_r2modal2))
                all_truth_modal2_list.append(_load_array(batch_file_truth_modal2))
                with open(batch_file_obs_modal2, "rb") as fh:
                    all_obs_modal2_list.append(pickle.load(fh))
                if modal2 == "p":
                    all_modal22r_preds.append(_load_array(batch_file_modal22r))
                    all_truth_rna_list.append(_load_array(batch_file_truth_rna))
                    with open(batch_file_obs_rna, "rb") as fh:
                        all_obs_rna_list.append(pickle.load(fh))
                continue

            self.logger.info(
                "Processing batch %d/%d: samples %d-%d",
                batch_idx + 1,
                num_batches,
                start_idx,
                end_idx,
            )

            val_batch = _slice_sc(val_sc, (start_idx, end_idx), modal2)
            rna_dataset_b = self._build_rna_dataset(train_sc, val_batch)
            modal2_dataset_b = self._build_modal2_dataset(train_sc, val_batch, modal2)
            rna_test_split = SingleCellDatasetSplit(rna_dataset_b, split="test")
            modal2_test_split = SingleCellDatasetSplit(modal2_dataset_b, split="test")
            sc_dual_test = self._build_paired_dataset(rna_test_split, modal2_test_split)

            # RNA -> modal2
            self.logger.info("Predicting RNA > %s", modal2)
            engine = self.model if modal2 == "a" else self.decoder
            r2modal2_preds = engine.translate_1_to_2(sc_dual_test) if modal2 == "a" \
                else engine.predict(sc_dual_test)
            all_r2modal2_preds.append(r2modal2_preds)
            _save_array(batch_file_r2modal2, r2modal2_preds)

            truth_modal2_batch = get_array(modal2_test_split.data_raw.X)
            all_truth_modal2_list.append(truth_modal2_batch)
            _save_array(batch_file_truth_modal2, truth_modal2_batch)

            obs_modal2 = modal2_test_split.data_raw.obs
            with open(batch_file_obs_modal2, "wb") as fh:
                pickle.dump(obs_modal2, fh)
            all_obs_modal2_list.append(obs_modal2)

            if modal2 == "a":
                # Also store ATAC -> RNA direction (BABEL is symmetric for r/a).
                modal22r_preds = self.model.translate_2_to_1(sc_dual_test)
                all_modal22r_preds.append(modal22r_preds)
                _save_array(batch_file_modal22r, modal22r_preds)

                truth_rna_batch = get_array(rna_test_split.data_raw.raw.X)
                all_truth_rna_list.append(truth_rna_batch)
                _save_array(batch_file_truth_rna, truth_rna_batch)
                obs_rna = rna_test_split.data_raw.obs
                with open(batch_file_obs_rna, "wb") as fh:
                    pickle.dump(obs_rna, fh)
                all_obs_rna_list.append(obs_rna)

            self.logger.info("Batch %d/%d completed", batch_idx + 1, num_batches)

        # Aggregate
        self.logger.info("Aggregating batch predictions")
        sc_rna_modal2_test_preds = _vstack(all_r2modal2_preds)
        truth_modal2_all = _vstack(all_truth_modal2_list)
        obs_modal2_combined = pd.concat(all_obs_modal2_list, ignore_index=True)

        evaluator = scTranslation_eval(model="BABEL", saved_path=self.saved_path)
        pred_modal2 = sc.AnnData(
            get_array(sc_rna_modal2_test_preds),
            var=val_sc.ad[modal2].var,
            obs=obs_modal2_combined,
        )
        truth_modal2 = sc.AnnData(
            get_array(truth_modal2_all),
            var=val_sc.ad[modal2].var,
            obs=obs_modal2_combined,
        )
        evaluator.add_data(pred=pred_modal2, truth=truth_modal2, name=f"r2{modal2}")

        if modal2 == "a":
            sc_modal22r_test_preds = _vstack(all_modal22r_preds)
            truth_rna_all = _vstack(all_truth_rna_list)
            obs_rna_combined = pd.concat(all_obs_rna_list, ignore_index=True)
            pred_rna = sc.AnnData(
                get_array(sc_modal22r_test_preds),
                var=val_sc.ad["r"].var,
                obs=obs_rna_combined,
            )
            truth_rna = sc.AnnData(
                get_array(truth_rna_all),
                var=val_sc.ad["r"].var,
                obs=obs_rna_combined,
            )
            sc.pp.normalize_total(truth_rna, inplace=True)
            evaluator.add_data(pred=pred_rna, truth=truth_rna, name=f"{modal2}2r")

        elapsed = time.time() - start_time
        self.logger.info("Test took %.2f s (%.2f min)", elapsed, elapsed / 60)

    # -- Model construction ----------------------------------------------

    def initialize_model(
        self,
        sc_rna_dataset: SingleCellDataset,
        input_dim2: int,
        sc_dual_valid_dataset: PairedDataset,
        train: bool = True,
        modal2: str = "a",
    ) -> None:
        """Instantiate the spliced autoencoder (RNA <-> ATAC)."""
        param_combos = list(
            itertools.product(
                self.config.hidden,
                self.config.lossweight,
                self.config.lr,
                self.config.batchsize,
                [self.random_seed],
            )
        )
        for h_dim, lw, lr, bs, rand_seed in param_combos:
            outdir_name = self.model_filename
            outdir_name.mkdir(parents=True, exist_ok=True)
            with open(outdir_name / "rna_genes.txt", "w") as sink:
                for gene in sc_rna_dataset.data_raw.var_names:
                    sink.write(gene + "\n")

            model_class = (
                NaiveSplicedAutoEncoder if self.config.naive else AssymSplicedAutoEncoder
            )
            self.model = SplicedAutoEncoderSkorchNet(
                module=model_class,
                module__hidden_dim=h_dim,
                module__input_dim1=sc_rna_dataset.data_raw.shape[1],
                module__input_dim2=input_dim2,
                module__final_activations1=[Exp(), ClippedSoftplus()],
                module__final_activations2=nn.PReLU(),
                module__flat_mode=False,
                module__seed=rand_seed,
                lr=lr,
                criterion=QuadLoss,
                criterion__loss2=L1Loss,
                criterion__loss2_weight=lw,
                criterion__record_history=True,
                optimizer=OPTIMIZER_DICT[self.config.optim],
                iterator_train__shuffle=True,
                device=get_device(self.config.device),
                batch_size=bs,
                max_epochs=500,
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=self.config.earlystop),
                    skorch.callbacks.LRScheduler(
                        policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                        **REDUCE_LR_ON_PLATEAU_PARAMS,
                    ),
                    skorch.callbacks.GradientNormClipping(gradient_clip_value=5),
                    skorch.callbacks.Checkpoint(
                        dirname=str(outdir_name),
                        fn_prefix="net_",
                        monitor="valid_loss_best",
                    ),
                ],
                train_split=predefined_split(sc_dual_valid_dataset),
                iterator_train__num_workers=2,
                iterator_valid__num_workers=2,
            )

        self.logger.info("Instantiated model: %s", type(self.model).__name__)
        if not train:
            cp = skorch.callbacks.Checkpoint(
                dirname=str(self.model_filename), fn_prefix="net_"
            )
            self.model.load_params(checkpoint=cp)
            self.logger.info("Loaded model from %s", self.model_filename)

    def initialize_decoder(
        self,
        input_dim2: int,
        sc_dual_valid_dataset: PairedDataset,
        train: bool = True,
    ) -> None:
        """Instantiate the protein decoder (RNA -> ADT branch)."""
        param_combos = list(
            itertools.product(
                self.config.hidden,
                self.config.lossweight,
                self.config.lr,
                self.config.batchsize,
                [self.random_seed],
            )
        )
        for h_dim, lw, lr, bs, _rand_seed in param_combos:
            outdir_name = self.model_filename
            outdir_name.mkdir(parents=True, exist_ok=True)

            self.decoder = skorch.NeuralNet(
                module=Decoder,
                module__num_units=16,
                module__intermediate_dim=64,
                module__num_outputs=input_dim2,
                module__activation=nn.PReLU,
                module__final_activation=nn.Identity(),
                lr=lr,
                criterion=L1Loss,
                optimizer=OPTIMIZER_DICT[self.config.optim],
                batch_size=bs,
                max_epochs=500,
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=15),
                    skorch.callbacks.LRScheduler(
                        policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                        patience=5,
                        factor=0.1,
                        min_lr=1e-6,
                    ),
                    skorch.callbacks.GradientNormClipping(gradient_clip_value=5),
                    skorch.callbacks.Checkpoint(
                        dirname=str(outdir_name),
                        fn_prefix="net_",
                        monitor="valid_loss_best",
                    ),
                ],
                train_split=predefined_split(sc_dual_valid_dataset),
                iterator_train__num_workers=2,
                iterator_valid__num_workers=2,
            )

        if not train:
            cp = skorch.callbacks.Checkpoint(
                dirname=str(self.model_filename), fn_prefix="net_"
            )
            self.decoder.load_params(checkpoint=cp)
            self.logger.info("Loaded decoder from %s", self.model_filename)


# ---------------------------------------------------------------------------
# Module-private helpers (kept at the bottom so the public API reads first)
# ---------------------------------------------------------------------------


class _ValSlice:
    """Lightweight shim mirroring the ``scData`` API for a slice of cells."""

    __slots__ = ("ad", "shape_r", "shape_a", "shape_p")

    def __init__(self, ad: dict, shape_r, shape_a=None, shape_p=None) -> None:
        self.ad = ad
        self.shape_r = shape_r
        self.shape_a = shape_a
        self.shape_p = shape_p


def _slice_sc(val_sc, batch, modal2: str) -> _ValSlice:
    """Slice ``val_sc`` along the cell axis to ``batch``.

    ``batch`` may be an ``int`` (take the first N cells) or a ``(start, end)``
    pair.
    """
    if isinstance(batch, int):
        start, end = 0, batch
    else:
        start, end = batch
    rna = val_sc.ad["r"][start:end].copy()
    modal2_ad = val_sc.ad[modal2][start:end].copy()
    n = end - start
    return _ValSlice(
        ad={"r": rna, modal2: modal2_ad},
        shape_r=(n, val_sc.shape_r[1]),
        shape_a=(n, val_sc.shape_a[1]) if modal2 == "a" else None,
        shape_p=(n, val_sc.shape_p[1]) if modal2 == "p" else None,
    )


def _save_array(path: Path, arr) -> None:
    """Save ``arr`` as sparse npz when possible, else compressed npz."""
    if scipy.sparse.issparse(arr):
        scipy.sparse.save_npz(path, arr)
    else:
        np.savez_compressed(path, data=arr)


def _load_array(path: Path):
    """Inverse of ``_save_array``."""
    try:
        return scipy.sparse.load_npz(path)
    except (ValueError, KeyError):
        return np.load(path, allow_pickle=True)["data"]


def _vstack(arrays):
    """Stack sparse or dense matrices along axis 0."""
    if not arrays:
        return None
    if scipy.sparse.issparse(arrays[0]):
        return scipy.sparse.vstack(arrays)
    return np.vstack([ensure_arr(a) for a in arrays])
