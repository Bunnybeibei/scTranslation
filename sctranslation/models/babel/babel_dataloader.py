"""Dataset wrappers used by the BABEL runner.

Mirrors the public surface of ``wukevin/babel`` ``sc_data_loaders.py`` but
slimmed down to what the benchmark needs: pre-split AnnData objects are
provided externally and only the train/valid/test indexing, splice/pair
wrapping and per-chromosome view live here.
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch
from anndata import AnnData
from cached_property import cached_property
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_arr(x) -> np.ndarray:
    """Return ``x`` as a dense ``np.ndarray``."""
    if isinstance(x, np.matrix):
        return np.squeeze(np.asarray(x))
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        return x.toarray()
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    raise TypeError(f"Unrecognized type: {type(x)}")


def get_anndata_X(adata: AnnData) -> np.ndarray:
    """Return ``adata.X`` as a dense ``np.ndarray``."""
    return ensure_arr(adata.X)


def is_integral_val(x) -> bool:
    """Check if value(s) can be cast as integer without losing precision."""
    if isinstance(x, (np.ndarray, scipy.sparse.csr_matrix)):
        x_int = x.astype(int)
    else:
        x_int = int(x)
    residuals = x - x_int
    if isinstance(residuals, scipy.sparse.csr_matrix):
        residuals = ensure_arr(residuals[residuals.nonzero()])
    return bool(np.all(np.isclose(residuals, 0)))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SingleCellDataset(Dataset):
    """Dataset over an already-preprocessed AnnData with pre-computed splits.

    The benchmark framework prepares ``train``/``valid``/``test`` indices and
    a fully preprocessed AnnData; this class indexes into that object and
    materialises ``(x, target)`` tensors for skorch.
    """

    def __init__(
        self,
        raw_adata: AnnData,
        data_split_to_idx: Dict[str, List[int]],
        mode: str = "all",
        selfsupervise: bool = True,
        binarize: bool = False,
        calc_size_factors: bool = True,
        concat_outputs: bool = True,
        x_dropout: bool = False,
        y_mode: str = "size_norm",
        return_sf: bool = False,
        transforms: Optional[List[Callable]] = None,
        cluster_res: float = 2.0,
    ) -> None:
        assert mode in ("all", "skip"), (
            "SingleCellDataset operates as a full dataset only. Use "
            "SingleCellDatasetSplit to define data splits."
        )
        assert y_mode in (
            "size_norm",
            "log_size_norm",
            "raw_count",
            "log_raw_count",
            "x",
        ), f"Unrecognized y_mode: {y_mode}"
        if y_mode == "size_norm":
            assert calc_size_factors, "size_norm y_mode requires size factors"
        assert raw_adata is not None
        assert data_split_to_idx is not None

        self.mode = mode
        self.selfsupervise = selfsupervise
        self.x_dropout = x_dropout
        self.y_mode = y_mode
        self.binarize = binarize
        self.calc_size_factors = calc_size_factors
        self.return_sf = return_sf
        self.transforms = transforms or []
        self.concat_outputs = concat_outputs
        self.cluster_res = cluster_res

        self.data_raw = raw_adata
        # Defensive copy so we don't mutate the caller's dict
        self.data_split_to_idx = copy.copy(data_split_to_idx)
        self.data_split_to_idx["all"] = np.arange(len(self.data_raw))

        self.features_names = self.data_raw.var_names
        self.sample_names = self.data_raw.obs_names

        # Materialise the feature matrix once as a dense float tensor.
        self.features = torch.from_numpy(get_anndata_X(self.data_raw)).type(
            torch.FloatTensor
        )

        # Optional size factors (only consumed when ``return_sf`` is True).
        if "size_factors" in self.data_raw.obs.columns:
            self.size_factors = torch.from_numpy(
                self.data_raw.obs["size_factors"].values
            ).type(torch.FloatTensor)
        else:
            self.size_factors = None

        if self.y_mode == "size_norm":
            self._size_norm_counts = self._set_size_norm_counts()

        # Lazily built; only relevant when chrom-split mode is enabled.
        self._chrom_to_idx: Optional[Dict[str, np.ndarray]] = None

    # -- size-normalised view ---------------------------------------------

    def _set_size_norm_counts(self) -> AnnData:
        logging.info("Setting size normalised counts")
        if self.data_raw.raw is None:
            raise ValueError(
                "y_mode='size_norm' requires data_raw.raw to be populated by "
                "the preprocessing pipeline (use normalize_r before scaling)."
            )
        raw_counts = AnnData(
            scipy.sparse.csr_matrix(ensure_arr(self.data_raw.raw.X)),
            obs=pd.DataFrame(index=self.data_raw.obs_names),
            var=pd.DataFrame(index=self.data_raw.var_names),
        )
        sc.pp.normalize_total(raw_counts, inplace=True)
        return raw_counts

    @property
    def size_norm_counts(self) -> AnnData:
        if not hasattr(self, "_size_norm_counts"):
            self._size_norm_counts = self._set_size_norm_counts()
        assert self._size_norm_counts.shape == self.data_raw.shape
        return self._size_norm_counts

    # -- chrom-split helpers ---------------------------------------------

    def _get_chrom_idx(self) -> Dict[str, np.ndarray]:
        if self._chrom_to_idx is None:
            if "chrom" not in self.data_raw.var.columns:
                raise KeyError(
                    "data_raw.var['chrom'] is missing; run annotate_babel "
                    "in the preprocessing pipeline to populate it."
                )
            chroms = sorted(set(self.data_raw.var["chrom"]))
            self._chrom_to_idx = {
                chrom: np.where(self.data_raw.var["chrom"] == chrom)[0]
                for chrom in chroms
            }
        return self._chrom_to_idx

    def get_per_chrom_feature_count(self) -> List[int]:
        """Number of features per chromosome (used for ATAC branch sizing)."""
        return [len(indices) for indices in self._get_chrom_idx().values()]

    def _get_chrom_split_features(self, i: int):
        if self.x_dropout:
            raise NotImplementedError
        features = self.features[i]
        assert features.ndim == 1
        retval = tuple(features[indices] for indices in self._get_chrom_idx().values())
        if self.concat_outputs:
            retval = torch.cat(retval)
        return retval

    # -- Dataset protocol --------------------------------------------------

    def __len__(self) -> int:
        return self.data_raw.n_obs

    def get_item_data_split(self, idx: int, split: str):
        assert split in ("train", "valid", "test", "all")
        if split == "all":
            return self.__getitem__(idx)
        return self.__getitem__(self.data_split_to_idx[split][idx])

    def __getitem__(self, i: int):
        expression_data = self.features[i]

        y_idx = i
        if self.y_mode.endswith("size_norm"):
            target = torch.from_numpy(
                ensure_arr(self._size_norm_counts.X[y_idx]).flatten()
            ).type(torch.FloatTensor)
        elif self.y_mode.endswith("raw_count"):
            target = torch.from_numpy(
                ensure_arr(self.data_raw.raw.X[y_idx]).flatten()
            ).type(torch.FloatTensor)
        elif self.y_mode == "x":
            target = self.features[y_idx]
        else:
            raise NotImplementedError(f"Unrecognized y_mode: {self.y_mode}")

        if self.y_mode.startswith("log"):
            target = torch.log1p(target)

        retval: List[torch.Tensor] = [expression_data]
        if self.return_sf and self.size_factors is not None:
            retval.append(self.size_factors[i])
        retval.append(target)
        return tuple(retval)


class SingleCellDatasetSplit(Dataset):
    """View into a ``SingleCellDataset`` restricted to one split."""

    def __init__(self, sc_dataset: SingleCellDataset, split: str) -> None:
        assert isinstance(sc_dataset, SingleCellDataset)
        assert split in sc_dataset.data_split_to_idx
        self.dset = sc_dataset
        self.split = split
        logging.info(
            "Created %s data split with %d examples",
            split,
            len(self.dset.data_split_to_idx[self.split]),
        )

    def __len__(self) -> int:
        return len(self.dset.data_split_to_idx[self.split])

    def __getitem__(self, index: int):
        return self.dset.get_item_data_split(index, self.split)

    @cached_property
    def size_norm_counts(self) -> AnnData:
        indices = self.dset.data_split_to_idx[self.split]
        return self.dset.size_norm_counts[indices].copy()

    @cached_property
    def data_raw(self) -> AnnData:
        indices = self.dset.data_split_to_idx[self.split]
        return self.dset.data_raw[indices].copy()

    @cached_property
    def obs_names(self):
        indices = self.dset.data_split_to_idx[self.split]
        return self.dset.data_raw.obs_names[indices]


class SingleCellProteinDataset(Dataset):
    """CLR-transformed protein counts (already prepared upstream)."""

    def __init__(self, raw_data: AnnData, mode: str = "all") -> None:
        assert mode in ("all", "train", "valid", "test")
        self.data_raw = raw_data.copy()
        self.data_raw.X = ensure_arr(self.data_raw.X)
        suffix = {"train": "-0", "valid": "-1", "test": "-2", "all": ""}[mode]
        if suffix:
            self.data_raw.obs_names = self.data_raw.obs_names + suffix

    def __len__(self) -> int:
        return self.data_raw.n_obs

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vec = self.data_raw.X[i].flatten()
        tensor = torch.from_numpy(vec).type(torch.FloatTensor)
        return tensor, tensor


class DummyDataset(Dataset):
    """Returns dummy values of a given shape for each ``__getitem__`` call."""

    def __init__(self, shape: int, length: int, mode: str = "zeros") -> None:
        assert mode in ("zeros", "random")
        self.shape = shape
        self.length = length
        self.mode = mode

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "zeros":
            x = torch.zeros(self.shape, dtype=torch.float32)
        else:
            x = torch.from_numpy(np.random.random(size=self.shape)).type(
                torch.FloatTensor
            )
        return x, x


def obs_names_from_dataset(dset: Dataset) -> Optional[List[str]]:
    """Extract obs names from a dataset (or ``None`` if unavailable)."""
    if isinstance(dset, DummyDataset):
        return None
    if isinstance(dset, (SplicedDataset, PairedDataset)):
        return dset.obs_names
    if isinstance(dset, EncodedDataset):
        return dset.obs_names
    if isinstance(dset, SingleCellDatasetSplit):
        return list(dset.obs_names)
    if hasattr(dset, "data_raw") and isinstance(dset.data_raw, AnnData):
        return list(dset.data_raw.obs_names)
    return None


class SplicedDataset(Dataset):
    """Combine two datasets so that ``dataset_x``'s input predicts ``dataset_y``'s target."""

    def __init__(
        self,
        dataset_x: Dataset,
        dataset_y: Dataset,
        flat_mode: bool = False,
    ) -> None:
        assert isinstance(dataset_x, Dataset), f"Bad type for dataset_x: {type(dataset_x)}"
        assert isinstance(dataset_y, Dataset), f"Bad type for dataset_y: {type(dataset_y)}"
        assert len(dataset_x) == len(dataset_y), "Mismatched length"
        self.flat_mode = flat_mode

        x_obs_names = obs_names_from_dataset(dataset_x)
        y_obs_names = obs_names_from_dataset(dataset_y)
        if x_obs_names is not None and y_obs_names is not None:
            for i, (x, y) in enumerate(zip(x_obs_names, y_obs_names)):
                # Tolerate suffix-only mismatches (e.g. -0 / -1 from
                # SingleCellProteinDataset) but reject genuine misalignment.
                if x.split("-")[0] != y.split("-")[0]:
                    raise ValueError(
                        f"Datasets have a different label at index {i}: {x} {y}"
                    )
            self.obs_names = list(x_obs_names)
        elif x_obs_names is not None:
            self.obs_names = x_obs_names
        elif y_obs_names is not None:
            self.obs_names = y_obs_names
        else:
            raise ValueError("Both components of the combined dataset are dummy")

        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

    def get_feature_labels(self) -> List[str]:
        return list(self.dataset_x.data_raw.var_names) + list(
            self.dataset_y.data_raw.var_names
        )

    def get_obs_labels(self) -> List[str]:
        return self.obs_names

    def __len__(self) -> int:
        return len(self.dataset_x)

    def __getitem__(self, i: int):
        pair = (self.dataset_x[i][0], self.dataset_y[i][-1])
        if self.flat_mode:
            raise NotImplementedError("Flat mode is not defined for SplicedDataset")
        return pair


class PairedDataset(SplicedDataset):
    """Combines two datasets into ``((x1, x2), (y1, y2))``."""

    def __getitem__(self, i: int):
        x1 = self.dataset_x[i]
        x2 = self.dataset_y[i]
        x_pair = (x1[0], x2[0])
        y_pair = (x1[-1], x2[-1])
        if self.flat_mode:
            return torch.cat(x_pair), torch.cat(y_pair)
        return x_pair, y_pair


class EncodedDataset(Dataset):
    """Sits on top of a PairedDataset returning ``(encoded(x), encoded(x))``."""

    def __init__(
        self,
        sc_dataset: PairedDataset,
        model,
        input_mode: str = "RNA",
    ) -> None:
        assert input_mode in ("RNA", "ATAC"), f"Unrecognized mode: {input_mode}"
        rna_encoded, atac_encoded = model.get_encoded_layer(sc_dataset)
        encoded = rna_encoded if input_mode == "RNA" else atac_encoded
        self.encoded = AnnData(encoded, obs=pd.DataFrame(index=sc_dataset.obs_names))
        self.obs_names = sc_dataset.obs_names

    def __len__(self) -> int:
        return self.encoded.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoded.X[idx]
        tensor = torch.from_numpy(enc).type(torch.FloatTensor)
        return tensor, tensor


def get_encoded_dataset(
    pretrain_encoder,
    atac_bins: Iterable,
    rna_dataset: SingleCellDatasetSplit,
) -> PairedDataset:
    """Pair an RNA dataset with a dummy ATAC dataset (used for ADT prediction)."""
    sc_atac_train_dummy_dataset = DummyDataset(
        shape=len(list(atac_bins)),
        length=len(rna_dataset),
    )
    return PairedDataset(rna_dataset, sc_atac_train_dummy_dataset, flat_mode=True)
