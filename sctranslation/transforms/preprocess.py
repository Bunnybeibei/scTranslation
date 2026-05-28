from sctranslation.transforms.base import BaseTransform
import scanpy as sc
import numpy as np
import anndata as ad
import scipy
import episcanpy.api as epi
from typing import List
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from .others import (
    annotate_basic_adata_metrics,
    filter_adata_cells_and_genes,
)


# RNA-seq data preprocessing : normalize_r, log1p, use_hvg, clip_data
class normalize_r(BaseTransform):
    """
    Normalize RNA-seq data by size factor.

    This transformation ensures that the total count of each cell is equalized, which helps in reducing the effect of sequencing depth differences between cells.

    Parameters:
        data (sc.AnnData): The RNA-seq data to be normalized.
        data_type (str): The type of data, must be 'r' for RNA-seq.

    Returns:
        sc.AnnData: The normalized RNA-seq data.
    """

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        print("RNA preprocessing: normalize size factor.")
        if data_type != "r":
            raise ValueError("data_type must be RNA-seq (r)")
        if data.raw is None:
            data.raw = data.copy()
        n_counts = np.squeeze(
            np.array(data.X.sum(axis=1))
        )  # Number of total counts per cell
        sc.pp.normalize_total(data)
        data.obs["size_factors"] = n_counts / np.median(n_counts)
        data.uns["median_counts"] = np.median(n_counts)
        print(f"Found median counts of {data.uns['median_counts']}")
        print(f"Found maximum counts of {np.max(n_counts)}")
        return data


class log1p(BaseTransform):
    """
    Apply log1p transformation to RNA-seq data.

    This transformation stabilizes the variance of gene expression data and makes it more suitable for downstream analysis.

    Parameters:
        data (sc.AnnData): The RNA-seq data to be transformed.
        data_type (str): The type of data, must be 'r' for RNA-seq.

    Returns:
        sc.AnnData: The log1p transformed RNA-seq data.
    """

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        if data.raw is None:
            data.raw = data.copy()
        print("RNA preprocessing: log transform RNA data.")
        if data_type != "r":
            raise ValueError("data_type must be RNA-seq (r)")
        sc.pp.log1p(
            data,
            chunked=True,
            copy=False,
            chunk_size=100000,
        )
        return data


class use_hvg(BaseTransform):
    """
    Select highly variable genes (HVGs) from RNA-seq data.

    This transformation helps in reducing the dimensionality of the data by selecting genes that are most informative for downstream analysis.

    Parameters:
        n_top_genes (int): The number of top genes to select. If None, it will use the default value.
        data (sc.AnnData): The RNA-seq data from which to select HVGs.
        data_type (str): The type of data, must be 'r' for RNA-seq.

    Returns:
        sc.AnnData: The RNA-seq data with only the selected HVGs.
    """

    def __init__(
        self, n_top_genes: int = 1000, replace: bool = True, retain_hvg: bool = False
    ):
        super(use_hvg, self).__init__()
        self.n_top_genes = n_top_genes  # if use_hvg_flag else None
        self.replace = replace
        self.retain_hvg = retain_hvg

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        print(
            f"RNA preprocessing: choose top {self.n_top_genes} genes for following training"
        )
        if data_type != "r":
            raise ValueError("data_type must be RNA-seq (r)")
        try:
            sc.pp.highly_variable_genes(data, n_top_genes=self.n_top_genes)
            # New x and change raw
            x_new_raw = sc.AnnData(scipy.sparse.csr_matrix(data.raw.X[:, data.var.highly_variable]), obs=data.obs, var=data.var[data.var.highly_variable])
            data.raw = x_new_raw
        
            if self.replace:
                return data[:, data.var["highly_variable"]]
        except:
            print("Error in sc.pp.highly_variable_genes, using raw data")
            procesed_data = (
                sc.AnnData(X=data.raw.copy().X, var=data.var, obs=data.obs)
                if (hasattr(data, "raw") and data.raw is not None)
                else data.copy()
            )
            print(f"Normalizing data")
            sc.pp.normalize_total(procesed_data)
            print(f"Log transforming data")
            sc.pp.log1p(procesed_data)
            print(f"Finding HVGs")
            sc.pp.highly_variable_genes(procesed_data, n_top_genes=self.n_top_genes)
            data.var["highly_variable"] = procesed_data.var["highly_variable"]
            
            data = data[:, data.var["highly_variable"]]
            if not self.retain_hvg:
                data.var.drop("highly_variable", axis=1, inplace=True)
            if self.replace:
                return data
        return data


class clip_data(BaseTransform):
    """
    Clip the values in the data to a specified percentile range.

    This transformation helps in reducing the effect of outliers in the data.

    Parameters:
        clip (float): The percentile to clip the data. Must be between 0 and 50.
        data (sc.AnnData): The data to be clipped.
        data_type (str): The type of data, must be 'r' for RNA-seq or 'a' for ATAC-seq.

    Returns:
        sc.AnnData: The clipped data.
    """

    def __init__(self, clip: float = 0.5):
        super(clip_data, self).__init__()
        self.clip = clip

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        if self.clip > 0:
            assert isinstance(self.clip, float) and 0.0 < self.clip < 50.0
            print(f"Clipping to {self.clip} percentile")
            clip_low, clip_high = np.percentile(
                data.X.flatten(), [self.clip, 100.0 - self.clip]
            )
            if clip_low == clip_high == 0:
                print("Skipping clipping, as clipping intervals are 0")
            else:
                assert (
                    clip_low < clip_high
                ), f"Got discordant values for clipping ends: {clip_low} {clip_high}"
                data.X = np.clip(data.X, clip_low, clip_high)
        return data


# ATAC-seq data preprocessing : TFIDF, binary_data, filter_features, normalize_a
def TFIDF(count_mat):
    """
    Apply TF-IDF transformation to a count matrix.

    TF-IDF is a technique used to normalize and scale the count data, which helps in reducing the impact of highly abundant features.

    Parameters:
        count_mat (numpy matrix): The count matrix with cells as rows and peaks as columns.

    Returns:
        tfidf_mat (scipy sparse matrix): The TF-IDF transformed matrix.
        divide_title (numpy matrix): The matrix used for division in the TF-IDF process.
        multiply_title (numpy matrix): The matrix used for multiplication in the TF-IDF process.
    """
    count_mat = count_mat.T
    divide_title = np.tile(np.sum(count_mat, axis=0), (count_mat.shape[0], 1))
    # cHeck if divide_title has any 0
    if np.sum(divide_title == 0) > 0:
        # replace 0 with 0.5
        divide_title[divide_title == 0] = 0.5
        # Count columns with 0
        zero_columns = np.sum(count_mat == 0, axis=0)
        # Assure all 1 in column 2
        np.all(count_mat[:, zero_columns == 0] == 1)
        print("Attention: divide_title has 0")
    nfreqs = 1.0 * count_mat / divide_title
    multiply_title = np.tile(
        np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat, axis=1)).reshape(-1, 1),
        (1, count_mat.shape[1]),
    )
    tfidf_mat = scipy.sparse.csr_matrix(np.multiply(nfreqs, multiply_title)).T
    return tfidf_mat, divide_title, multiply_title


# Construct class binary_data, filter_features, tfidf, normalize, chrom_list_generator
class binary_data(BaseTransform):
    """
    Binarize ATAC-seq data.

    This transformation converts the count data into binary values, indicating whether a peak is accessible or not.

    Parameters:
        data (sc.AnnData): The ATAC-seq data to be binarized.
        data_type (str): The type of data, must be 'a' for ATAC-seq.

    Returns:
        sc.AnnData: The binarized ATAC-seq data.
    """

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        print("ATAC preprocessing: binarizing data.")
        if data_type != "a":
            raise ValueError("data_type must be ATAC-seq (a)")
        epi.pp.binarize(data)
        return data


class filter_features(BaseTransform):
    """
    Filter out features (peaks) in ATAC-seq data based on their presence in cells.

    This transformation helps in removing peaks that are not informative due to their low presence across cells.

    Parameters:
        fpeaks (float): The fraction of cells in which a peak must be present to be retained. If None, it will use the default value.
        data (sc.AnnData): The ATAC-seq data from which to filter features.
        data_type (str): The type of data, must be 'a' for ATAC-seq.

    Returns:
        sc.AnnData: The ATAC-seq data with filtered features.
    """

    def __init__(self, fpeaks: float = None):
        super(filter_features, self).__init__()
        self.fpeaks = fpeaks

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        print(
            f"ATAC preprocessing: filter out peaks appear lower than {self.fpeaks*100}% cells."
        )
        if data_type != "a":
            raise ValueError("data_type must be ATAC-seq (a)")
        epi.pp.filter_features(data, min_cells=np.ceil(self.fpeaks * data.shape[0]))
        return data


class tfidf(BaseTransform):
    """
    Apply TF-IDF transformation to ATAC-seq data.

    This transformation helps in normalizing and scaling the count data, reducing the impact of highly abundant peaks.

    Parameters:
        data (sc.AnnData): The ATAC-seq data to be transformed.
        data_type (str): The type of data, must be 'a' for ATAC-seq.

    Returns:
        sc.AnnData: The TF-IDF transformed ATAC-seq data.
    """

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        print("ATAC preprocessing: TF-IDF transformation.")
        if data_type != "a":
            raise ValueError("data_type must be ATAC-seq (a)")
        count_mat = data.X.A.copy()
        data.X, divide_title, multiply_title = TFIDF(count_mat)
        # data.uns['divide_title'] = divide_title
        # data.uns['multiply_title'] = multiply_title
        return data


class normalize_a(BaseTransform):
    """
    Normalize ATAC-seq data by scaling.

    This transformation ensures that the data values are within a comparable range, which can improve downstream analysis.

    Parameters:
        data (sc.AnnData): The ATAC-seq data to be normalized.
        data_type (str): The type of data, must be 'a' for ATAC-seq.

    Returns:
        sc.AnnData: The normalized ATAC-seq data.
    """

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        print("ATAC preprocessing: normalizing data.")
        if data_type != "a":
            raise ValueError("data_type must be ATAC-seq (a)")
        max_temp = np.max(data.X)
        data.X = data.X / max_temp
        return data


# Common functions: identity, add_attribution, scale, is_numeric, is_numeric, annotate_babel, filter_features_babel
class identity(BaseTransform):

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        return data


class add_attribution(BaseTransform):
    def __init__(self, covariate: List[str] = None):
        super(add_attribution, self).__init__()
        self.covariate = covariate

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        if data_type == "a":
            data.var["modality"] = "ATAC"
        elif data_type == "r":
            data.var["modality"] = "GEX"
        elif data_type == "p":
            data.var["modality"] = "ADT"
        if "cell_type" not in data.obs.columns:
            data.obs["cell_type"] = "cell_type"
        if self.covariate is not None:
            for cov in self.covariate:
                data.obs["covariate_" + cov] = data.obs[cov]
        return data


class scale(BaseTransform):
    def __init__(self):
        super(scale, self).__init__()

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        # if data does't have raw, create a copy of data
        if data.raw is None:
            data.raw = data.copy()
        print("Normalizing data to zero mean unit variance")
        try:
            sc.pp.scale(data, zero_center=True, copy=False)
        except:
            scaled = csr_matrix(preprocessing.scale(data.X.A, axis=0))
            scaled[np.isnan(scaled.A)] = 0  # Replace NaN with average
            # Replace scaled x with data1.X
            data = sc.AnnData(X=scaled, obs=data.obs, var=data.var)
        return data


def is_numeric(x) -> bool:
    """Return True if x is numeric"""
    try:
        x = float(x)
        return True
    except ValueError:
        return False


class annotate_babel(BaseTransform):
    # copy from link https://github.com/babel/babel
    def __init__(self):
        super(annotate_babel, self).__init__()

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        return annotate_basic_adata_metrics(data)


class filter_features_babel(BaseTransform):
    # copy from link https://github.com/babel/babel
    def __init__(
        self,
        data_type,
        filter_cell_min_counts=1,
        filter_cell_max_counts=None,
        filter_cell_min_genes=1,
        filter_cell_max_genes=None,
        filter_gene_min_counts=1,
        filter_gene_max_counts=None,
        filter_gene_min_cells=1,
        filter_gene_max_cells=None,
    ):
        super(filter_features_babel, self).__init__()
        self.filter_cell_min_counts = (
            filter_cell_min_counts if data_type == "r" else None
        )
        self.filter_cell_max_counts = filter_cell_max_counts
        self.filter_cell_min_genes = filter_cell_min_genes if data_type == "r" else None
        self.filter_cell_max_genes = filter_cell_max_genes
        self.filter_gene_min_counts = (
            # filter_gene_min_counts if data_type == "r" else 5
            filter_gene_min_counts if data_type == "r" else None
        )
        self.filter_gene_max_counts = filter_gene_max_counts
        self.filter_gene_min_cells = (
            # filter_gene_min_cells if data_type == "r" else 5
            filter_gene_min_cells if data_type == "r" else None
        )
        self.filter_gene_max_cells = (
            # filter_gene_max_cells if data_type == "r" else 0.1
            filter_gene_max_cells if data_type == "r" else None
        )

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        print("Filtering adata cells and genes")
        data = filter_adata_cells_and_genes(
            data,
            filter_cell_min_counts=self.filter_cell_min_counts,
            filter_cell_max_counts=self.filter_cell_max_counts,
            filter_cell_min_genes=self.filter_cell_min_genes,
            filter_cell_max_genes=self.filter_cell_max_genes,
            filter_gene_min_counts=self.filter_gene_min_counts,
            filter_gene_max_counts=self.filter_gene_max_counts,
            filter_gene_min_cells=self.filter_gene_min_cells,
            filter_gene_max_cells=self.filter_gene_max_cells,
        )
        return data


# ADT data preprocessing : CLR_transform
class ADT_CLR_Transform(BaseTransform):
    """
    Binarize ATAC-seq data.

    This transformation converts the count data into binary values, indicating whether a peak is accessible or not.

    Parameters:
        data (sc.AnnData): The ATAC-seq data to be binarized.
        data_type (str): The type of data, must be 'a' for ATAC-seq.

    Returns:
        sc.AnnData: The binarized ATAC-seq data.
    """

    def __call__(self, data: sc.AnnData, data_type: str = None) -> sc.AnnData:
        print("ADT preprocessing: CLR transformation.")
        if data_type != "p":
            raise ValueError("data_type must be ADT (p)")
        return self.CLR_transform(data)  # [0]

    def CLR_transform(self, ADT_data, add_pseudocount: bool = True) -> np.ndarray:
        # copy from link https://github.com/babel/babel
        """
        Centered logratio transformation. Useful for protein data, but

        >>> clr_transform(np.array([0.1, 0.3, 0.4, 0.2]), add_pseudocount=False)
        array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])
        >>> clr_transform(np.array([[0.1, 0.3, 0.4, 0.2], [0.1, 0.3, 0.4, 0.2]]), add_pseudocount=False)
        array([[-0.79451346,  0.30409883,  0.5917809 , -0.10136628],
            [-0.79451346,  0.30409883,  0.5917809 , -0.10136628]])
        """
        x = ADT_data.X.A.copy()
        assert isinstance(x, np.ndarray)
        if add_pseudocount:
            x = x + 1.0
        if len(x.shape) == 1:
            denom = scipy.stats.mstats.gmean(x)
            retval = np.log(x / denom)
        elif len(x.shape) == 2:
            # Assumes that each row is an independent observation
            # and that columns denote features
            per_row = []
            for i in range(x.shape[0]):
                denom = scipy.stats.mstats.gmean(x[i])
                row = np.log(x[i] / denom)
                per_row.append(row)
            assert len(per_row) == x.shape[0]
            retval = np.stack(per_row)
            assert retval.shape == x.shape
        else:
            raise ValueError(f"Cannot CLR transform array with {len(x.shape)} dims")
        new_ADT_data = ad.AnnData(
            X=csr_matrix(retval), obs=ADT_data.obs, var=ADT_data.var
        )
        return new_ADT_data
