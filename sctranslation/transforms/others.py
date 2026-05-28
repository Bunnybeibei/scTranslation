import logging

import numpy as np
import scanpy as sc


def annotate_basic_adata_metrics(adata: sc.AnnData) -> None:
    """Annotate with some basic metrics"""
    assert isinstance(adata, sc.AnnData)
    adata.obs["n_counts"] = np.squeeze(np.asarray((adata.X.sum(1))))
    adata.obs["log1p_counts"] = np.log1p(adata.obs["n_counts"])
    adata.obs["n_genes"] = np.squeeze(np.asarray(((adata.X > 0).sum(1))))

    adata.var["n_counts"] = np.squeeze(np.asarray(adata.X.sum(0)))
    adata.var["log1p_counts"] = np.log1p(adata.var["n_counts"])
    adata.var["n_cells"] = np.squeeze(np.asarray((adata.X > 0).sum(0)))
    # check if nan in adata.var['n_cells']
    if np.isnan(adata.var["n_cells"]).any():
        raise ValueError("n_cells contains nan")

    return adata


def filter_adata_cells_and_genes(
    x: sc.AnnData,
    filter_cell_min_counts=None,
    filter_cell_max_counts=None,
    filter_cell_min_genes=None,
    filter_cell_max_genes=None,
    filter_gene_min_counts=None,
    filter_gene_max_counts=None,
    filter_gene_min_cells=None,
    filter_gene_max_cells=None,
) -> None:
    """Filter the count table in place given the parameters based on actual data"""
    args = locals()
    filtering_cells = any(
        [args[arg] is not None for arg in args if arg.startswith("filter_cell")]
    )
    filtering_genes = any(
        [args[arg] is not None for arg in args if arg.startswith("filter_gene")]
    )

    def ensure_count(value, max_value) -> int:
        """Ensure that the value is a count, optionally scaling to be so"""
        if value is None:
            return value  # Pass through None
        retval = value
        if isinstance(value, float):
            assert 0.0 < value < 1.0
            retval = int(round(value * max_value))
        assert isinstance(retval, int)
        return retval

    assert isinstance(x, sc.AnnData)
    # Perform filtering on cells
    if filtering_cells:
        logging.info(f"Filtering {x.n_obs} cells")

    if filter_cell_min_counts is not None:
        sc.pp.filter_cells(
            x,
            min_counts=ensure_count(
                filter_cell_min_counts, max_value=np.max(x.obs["n_counts"])
            ),
        )
        logging.info(f"Remaining cells after min count: {x.n_obs}")
    if filter_cell_max_counts is not None:
        sc.pp.filter_cells(
            x,
            max_counts=ensure_count(
                filter_cell_max_counts, max_value=np.max(x.obs["n_counts"])
            ),
        )
        logging.info(f"Remaining cells after max count: {x.n_obs}")
    if filter_cell_min_genes is not None:
        sc.pp.filter_cells(
            x,
            min_genes=ensure_count(
                filter_cell_min_genes, max_value=np.max(x.obs["n_genes"])
            ),
        )
        logging.info(f"Remaining cells after min genes: {x.n_obs}")
    if filter_cell_max_genes is not None:
        sc.pp.filter_cells(
            x,
            max_genes=ensure_count(
                filter_cell_max_genes, max_value=np.max(x.obs["n_genes"])
            ),
        )
        logging.info(f"Remaining cells after max genes: {x.n_obs}")

    # Perform filtering on genes
    if filtering_genes:
        logging.info(f"Filtering {x.n_vars} vars")
    if filter_gene_min_counts is not None:
        sc.pp.filter_genes(
            x,
            min_counts=ensure_count(
                filter_gene_min_counts, max_value=np.max(x.var["n_counts"])
            ),
        )
        logging.info(f"Remaining vars after min count: {x.n_vars}")
    if filter_gene_max_counts is not None:
        sc.pp.filter_genes(
            x,
            max_counts=ensure_count(
                filter_gene_max_counts, max_value=np.max(x.var["n_counts"])
            ),
        )
        logging.info(f"Remaining vars after max count: {x.n_vars}")
    if filter_gene_min_cells is not None:
        sc.pp.filter_genes(
            x,
            min_cells=ensure_count(
                filter_gene_min_cells, max_value=np.max(x.var["n_cells"])
            ),
        )
        logging.info(f"Remaining vars after min cells: {x.n_vars}")
    if filter_gene_max_cells is not None:
        sc.pp.filter_genes(
            x,
            max_cells=ensure_count(
                filter_gene_max_cells, max_value=np.max(x.var["n_cells"])
            ),
        )
        logging.info(f"Remaining vars after max cells: {x.n_vars}")
    return x
