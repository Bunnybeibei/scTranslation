import scanpy as sc
from sklearn import metrics
from .utils import MMD, LISI, cal_auroc_correlation, pca_neighbors, get_X_array
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import os
import pandas as pd
from pathlib import Path
import numpy as np


def calculate_cluster_index(raw_adata: sc.AnnData):
    """
    Evaluate clustering performance metrics using Leiden clustering.
    
    Parameters
    ----------
    raw_adata : sc.AnnData
        AnnData object containing ground truth labels in obs['cell_type']
    
    Returns
    -------
    tuple
        (ARI, AMI, NMI, HOM) clustering metrics:
        - ARI: Adjusted Rand Index
        - AMI: Adjusted Mutual Information
        - NMI: Normalized Mutual Information
        - HOM: Homogeneity Score
    """
    adata = raw_adata.copy()
    if "neighbors" not in adata.uns:
        pca_neighbors(adata)

    adata = sc.tl.leiden(adata, copy=True)

    ARI = metrics.adjusted_rand_score(adata.obs["cell_type"], adata.obs["leiden"])
    AMI = metrics.adjusted_mutual_info_score(adata.obs["cell_type"], adata.obs["leiden"])
    NMI = metrics.normalized_mutual_info_score(adata.obs["cell_type"], adata.obs["leiden"])
    HOM = metrics.homogeneity_score(adata.obs["cell_type"], adata.obs["leiden"])

    return ARI, AMI, NMI, HOM

def calculate_expression_index(pred: sc.AnnData, truth: sc.AnnData):
    """
    Calculate expression-level evaluation metrics.
    
    Parameters
    ----------
    pred : sc.AnnData
        Predicted expression data
    truth : sc.AnnData
        Ground truth expression data
    
    Returns
    -------
    tuple
        (PCC, SPCC, MSE, MAE) metrics:
        - PCC: Pearson correlation coefficient
        - SPCC: Spearman correlation coefficient
        - MSE: Mean Squared Error
        - MAE: Mean Absolute Error
    """
    y_true = get_X_array(truth)
    y_pred = get_X_array(pred)

    pcc_sum = spearman_sum = mse_sum = mae_sum = 0
    nan_count = 0

    for ob in range(len(y_true)):
        pcc = pearsonr(y_true[ob], y_pred[ob])[0]
        if not np.isnan(pcc):
            pcc_sum += pcc
        else:
            nan_count += 1
        spcc = spearmanr(y_true[ob], y_pred[ob])[0]
        if not np.isnan(spcc):
            spearman_sum += spcc
        mse = mean_squared_error(y_true[ob], y_pred[ob])
        if not np.isnan(mse):
            mse_sum += mse
        mae = mean_absolute_error(y_true[ob], y_pred[ob])
        if not np.isnan(mae):
            mae_sum += mae

    return (
        pcc_sum/(len(y_true)-nan_count),
        spearman_sum/(len(y_true)-nan_count),
        mse_sum/len(y_true),
        mae_sum/len(y_true)
    )

def calculate_generate_index(raw_pred: sc.AnnData, raw_truth: sc.AnnData, n_components: int = None):
    """
    Evaluate generation quality metrics.
    
    Parameters
    ----------
    raw_pred : sc.AnnData
        Generated/predicted data
    raw_truth : sc.AnnData
        Ground truth data
    
    Returns
    -------
    tuple
        (MMD, LISI) metrics:
        - MMD: Maximum Mean Discrepancy
        - LISI: Local Inverse Simpson's Index
    """
    pred = raw_pred.copy()
    truth = raw_truth.copy()
    all_data = truth.concatenate(pred, join="outer", batch_key=None)
    all_data.obs["batch"] = ['true_cell'] * len(truth.obs) + ['gen_cell'] * len(pred.obs)
    
    if "neighbors" not in all_data.uns:
        pca_neighbors(all_data, n_components=n_components)
        
    return MMD(all_data), LISI(all_data)

def calculate_auroc(truth: sc.AnnData, pred: sc.AnnData):
    """Calculate AUROC and correlation metrics"""
    return cal_auroc_correlation(get_X_array(truth), get_X_array(pred))

def draw_tsne(data, title='a2r', color='cell_type'):
    """
    Visualize data using t-SNE
    
    Parameters
    ----------
    data : sc.AnnData
        Input data for visualization
    title : str, optional
        Figure title
    color : str, optional
        Observation key for coloring
    
    Returns
    -------
    matplotlib.figure.Figure
        Generated t-SNE plot
    """
    sc.settings.set_figure_params(dpi=120, facecolor='white')
    pca_neighbors(data)
    sc.tl.tsne(data)
    return sc.pl.tsne(data, color=color, title=title, return_fig=True)

class scTranslation_eval:
    """Single-cell translation evaluation framework"""
    
    def __init__(self, metric_names=None, model='scButterfly', saved_path=None):
        self.metric_names = metric_names or [
            'ARI', 'AMI', 'NMI', 'HOM',
            'PCC', 'SPCC',
            'MSE', 'MAE', 
            'MMD', 'LISI', 
            'AUROC', 'CORR'
        ]
        self.data = {}
        self.result = {}
        self.model = model
        self.saved_path = saved_path
    
    def add_data(self, pred, truth, name: str = 'r2a'):
        """Register data pair for evaluation"""
        # if truth.obs['cell_type'].isna().any():
        #     index = truth.obs["cell_type"].notna()
        #     print(f'Before filtering, truth shape: {truth.shape}, pred shape: {pred.shape}')
        #     truth = truth[index]
        #     pred = pred[index]
        #     print(f'After filtering NaN in cell_type, truth shape: {truth.shape}, pred shape: {pred.shape}')
        #     assert pred.shape[0] > 0
        self.data[name] = {'pred': pred, 'truth': truth}
        self.result[name] = []
        
        # save the data
        if self.saved_path:
            self.saved_path.mkdir(parents=True, exist_ok=True) 
            pred_file = self.saved_path/ f'test_{name}_pred.h5ad'
            truth_file = self.saved_path/ f'test_{name}_truth.h5ad'
            if not os.path.exists(pred_file):
                pred.write(str(pred_file))
            else:
                print(f"File {pred_file} already exists, skipping write.")
            if not os.path.exists(truth_file):
                truth.write(truth_file)
            else:
                print(f"File {truth_file} already exists, skipping write.")
                
        # evaluator.forward()
        pcc_list = []
        for i in range(self.data[name]['pred'].shape[1]):
            pcc = np.corrcoef(get_X_array(self.data[name]['pred'])[:, i], get_X_array(self.data[name]['truth'])[:, i])[0, 1]
            pcc_list.append(pcc)
        print("pcc_list: ", pcc_list)   
        print("mean pcc: ", np.nanmean(pcc_list))
        self.data[name]['pred'].var['pcc'] = pcc_list
        self.data[name]['truth'].var['pcc'] = pcc_list
    
        # pred_x = pred_r
        # truth_x = test_sc.ad['r'].X.A
        # # calcluate pcc between pred_x[:, i], truth_x[:, i]
        # pcc_list = []
        # for i in range(pred_x.shape[1]):
        #     pcc = np.corrcoef(pred_x[:, i], truth_x[:, i])[0, 1]
        #     pcc_list.append(pcc)
        # print("pcc_list: ", pcc_list)   
        # print("mean pcc: ", np.nanmean(pcc_list))

    def forward(self, return_list=False):
        """Execute full evaluation pipeline"""
        for name, data_pair in self.data.items():
            pred = data_pair['pred']
            truth = data_pair['truth']
            
            # if len(adata.obs["cell_type"].unique()) == 1, use leiden clustering truth
            if len(truth.obs["cell_type"].unique()) == 1:
                adata = truth.copy()
                if "neighbors" not in adata.uns:
                    pca_neighbors(adata)
                adata = sc.tl.leiden(adata, copy=True)
                truth.obs["cell_type"] = adata.obs['leiden'].copy().to_numpy()
                pred.obs["cell_type"] = adata.obs['leiden'].copy().to_numpy()
                print(f"After clustering, the number of clusters is {len(truth.obs['cell_type'].unique())}")
        
            # Filter NaN and inf row in pred
            filter_index = np.isfinite(get_X_array(pred)).all(axis=1)
            pred = pred[filter_index]
            truth = truth[filter_index]
            filter_index = np.isfinite(get_X_array(truth)).all(axis=1)
            truth = truth[filter_index]
            pred = pred[filter_index]
            print(f"After filtering NaN and inf, pred shape: {pred.shape}, truth shape: {truth.shape}")

            # Clustering metrics
            self.result[name].extend(calculate_cluster_index(raw_adata=pred))
            # Expression metrics
            self.result[name].extend(calculate_expression_index(truth=truth, pred=pred))
            # Generation metrics
            self.result[name].extend(calculate_generate_index(raw_truth=truth, raw_pred=pred))
            # Classification metrics
            self.result[name].extend(calculate_auroc(truth=truth, pred=pred))
        
        if return_list:
            return self.result
        else:
            self._save_results()
    
    def _save_results(self):
        """Save evaluation results to CSV"""
        os.makedirs(self.saved_path, exist_ok=True)
        
        df = pd.DataFrame.from_dict(
            self.result, 
            orient='index',
            columns=self.metric_names
        )
        csv_path = os.path.join(self.saved_path, 'evaluation_results3.csv')
        df.to_csv(csv_path, float_format='%.4f')
        print(f"Evaluation results saved to {csv_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run scTranslation evaluation on a single result directory.")
    parser.add_argument("--root", required=True, help="Folder containing test_<task>_pred.h5ad / test_<task>_truth.h5ad pairs.")
    parser.add_argument("--task", default="a2r", help="Translation task name (e.g. a2r, r2a, r2p).")
    args = parser.parse_args()

    root = Path(args.root)
    truth = sc.read_h5ad(root / f"test_{args.task}_truth.h5ad")
    pred = sc.read_h5ad(root / f"test_{args.task}_pred.h5ad")

    evaluator = scTranslation_eval(saved_path=root)
    evaluator.add_data(pred=pred, truth=truth, name=args.task)
    evaluator.forward()
