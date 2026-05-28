import numpy as np
import torch
from tqdm import tqdm
import scib
from sklearn.metrics import *
from torch.autograd import Variable
import scanpy as sc
from sklearn.feature_selection import r_regression
from scipy import sparse

def get_X_array(adata: sc.AnnData):
    if sparse.issparse(adata.X):
        # if sparse matrix, use adata.X.A
        return adata.X.A
    else:
        # if not sparse matrix, use adata.X
        return adata.X

def pca_neighbors(adata: sc.AnnData, n_components:int=None):
    """
    PCA and neighbors of adata.
    Parameters
    ----------
    adata: Anndata
        Anndata need to cumpute index, there need ground truth labels "cell_type" in adata.obs.
    n_components: int
        Number of components to use for PCA.

    Returns
    ----------
    adata: Anndata
        Anndata with PCA and neighbors.
    """
    if n_components is None:
        n_components = min(50, adata.shape[1]-1)
    
    if adata.X.dtype == "int32" or adata.X.dtype == "int64":
        # if adata.X is integer, normalize it
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    sc.pp.pca(adata, n_comps=n_components)
    sc.pp.neighbors(adata)


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Convert source and target domain data into kernel matrices.
    
    Parameters:
        source: Source domain data (n * len(x))
        target: Target domain data (m * len(y))
        kernel_mul: 
        kernel_num: Number of different Gaussian kernels to use
        fix_sigma: Sigma values for different Gaussian kernels
        
    Returns:
        sum(kernel_val): Sum of multiple kernel matrices
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])  # Get the total number of samples
    total = torch.cat([source, target], dim=0)  # Concatenate source and target data
    
    # Create a matrix where each row of total is repeated (n+m) times
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    
    # Create a matrix where each row of total is repeated (n+m) times in a different order
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    
    # Calculate the L2 distance between any two data points
    batch_size = 200
    num_window = int(total0.shape[0] / batch_size) + 1
    L2_dis = []
    for i in tqdm(range(num_window)):
        diff = (total0[i * batch_size:(i + 1) * batch_size] - total1[i * batch_size:(i + 1) * batch_size])
        diff.square_()
        L2_dis.append(diff.sum(2).cpu())
    L2_distance = torch.cat(L2_dis, dim=0)

    # Adjust the sigma value for the Gaussian kernel
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    # Calculate a list of bandwidths with kernel_mul as the multiplier
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    # Calculate the Gaussian kernel values
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    # Return the sum of all kernel matrices
    return sum(kernel_val)  # / len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Calculate the Maximum Mean Discrepancy (MMD) distance between source and target domain data.
    
    Parameters:
        source: Source domain data (n * len(x))
        target: Target domain data (m * len(y))
        kernel_mul: 
        kernel_num: Number of different Gaussian kernels to use
        fix_sigma: Sigma values for different Gaussian kernels
        
    Returns:
        loss: MMD loss
    '''
    batch_size = int(source.size()[0])  # Typically the batch size of both source and target
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    
    # Divide the kernel matrix into four parts
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    # Calculate the MMD loss
    loss = torch.mean(XX + YY - XY - YX)
    return loss.item()


def MMD(adata):
    '''
    Calculate the MMD distance between real and generated cells in the PCA space.
    
    Parameters:
        adata: AnnData object containing both real and generated cells
    
    Returns:
        MMD distance
    '''
    real = adata[adata.obs["batch"] == 'true_cell'].obsm['X_pca']
    gen = adata[adata.obs["batch"] == 'gen_cell'].obsm['X_pca']
    X = torch.Tensor(real).detach().cpu()
    Y = torch.Tensor(gen).detach().cpu()
    X, Y = Variable(X), Variable(Y)
    return mmd_rbf(X, Y)


def LISI(adata):
    '''
    Calculate the Local Inverse Similarity Index (LISI) for the given AnnData object.
    
    Parameters:
        adata: AnnData object
    
    Returns:
        LISI value
    '''
    lisi = scib.me.ilisi_graph(adata, batch_key="batch", type_="knn")
    return lisi

def cal_auroc(imputed_data, data, max_features=100_000, return_statistic=True):
    """Plot AUROC by feature for imputation on binarized data"""
    total_features = min(data.shape[1], max_features)
    # Samples by default
    feat_idx = np.random.choice(data.shape[1], total_features, replace=False)

    feat_auc = []
    pred = imputed_data
    true = data; true = 1 * (true > np.median(true))
    fpr, tpr, _thresholds = roc_curve(true.flatten(), pred.flatten())
    auc_score = auc(fpr, tpr)
    
    # temp = [] # true 特征全正或全负，为啥其他就行
    # for pr, tr in zip(np.transpose(pred)[feat_idx], np.transpose(true)[feat_idx]):
    #     if len(np.unique(tr)) == 2:
    #         import warnings
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             temp.append(roc_auc_score(tr, pr))
    #     feat_auc.append(temp)
    # if return_statistic:
    #     return np.mean(feat_auc[0])
    return auc_score

def cal_correlation(imputed_data, data, max_features=100_000, return_statistic=True):
    """Plot correlation by feature for imputed data"""
    total_features = min(data.shape[1], max_features)
    # Samples by default
    feat_idx = np.random.choice(data.shape[1], total_features, replace=False)

    feat_corr = []
    pred = imputed_data
    true = data

    temp = []
    for pr, tr in zip(np.transpose(pred)[feat_idx], np.transpose(true)[feat_idx]):
        if len(np.unique(tr)) > 1:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                temp.append(r_regression(np.reshape(pr, (-1, 1)), tr)[0])
            # p_per_feature.append(f_regression(predicted[:, [k]], actual[:, k])[1][0])
    feat_corr.append(temp)
    if return_statistic:
        # Filter NaN values
        corr = np.array(feat_corr[0])
        if not all(np.isnan(corr)):
            print('Attention: some features have no variance')
            corr = corr[np.isfinite(corr)]
        # Filter inf values
        if not all(np.isfinite(corr)):
            print('Attention: some features have inf values')
            corr = corr[np.isfinite(corr)]
        return np.mean(corr)
    
def cal_auroc_correlation(*args, **kwargs):
    """
    Calculate AUROC and correlation for imputed data."""
    return (
        cal_auroc(*args, **kwargs),
        cal_correlation(*args, **kwargs),
    )