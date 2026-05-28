import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm
import scib
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from torch.autograd import Variable
import argparse

def parse_segment(segment):
    chrom, positions = segment.split('-')[0], '-'.join(segment.split('-')[1:])
    start, end = map(int, positions.split('-'))
    return chrom, start, end

def find_enhancer_overlaps(df, segment):
    chrom, start, end = parse_segment(segment)

    filtered = df[df['chrom'] == chrom]
    # 检查重叠：片段起始位置小于等于 peak 结束，且片段结束位置大于等于 peak 起始
    # overlaps = filtered[((filtered['start'] <= end) & (filtered['end'] >= start)) | ((filtered['start2'] <= end) & (filtered['end2'] >= start))]
    overlaps = filtered[(filtered['start'] <= end) & (filtered['end'] >= start)]
    return overlaps

def find_enhancer_overlaps_loop(df, segment):
    chrom, start, end = parse_segment(segment)

    filtered = df[df['chrom'] == chrom]
    # 检查重叠：片段起始位置小于等于 peak 结束，且片段结束位置大于等于 peak 起始
    overlaps = filtered[((filtered['start'] <= end+4000) & (filtered['end'] >= start-4000)) | ((filtered['start2'] <= end+4000) & (filtered['end2'] >= start-4000))]
    # overlaps = filtered[(filtered['start'] <= end) & (filtered['end'] >= start)]
    return overlaps

# 找overlap的基因
def find_nearest_gene(final_df,segment):
    chrom, positions = segment.split('-')[0], '-'.join(segment.split('-')[1:])
    start, end = map(int, positions.split('-'))

    # 筛选与片段重叠的基因
    relevant_genes = final_df[(final_df['seqname'] == chrom) &
                               (final_df['start'] <= end) & 
                               (final_df['end'] >= start) &
                               (final_df['feature'] == 'gene')] 

    # 如果没有找到任何基因，返回 None
    if relevant_genes.empty:
        # 如果没有找到任何基因，寻找距离最近的基因
        nearest_genes = final_df[(final_df['seqname'] == chrom) & 
                                (final_df['feature'] == 'gene')]

        # 计算到片段的开始和结束位置的距离
        nearest_genes['distance_start'] = (nearest_genes['start'] - start).abs()
        nearest_genes['distance_end'] = (nearest_genes['end'] - end).abs()

        # 找到最近的基因（最小距离）
        relevant_genes = nearest_genes.loc[nearest_genes[['distance_start', 'distance_end']].min(axis=1).idxmin()]
        return [relevant_genes['gene_name']], relevant_genes['feature']
    return list(relevant_genes['gene_name']), relevant_genes['feature']

def check_tss_overlap(df, chrom_segment):
    # 解析染色体片段
    chrom, positions = chrom_segment.split('-')[0], '-'.join(chrom_segment.split('-')[1:])
    start_pos, end_pos = map(int, positions.split('-'))

    # 过滤出相关染色体的基因
    relevant_genes = df[(df['seqname'] == chrom) & (df['feature'] == 'gene')]

    # 根据链计算 TSS 位置
    relevant_genes['tss_position_start'] = relevant_genes.apply(
        lambda gene: gene['start']-3000 if gene['strand'] == '+' else gene['end'], axis=1
    )
    relevant_genes['tss_position_end'] = relevant_genes.apply(
        lambda gene: gene['start'] if gene['strand'] == '+' else gene['end'] + 3000, axis=1
    )

    # 检查给定片段是否与 TSS 重叠
    overlaps = relevant_genes[
        (start_pos <= relevant_genes['tss_position_end']) &
         (relevant_genes['tss_position_start'] <= end_pos)
    ]

    # 创建重叠结果
    result = overlaps['gene_name'].copy().values

    return result

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵, 即上文中的K
    Params: 
	    source: 源域数据(n * len(x))
	    target: 目标域数据(m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    
    batch_size = 200
    num_window = int(total0.shape[0]/batch_size)+1
    L2_dis = []
    for i in tqdm(range(num_window)):
        diff = (total0[i*batch_size:(i+1)*batch_size].cuda()-total1[i*batch_size:(i+1)*batch_size].cuda())
        diff.square_()
        L2_dis.append(diff.sum(2).cpu())
    L2_distance = torch.concatenate(L2_dis,dim=0)

    # L2_distance = ((total0-total1)**2).sum(2) 

    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
	    source: 源域数据(n * len(x))
	    target: 目标域数据(m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算


def MMD(adata):
    real = adata[adata.obs_names=='true_Cell'].obsm['X_pca']
    gen = adata[adata.obs_names=='gen_Cell'].obsm['X_pca']
    X = torch.Tensor(real)
    Y = torch.Tensor(gen)
    X,Y = Variable(X), Variable(Y)
    return mmd_rbf(X,Y)


def LISI(adata):
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    lisi = scib.me.ilisi_graph(adata, batch_key="batch", type_="knn")
    return lisi


def random_forest(adata, return_roc = False):
    real = adata[adata.obs_names=='true_Cell'].obsm['X_pca']
    sim = adata[adata.obs_names=='gen_Cell'].obsm['X_pca']

    data = np.concatenate((real,sim),axis=0)
    label = np.concatenate((np.ones((real.shape[0])),np.zeros((sim.shape[0]))))

    ##将训练集切分为训练集和验证集
    X_train,X_val,y_train,y_val = train_test_split(data, label,
                                                test_size = 0.25,random_state = 1)

    ## 使用随机森林对数据进行分类
    rfc1 = RandomForestClassifier(n_estimators = 1000, # 树的数量
                                max_depth= 5,       # 子树最大深度
                                oob_score=True,
                                class_weight = "balanced",
                                random_state=1)
    rfc1.fit(X_train,y_train)

    ## 可视化在验证集上的Roc曲线
    pre_y = rfc1.predict_proba(X_val)[:, 1]
    fpr_Nb, tpr_Nb, _ = roc_curve(y_val, pre_y)
    aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值
    if return_roc:
        return aucval, fpr_Nb, tpr_Nb
    return aucval

def norm_total(array, target_sum = 1e4):        
    current_sum = np.sum(array,axis=1)[:,None] if len(array.shape)>1 else np.sum(array)
    normalization_factor = target_sum / current_sum  
    normalized_array = array * normalization_factor  
    return normalized_array


def plot_mse_curves(mse_values,legend, x_list, layer, metric):
    # 获取第一个维度和第二个维度的大小
    n_curves, n_points = mse_values.shape
    
    # 创建一个绘图对象
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(12, 7))
    
    # 依次绘制每一条曲线
    for i in range(n_curves):
        plt.plot(x_list, mse_values[i], label=legend[i])  # 添加标签以便识别每条曲线
    
    # 添加图例、标题和标签
    plt.title(f"{metric} Curves layer {layer}")
    plt.xlabel("Time step / 100")
    plt.ylabel(f"{metric} Value with prev step")
    # 设置图例在图的外部
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 调整布局，使得图例不遮挡图
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # 显示图形
    # plt.show()
    plt.savefig(f'/stor/lep/workspace/multi_diffusion/MM-Diffusion/evaluate_script/figures/attention/map_info/{metric}_{layer}.pdf')


def calculate_mse(array):
    # 得到数组的第一个维度大小
    n, h, w, d = array.shape
    
    # 初始化存储结果的数组
    mse_values = np.zeros((n - 1, h))
    
    # 逐对计算前后张量的MSE
    for i in range(1, n):
        # 计算相邻两个张量之间每个位置的MSE
        mse = np.mean((array[i] - array[i - 1]) ** 2, axis=(1, 2))
        mse_values[i - 1] = mse
    
    return mse_values

def find_max_index(matrix, topk):
    # 展平矩阵
    flattened = matrix.flatten()

    # 获取排序后的索引并取后五个
    top_indices_flat = np.argsort(flattened)[-topk:]

    # 将一维索引转换为二维索引
    index_rna = np.unravel_index(top_indices_flat, matrix.shape)

    return index_rna

def calculate_entropy(matrix):
    # Flatten the matrix to calculate probabilities
    flat_matrix = matrix.flatten()
    
    # Normalize the matrix to create a probability distribution
    probabilities = flat_matrix / np.sum(flat_matrix)
    
    # Filter out zero probabilities to avoid log2(0)
    probabilities = probabilities[probabilities > 0]
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def calculate_entropies(array):
    # Initialize a 9x22 array to store entropies
    entropies = np.zeros((array.shape[0], array.shape[1]))
    
    # Iterate over each 128x128 matrix
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            entropies[i, j] = calculate_entropy(array[i, j])
    
    return entropies


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default=None,
        # required=True,
        help="Path to pretrained vae.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dream_training",
        action="store_true",
        help=(
            "Use the DREAM training method, which makes training more efficient and accurate at the ",
            "expense of doing an extra forward pass. See: https://arxiv.org/abs/2312.00210",
        ),
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor p (should be greater than 0; default=1.0, as suggested in the paper)",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args(args=[])

    return args

def norm_total(array, target_sum = 1e4):        
    current_sum = np.sum(array,axis=1)[:,None] if len(array.shape)>1 else np.sum(array)
    normalization_factor = target_sum / current_sum  
    normalized_array = array * normalization_factor  
    return normalized_array