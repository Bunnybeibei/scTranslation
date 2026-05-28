"""
Helpers for distributed training.
"""

import io
import os
import socket
import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).

GPUS_PER_NODE = 8

# def setup_dist(devices=None):
#     """
#     Setup a distributed process group.
#     """
#     global GPUS_PER_NODE
#     if dist.is_initialized():
#         return

#     if devices.startswith("G"):
#         GPUS_PER_NODE = int(devices[1:])
#         os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
#     else:
#         devices_list=devices.split(',')
#         GPUS_PER_NODE = len(devices_list)
#         os.environ["CUDA_VISIBLE_DEVICES"] =  f"{devices_list[MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE]}"

#     th.cuda.set_device(int(devices_list[MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE]))
#     print(int(devices_list[MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE]))
#     comm = MPI.COMM_WORLD

#     backend = "gloo" if not th.cuda.is_available() else "nccl"

#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)

#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)

#     dist.init_process_group(backend=backend, init_method="env://")
def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def setup_dist(devices=None):
    """
    torchrun兼容的分布式初始化函数 (支持单机/多机训练)
    """
    if dist.is_initialized():
        return

    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
    # 自动检测torchrun环境变量
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 设备分配逻辑
    if devices:
        if devices.startswith("G"):
            num_gpus = int(devices[1:])
            visible_devices = list(range(num_gpus))
        else:
            visible_devices = [int(x) for x in devices.split(",")]
        
        # 验证设备可用性
        assert th.cuda.device_count() >= len(visible_devices), \
            f"需要 {len(visible_devices)} GPU，但只有 {th.cuda.device_count()} 可用"
        
        # 设置可见设备 (需在进程启动前设置)
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_devices))
    else:
        visible_devices = list(range(th.cuda.device_count()))

    # 自动设备选择
    if visible_devices:
        device_id = visible_devices[local_rank % len(visible_devices)]
        th.cuda.set_device(device_id)
        print(f"Rank {rank} 使用 GPU {device_id}")

    # # 分布式初始化
    # if world_size == 1:
    #     return  # 单进程模式
    
    # 自动配置多机参数
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(_find_free_port()))
    
    dist.init_process_group(
        backend="nccl" if th.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
        init_method="env://"
    )

    # 验证分布式环境
    if dist.is_initialized():
        print(f"进程初始化完成 | Rank {rank}/{world_size} | 设备: {th.cuda.current_device()}")

def dev():
    """
    Get the device to use for torch.distributed.
    """

    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
       
    return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)
    



def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def cleanup_dist():
    """
    Clean up the distributed process group.
    """
