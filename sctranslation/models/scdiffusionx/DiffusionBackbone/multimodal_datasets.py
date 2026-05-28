from distutils.spawn import spawn
import torch.distributed as dist
import blobfile as bf
import numpy as np
import torch as th
import os
import pickle
import torch as th
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import yaml

import muon as mu
from muon import MuData

from ..Autoencoder.models.base.encoder_model import EncoderModel

def load_data_cell(
    *,
    batch_size,
    train_data=None,
    ae_path=None,
    rna_dim=0,
    atac_dim=0,
    deterministic=False,
    random_flip=True,
    num_workers=0,
    frame_gap=1,
    drop_last=True,
    condition='cell_type',
    encoder_config='encoder_multimodal',
    dev="cuda:0",
):
    """
    For a dataset, create a generator over (audio-video) pairs.

    Each video is an NxFxCxHxW float tensor, each audio is an NxCxL float tensor
   
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param rna_dim: the size to which video frames are resized.
    :atac_dim:the size to which audio are resized.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if train_data is None:
        raise ValueError("unspecified data directory")

    dataset = MultimodalDataset_cell(
        train_data = train_data,
        ae_path=ae_path,
        condition=condition,
        encoder_config=encoder_config,
        dev=dev,
    )
    
    
    if deterministic:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler,
    )
        
    while True:
        yield from loader


def _list_video_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["avi", "gif", "mp4"]:
           
            results.append(full_path)
        elif bf.isdir(full_path):
            
            results.extend(_list_video_files_recursively(full_path))
    return results


class MultimodalDataset_cell(Dataset):
    """
    :param rna_dim: [F,3,H,W] the size to which video frames are resized.
    :param atac_dim: [C,L] the size to which audio are resampled.
    :param video_clips: the meta info package of video clips. 
    :param shard: GPU id, used for allocating videos to different GPUs.
    :param num_shards: GPU num, used for allocating videos to different GPUs.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param audio_fps: the fps of audio.
    """
    def __init__(
        self,
        train_data,
        ae_path=None,
        condition='cell_type',
        encoder_config='encoder_multimodal',
        dev="cuda:0",
    ):
        super().__init__()

        self.condition = condition
        adata_rna = train_data.ad['r']
        if 'a' in train_data.ad:
            adata_atac = train_data.ad['a']
            modality2 = 'atac'
        elif 'p' in train_data.ad:
            adata_atac = train_data.ad['p']
            modality2 = 'adt'
        else:
            raise KeyError(
                "train_data.ad must contain either 'a' (ATAC) or 'p' (ADT)"
            )
        # Joint filtering: keep only cells with a valid ``cell_type`` in BOTH
        # modalities so the paired tensors stay aligned (RNA and modal-2 may
        # have NaN labels in different rows otherwise).
        rna_valid = adata_rna.obs.index[~adata_rna.obs["cell_type"].isna()]
        modal2_valid = adata_atac.obs.index[~adata_atac.obs["cell_type"].isna()]
        common = rna_valid.intersection(modal2_valid)
        adata_rna = adata_rna[common].copy()
        adata_atac = adata_atac[common].copy()
        try:
            celltype_num = np.unique(adata_rna.obs[condition]).shape[0]
        except:
            adata_rna.obs[condition] = adata_rna.obs[condition].astype(str)
            celltype_num = np.unique(adata_rna.obs[condition]).shape[0]

        labels = adata_rna.obs[condition].values
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        self.classes = label_encoder.transform(labels)

        print("loading encoder and processing data...")
        self.adata_rna, self.adata_atac, self.rna_std10, self.atac_std10 = self.encode_raw_data(ae_path, adata_rna, adata_atac, celltype_num,encoder_config,dev,modality2=modality2)
        # Write the second-modality std under BOTH a modality-aware key
        # (``adt_std`` / ``atac_std``) and the legacy generic key ``atac_std``
        # so older checkpoints continue to load.
        modal2_std_np = self.atac_std10.cpu().detach().numpy()
        save_kwargs = {
            "rna_std": self.rna_std10.cpu().detach().numpy(),
            "atac_std": modal2_std_np,  # legacy key (always second modality)
            f"{modality2}_std": modal2_std_np,  # modality-specific key
            "modal2": np.array(modality2),
        }
        np.savez('/'.join(ae_path.split('/')[:-2])+'/norm_factor.npz', **save_kwargs)
        print("done!")

    def encode_raw_data(self, ae_path, adata_rna, adata_atac,celltype_num,encoder_config,dev, modality2='atac'):
        with open(encoder_config, 'r') as file:
            yaml_content = file.read()
        autoencoder_args = yaml.safe_load(yaml_content)

        in_dim = {modality2: adata_atac.shape[1], 'rna': adata_rna.shape[1]}
        # Initialize encoder
        encoder_model = EncoderModel(in_dim=in_dim,
                                            n_cat=celltype_num,
                                            conditioning_covariate=self.condition, 
                                            encoder_type='learnt_autoencoder',
                                            **autoencoder_args)
        
        # Load weights 
        weight = torch.load(ae_path, map_location=torch.device(dev), weights_only=False)["state_dict"]
        new_weight = {}
        for k in encoder_model.state_dict().keys():
            if k not in weight.keys():
                print('Remove key {} from encoder_model.state_dict()'.format(k))
            else:
                new_weight[k] = weight[k]
        encoder_model.load_state_dict(new_weight)
        encoder_model.eval()

        rna = []
        atac = []
        adt = []
        bs = 1000
        batch_num = int(adata_rna.shape[0]/bs)+1
        for i in range(batch_num):
            batch = {}
            batch["X_norm"] = {'rna':torch.tensor(adata_rna[i*bs:(i+1)*bs].X.toarray(),dtype=encoder_model.dtype),modality2:torch.tensor(adata_atac[i*bs:(i+1)*bs].X.toarray(),dtype=encoder_model.dtype)}
            
            X = {mod: batch["X_norm"][mod].to(encoder_model.device) for mod in batch["X_norm"]}
            size_factor = {}
            for mod in X:
                size_factor_mod = X[mod].sum(1).unsqueeze(1).to(encoder_model.device)
                size_factor[mod] = size_factor_mod

            z = encoder_model.encode(batch)
            rna.append(z['rna'])
            atac.append(z[modality2])
        
        # rescaling into std = 1
        rna = torch.concat(rna)
        atac = torch.concat(atac)
        rna_std10 = rna.std(0).mean()*10
        atac_std10 = atac.std(0).mean()*10
        return (rna/rna_std10).unsqueeze(1).cpu().detach().numpy(), (atac/atac_std10).unsqueeze(1).cpu().detach().numpy(), rna_std10, atac_std10
        

    def __len__(self):
        return self.adata_rna.shape[0]

    def get_item(self, idx):
   
        rna = self.adata_rna[idx]
        atac = self.adata_atac[idx]
        
        return rna, atac, self.classes[idx]
    
    def __getitem__(self, idx):
        video_after_process, audio, class_num = self.get_item(idx)

        return video_after_process, audio, class_num


if __name__=='__main__':
    from einops import rearrange
    import torch.nn.functional as F

    audio_fps=16000
    video_fps= 10
    batch_size=4
    seconds = 1.6
    image_resolution=64

    dataset64=load_data(
    data_dir="/data6/rld/data/landscape/test",
    batch_size=batch_size,
    rna_dim=[int(seconds*video_fps), 3, 64, 64],
    atac_dim=[1, int(seconds*audio_fps)],
    frame_gap=1,
    random_flip=False,
    num_workers=0,
    deterministic=True,
    video_fps=video_fps,
    audio_fps=audio_fps
    )

  
    group = 0

    while True:    
        group += 1
        batch_video, batch_audio,  cond= next(dataset64)
   
