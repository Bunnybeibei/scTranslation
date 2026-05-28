"""Training and testing entry point for scDiffusion-X.

Wraps ``EperLuo/scDiffusion-X`` so it plugs into the benchmark's ``Runner``
interface. The underlying ``EncoderModel`` / ``MultimodalUNet`` /
``MultimodalDiffusion`` live under :mod:`Autoencoder` and
:mod:`DiffusionBackbone`; this module only orchestrates dataset construction,
training (autoencoder + diffusion) and translation inference.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist
import yaml

from sctranslation.metrics import scTranslation_eval
from sctranslation.metrics.utils import pca_neighbors
from sctranslation.transforms import (
    ADT_CLR_Transform,
    annotate_babel,
    binary_data,
    filter_features,
    filter_features_babel,
    identity,
    normalize_r,
    use_hvg,
)
from sctranslation.transforms.misc import Compose
from sctranslation.utils.logger import ConfigLogger

from .DiffusionBackbone import dist_util
from .DiffusionBackbone.multimodal_train_util import TrainLoop
from .DiffusionBackbone.resample import create_named_schedule_sampler
from .scdiffusionx_dataloader import load_training_data
from .scdiffusionx_modelling import EncoderEstimator, create_model_and_diffusion, generate


logger = logging.getLogger("scDiffusionX")
PACKAGE_DIR = Path(__file__).resolve().parent

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
    
class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DictToObject(value)
            setattr(self, key, value)
    
    def to_dict(self, k_list=None):
        """change the object to dict"""
        return {key: value.to_dict() if isinstance(value, DictToObject) else value for key, value in self.__dict__.items() if k_list is None or key in k_list}
    
    def add_dict_to_object(self, default_dict):
        """change the object to dict"""
        for k, v in default_dict.items():
            setattr(self, k, v)
    
    # if has no attrribute, return None
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return None

def load_default_yaml_files(config_folder=None):
    if config_folder is None:
        config_folder = os.path.join(os.path.dirname(__file__), "configs_ae")
    result = {}
    # Assrue not empty
    if not os.path.exists(config_folder):
        raise ValueError(f"Config folder {config_folder} does not exist.")
    if not os.path.isdir(config_folder):
        raise ValueError(f"Config folder {config_folder} is not a directory.")
    if not os.listdir(config_folder):
        raise ValueError(f"Config folder {config_folder} is empty.")
    for root, dirs, files in os.walk(config_folder):
        if 'default.yaml' in files:
            # Get the subfolder relative to config_folder
            subfolder = os.path.relpath(root, config_folder)
            # Full path to default.yaml
            yaml_path = os.path.join(root, 'default.yaml')
            # Optionally, load YAML content if needed
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            # Store by subfolder key
            result[subfolder] = yaml_content
    return result


class scDiffusionXRunner:
    """Drive training and inference for scDiffusion-X.

    Parameters
    ----------
    config
        Parsed configuration object (see :mod:`sctranslation.utils.config`).
    model_filename
        Root directory holding ``<dataset>/<seed>`` checkpoints (AE + diffusion).
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
        n_jobs: Optional[int] = -1,
    ) -> None:
        self.config = config
        self.model_filename = Path(model_filename) / data_name / str(random_seed)
        self.saved_path = Path(saved_path) / data_name / str(random_seed)
        self.random_seed = random_seed
        self.n_jobs = n_jobs

        self.model_filename.mkdir(parents=True, exist_ok=True)
        self.logger = ConfigLogger(self.model_filename / "log.txt")
        self.logger.info("Preparing to run scDiffusionX on dataset: %s", data_name)
        self.logger.log_config(config)

    @staticmethod
    def preprocessing_pipeline_r(config):
        """Create a preprocessing pipeline for RNA data."""
        transforms = [
            annotate_babel(),
            filter_features_babel('r'),
            use_hvg(n_top_genes=config.n_top_genes) if config.use_hvg_flag else identity(),
        ]

        return Compose(*transforms)

    @staticmethod
    def preprocessing_pipeline_a(config):
        """Create a preprocessing pipeline for ATAC data."""
        transforms = [
            binary_data(),
            annotate_babel(),
            filter_features_babel('a'),
            filter_features(fpeaks=config.fpeaks) if config.fpeaks != 0. else identity(),
        ]
        return Compose(*transforms)
    
    @staticmethod
    def preprocessing_pipeline_p(config):
        """Create a preprocessing pipeline for ATAC data."""
        transforms = [
            ADT_CLR_Transform(),
        ]
        return Compose(*transforms)

    def initialize_ae(self, train_sc=None, val_sc=None, test_sc=None, fix_str='', modal2='a', n_cat=None):
        ae_config_path = PACKAGE_DIR / "Autoencoder" / f"ae{fix_str}.yaml"
        with open(ae_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.logger.info("Autoencoder config loaded from %s", ae_config_path)
        cfg = DictToObject(cfg)
        cfg.logger.project = ""
        cfg.trainer.max_epochs = 300
        cfg.training_config.chekpoint_path = self.model_filename / "ae"
        return EncoderEstimator(cfg, train_sc, val_sc, modal2=modal2, n_cat=n_cat)

    def get_model_and_diffusion_defaults(self):
        res = dict(
            rna_dim=self.config.rna_dim,
            atac_dim=self.config.atac_dim,
            num_channels=self.config.num_channels,
            num_res_blocks=self.config.num_res_blocks,
            num_heads=self.config.num_heads,
            num_heads_upsample=self.config.num_heads_upsample,
            num_head_channels=self.config.num_head_channels,
            cross_attention_resolutions=self.config.cross_attention_resolutions,
            cross_attention_windows=self.config.cross_attention_windows,
            cross_attention_shift=self.config.cross_attention_shift,
            channel_mult=self.config.channel_mult,
            dropout=self.config.dropout,
            class_cond=self.config.class_cond,
            use_checkpoint=self.config.use_checkpoint,
            use_scale_shift_norm=self.config.use_scale_shift_norm,
            resblock_updown=self.config.resblock_updown,
            use_fp16=self.config.use_fp16,
            num_class=self.config.num_class,
        )

        diffusion = dict(
            learn_sigma=self.config.learn_sigma,
            diffusion_steps=self.config.diffusion_steps,
            noise_schedule=self.config.noise_schedule,
            timestep_respacing=self.config.timestep_respacing,
            use_kl=self.config.use_kl,
            predict_xstart=self.config.predict_xstart,
            rescale_timesteps=self.config.rescale_timesteps,
            rescale_learned_sigmas=self.config.rescale_learned_sigmas,
        )

        res.update(diffusion)
        return res

    
    def initialize_model(self, args, model_args, fix_str=''):
        encoder_cfg_path = PACKAGE_DIR / "DiffusionBackbone" / f"encoder_multimodal{fix_str}.yaml"
        args.encoder_config = str(encoder_cfg_path)
        self.logger.info("Encoder config loaded from %s", args.encoder_config)
        if fix_str == '_large':
            self.config.rna_dim[-1] = 150
            self.config.atac_dim[-1] = 200
        elif fix_str == '_small':
            self.config.rna_dim[-1] = 64
            self.config.atac_dim[-1] = 64
        model, diffusion = create_model_and_diffusion(**model_args)
        return model, diffusion
    
    def initialize_trainer(self, args, train_sc=None, val_sc=None, test_sc=None, model=None, diffusion=None):
        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        data = load_training_data(train_sc, args, dev=dist_util.dev())

        return TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            lr=args.lr,
            t_lr=args.t_lr,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            use_db=args.use_db,
            sample_fn=args.sample_fn,
            output_dir=args.output_dir,
            num_classes=args.num_class,
        )
        
    def train(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData,
    ) -> None:
        cell_type_list = set(list(train_sc.ad['r'].obs[self.config.condition].values) + list(val_sc.ad['r'].obs[self.config.condition].values) + list(test_sc.ad['r'].obs[self.config.condition].values))
        # sort the cell_type_list
        cell_type_list = sorted(list(cell_type_list))
        cell2id = {cell_type: i for i, cell_type in enumerate(cell_type_list)}
        # self.config.num_class = len(cell_type_list)
        
        if 'a' in train_sc.ad.keys():
            modal2 = 'a'
        elif 'p' in train_sc.ad.keys():
            modal2 = 'p'
        else:
            raise ValueError("No valid modality found in train_sc.")
        gene_num = train_sc.ad['r'].shape[1]
        peak_num = train_sc.ad[modal2].shape[1]
        if (gene_num >= 50000) or (peak_num>200000):
            fix_str = '_large'
        elif (gene_num <= 5000) or (peak_num<15000):
            fix_str = '_small'
        else:
            fix_str = ''
        self.config.ae_path = str(self.model_filename  /'ae' /'checkpoints'/'last.ckpt')
        # Train AE
        self.ae = self.initialize_ae(train_sc, val_sc, fix_str=fix_str, modal2=modal2, n_cat=self.config.num_class)
        if os.path.exists(self.config.ae_path):
            self.logger.info(f"Autoencoder checkpoint already exists at {self.config.ae_path}.")
            training_time_ae = 0.
        else:
            self.logger.info("Starting training ae...")
            start_time = time.time()
            self.ae.train()
            training_time_ae = time.time() - start_time
            self.logger.info(f"Training ae took {training_time_ae:.2f} seconds")
            self.logger.info(f"Training ae took {training_time_ae / 60:.2f} minutes")
            self.logger.info(f"Training ae took {training_time_ae / 3600:.2f} hours")

        # Train Diffusion model
        dist_util.setup_dist('0')
        model_args = self.get_model_and_diffusion_defaults()
        args = self.config
        
        self.config.seed = self.random_seed
        self.config.output_dir = str(self.model_filename /'diffusion')

        self.model, diffusion = self.initialize_model(self.config, model_args, fix_str=fix_str)
        trainer = self.initialize_trainer(args, train_sc, val_sc, test_sc, self.model, diffusion)
        start_time = time.time()
        trainer.run_loop()
        training_time_diffusion = time.time() - start_time
        self.logger.info(f"Training diffusion took {training_time_diffusion:.2f} seconds")
        self.logger.info(f"Training diffusion took {training_time_diffusion / 60:.2f} minutes")
        self.logger.info(f"Training diffusion took {training_time_diffusion / 3600:.2f} hours")
        
        training_time = training_time_ae + training_time_diffusion
        self.logger.info(f"Training took {training_time:.2f} seconds")
        self.logger.info(f"Training took {training_time / 60:.2f} minutes")
        self.logger.info(f"Training took {training_time / 3600:.2f} hours")
        
        # close ddp
        dist.barrier()
        dist.destroy_process_group()
        print("Distributed process group destroyed.")
        # close ddp
        self.logger.info("Training completed.")
        
    def norm_total(self, array, target_sum = 1e4):        
        current_sum = np.sum(array,axis=1)[:,None] if len(array.shape)>1 else np.sum(array)
        normalization_factor = target_sum / current_sum  
        normalized_array = array * normalization_factor  
        return normalized_array

    def test(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData,
    ) -> None:
        # Sampling defaults (separate from training-time config). ``batch_size``
        # for inference is taken from the user config; callers can raise it via
        # CLI/config when memory allows.
        self.config.sample_fn = 'ddpm'
        self.config.classifier_scale = 3.0
        self.config.is_strict = True

        # If a dataset is missing usable ``cell_type`` labels (e.g. some MCC
        # subsets), derive labels via Leiden clustering on the combined set so
        # the conditional-generation pathway has something to condition on.
        if any(train_sc.ad[m].obs["cell_type"].isna().all() for m in train_sc.ad):
            for modal in train_sc.ad.keys():
                adata = sc.concat([train_sc.ad[modal].copy(), val_sc.ad[modal].copy()])
                if "neighbors" not in adata.uns:
                    pca_neighbors(adata)
                adata = sc.tl.leiden(adata, copy=True, resolution=0.5)
                for sc_obj in (train_sc, val_sc, test_sc):
                    sc_obj.ad[modal].obs["cell_type"] = (
                        adata.obs["leiden"][sc_obj.ad[modal].obs_names].copy().to_numpy()
                    )
                self.logger.info(
                    "After clustering, the number of clusters is %d",
                    len(train_sc.ad[modal].obs["cell_type"].unique()),
                )
        
        if 'a' in train_sc.ad.keys():
            modal2 = 'a'
        elif 'p' in train_sc.ad.keys():
            modal2 = 'p'
        else:
            raise ValueError("No valid modality found in train_sc.")
        gene_num = train_sc.ad['r'].shape[1]
        peak_num = train_sc.ad[modal2].shape[1]
        if (gene_num >= 50000) or (peak_num>200000):
            fix_str = '_large'
        elif (gene_num <= 5000) or (peak_num<15000):
            fix_str = '_small'
        else:
            fix_str = ''
        self.logger.info(f"Fix string: {fix_str}")
        # Train Diffusion model
        dist_util.setup_dist('0')
        model_args = self.get_model_and_diffusion_defaults()
        args = self.config
        self.config.ae_path = str(self.model_filename /'ae' /'checkpoints'/'last.ckpt')
        self.config.seed = self.random_seed
        self.config.output_dir = str(self.model_filename /'diffusion')
        multimodal_model, multimodal_diffusion = self.initialize_model(self.config, model_args, fix_str=fix_str)
        
        if os.path.isdir(args.output_dir):
            multimodal_name_list = [model_name for model_name in os.listdir(args.output_dir) if ((model_name.startswith('model') and model_name.endswith('.pt')))]
            multimodal_name_list.sort() # sort by name
            multimodal_name_list = [args.output_dir + f'/{model_name}'  for model_name in multimodal_name_list[::1]]
        else:
            multimodal_name_list = [model_path for model_path in self.config.output_dir.split(',')]

        # for model_path in multimodal_name_list:
        # model_path is the newest model
        model_path = multimodal_name_list[-1]
        multimodal_model.load_state_dict_(
            dist_util.load_state_dict(model_path, map_location="cpu"), is_strict=True
        )
        
        multimodal_model.to(dist_util.dev())

        adata_rna = test_sc.ad['r']
        adata_modal2 = test_sc.ad[modal2]
        
        self.logger.info("Starting testing...")
        start_time = time.time()
        
        evaluator = scTranslation_eval(model='scDiffusionX', saved_path=self.saved_path)
        # gen_mode is the *target* modality for translation; mapping is symmetric
        # in the diffusion backbone where "atac" denotes the second branch.
        self.config.gen_mode = 'atac' if modal2 == 'a' else 'adt'
        if adata_rna.shape[0] <= 10000:
            _, reconstruct2 = generate(multimodal_model, multimodal_diffusion, adata_rna=adata_rna, adata_modal2=adata_modal2, modal2=modal2, args=self.config, saved_path=self.saved_path, model_path=model_path)
            torch.cuda.empty_cache()
        else:
            chunk = 10000
            reconstruct2 = []
            for i in range(0, adata_rna.shape[0], chunk):
                adata_rna_chunk = adata_rna[i:i+chunk].copy()
                adata_modal2_chunk = adata_modal2[i:i+chunk].copy()
                _, reconstruct2_chunk = generate(multimodal_model, multimodal_diffusion, adata_rna=adata_rna_chunk, adata_modal2=adata_modal2_chunk, modal2=modal2, args=self.config, saved_path=self.saved_path, model_path=model_path)
                reconstruct2.append(reconstruct2_chunk)
                torch.cuda.empty_cache()
            reconstruct2 = np.concatenate(reconstruct2, axis=0)
        pred_modal2 = sc.AnnData(
            reconstruct2,
            var=adata_modal2.var,
            obs=adata_modal2.obs,
        )
        if modal2 == 'a':
            pred_modal2.X = self.norm_total(pred_modal2.X)
            modal2_X = test_sc.ad[modal2].X
            if hasattr(modal2_X, "toarray"):
                modal2_X = modal2_X.toarray()
            test_sc.ad[modal2].X = self.norm_total(modal2_X)
        evaluator.add_data(pred=pred_modal2, truth=test_sc.ad[modal2], name='r2a' if modal2 == 'a' else 'r2p')
        
        # self.config.diffusion_steps = 1000 if modal2 == 'a' else 100
        if modal2 == 'a':
            self.config.gen_mode = 'rna'
            if adata_rna.shape[0] <= 10000:
                reconstruct, _ = generate(multimodal_model, multimodal_diffusion, adata_rna=adata_rna, adata_modal2=adata_modal2, modal2=modal2, args=self.config, saved_path=self.saved_path, model_path=model_path)
            else:
                chunk = 10000
                reconstruct = []
                for i in range(0, adata_rna.shape[0], chunk):
                    adata_rna_chunk = adata_rna[i:i+chunk].copy()
                    adata_modal2_chunk = adata_modal2[i:i+chunk].copy()
                    reconstruct_chunk, _ = generate(multimodal_model, multimodal_diffusion, adata_rna=adata_rna_chunk, adata_modal2=adata_modal2_chunk, modal2=modal2, args=self.config, saved_path=self.saved_path, model_path=model_path)
                    reconstruct.append(reconstruct_chunk)
                    torch.cuda.empty_cache()
                reconstruct = np.concatenate(reconstruct, axis=0)
            pred_r = sc.AnnData(
                reconstruct,
                var=adata_rna.var,
                obs=adata_rna.obs,
            )
            pred_r.X = self.norm_total(pred_r.X)
            rna_X = test_sc.ad['r'].X
            if hasattr(rna_X, "toarray"):
                rna_X = rna_X.toarray()
            test_sc.ad['r'].X = self.norm_total(rna_X)
            evaluator.add_data(pred=pred_r, truth=test_sc.ad['r'], name='a2r' if modal2 == 'a' else 'p2r')
        
        evaluator.forward()
        test_time = time.time() - start_time
        self.logger.info(f"Test took {test_time:.2f} seconds")
        self.logger.info(f"Test took {test_time:.2f} seconds")
        self.logger.info(f"Test took {test_time / 60:.2f} minutes")
        self.logger.info(f"Test took {test_time / 3600:.2f} hours")
        
        print("Save the results")