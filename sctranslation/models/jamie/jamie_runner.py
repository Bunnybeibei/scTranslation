"""
Training and testing functionality for scButterfly models.
"""

import logging
import sys
from typing import Optional
import numpy as np
from scipy.sparse import hstack
import pandas as pd
import os
import time
import scanpy as sc
import torch

from sctranslation.transforms import scale, use_hvg, filter_features, identity, ADT_CLR_Transform

from sctranslation.transforms.misc import Compose
from sctranslation.metrics import scTranslation_eval
from sctranslation.utils.logger import ConfigLogger

from .jamie import JAMIE
from .jamie.utilities import hash_kwargs

from scipy.sparse import csr_matrix

from joblib import Parallel, delayed
from pathlib import Path

logger = logging.getLogger("JAMIE")


def init_logger(config):
    # Set up logging
    output = config.logger_save_path
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(output)
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)
    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class JAMIERunner:
    """A class to run JAMIE models.
    Parameters
    ----------
    config : Config object
        The JAMIE configuration.
    model_filename : str, optional
        The model filename is required for test modes,
        but not for training a model from scratch.
    saved_path : str, optional
        The path to save the model output.
    """

    def __init__(
        self,
        config,
        model_filename: Optional[str] = None,
        data_name: Optional[str] = None,
        saved_path: Optional[str] = "",
        random_seed: Optional[int] = 0,
    ) -> None:
        """Initialize a JAMIERunner."""
        self.model_filename = Path(model_filename) / data_name / str(random_seed)
        self.logger = ConfigLogger(self.model_filename /f'log.txt')
        self.logger.info("Preparing to run JAMIE on dataset: %s", data_name)
        self.logger.log_config(config)
        
        """Initialize a ModelRunner"""
        self.config = config
        self.saved_path = Path(saved_path) / data_name / str(random_seed)
        self.random_seed = random_seed

    @staticmethod
    def preprocessing_pipeline_r(config):
        """Create a preprocessing pipeline for RNA data."""
        transforms = [
            # scale(),
            # use_hvg(n_top_genes=config.n_top_genes)
        ]

        return Compose(*transforms)

    @staticmethod
    def preprocessing_pipeline_a(config):
        """Create a preprocessing pipeline for ATAC data."""
        transforms = [
            filter_features(fpeaks=config.fpeaks) if config.fpeaks != 0. else identity(),
            scale(),
        ]
        return Compose(*transforms)
    
    @staticmethod
    def preprocessing_pipeline_p(config):
        """Create a preprocessing pipeline for ATAC data."""
        transforms = [
            # ADT_CLR_Transform(),
        ]
        return Compose(*transforms)
    
    def train(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData,
    ) -> None:
        # If train_sc.ad['r'].X is a density matrix, convert it to a sparse matrix
        if isinstance(train_sc.ad['r'].X, np.ndarray):
            train_sc.ad['r'].X = csr_matrix(train_sc.ad['r'].X)
            test_sc.ad['r'].X = csr_matrix(test_sc.ad['r'].X)
        if 'a' in train_sc.ad:
            modal2 = 'a'
        elif 'p' in train_sc.ad:
            modal2 = 'p'
        else:
            raise ValueError("No valid modality found in train_sc.ad.")
        if isinstance(train_sc.ad[modal2].X, np.ndarray):
            train_sc.ad[modal2].X = csr_matrix(train_sc.ad[modal2].X)
            test_sc.ad[modal2].X = csr_matrix(test_sc.ad[modal2].X)
        dataset, model_str = self.initialize_model(train_sc=train_sc, test_sc=test_sc, train=True, modal2=modal2)
        self.model.manual_seed = self.random_seed
        # Start training
        self.logger.info("Starting training...")
        start_time = time.time()
        jm_data = self.model.fit_transform(dataset=[train_sc.ad['r'].X.A, train_sc.ad[modal2].X.A])
        training_time = time.time() - start_time
        self.logger.info(f"Training took {training_time:.2f} seconds")
        # Change to minutes and hours
        self.logger.info(f"Training took {training_time / 60:.2f} minutes")
        self.logger.info(f"Training took {training_time / 3600:.2f} hours")
        self.model.save_model(model_str)
    
    def test(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData,
    ) -> None:
        if isinstance(train_sc.ad['r'].X, np.ndarray):
            train_sc.ad['r'].X = csr_matrix(train_sc.ad['r'].X)
            test_sc.ad['r'].X = csr_matrix(test_sc.ad['r'].X)
        if 'a' in train_sc.ad:
            modal2 = 'a'
        elif 'p' in train_sc.ad:
            modal2 = 'p'
        else:
            raise ValueError("No valid modality found in train_sc.ad.")
        if isinstance(train_sc.ad[modal2].X, np.ndarray):
            train_sc.ad[modal2].X = csr_matrix(train_sc.ad[modal2].X)
            test_sc.ad[modal2].X = csr_matrix(test_sc.ad[modal2].X)
        dataset, model_str = self.initialize_model(train_sc=train_sc, test_sc=test_sc, train=False, modal2=modal2)
        self.logger.info("Starting testing...")
        start_time = time.time()
        R2A_predict = self.model.modal_predict(dataset[0], 0)[train_sc.ad['r'].shape[0]:]
        A2R_predict = self.model.modal_predict(dataset[1], 1)[train_sc.ad[modal2].shape[0]:] # [train, test]
        
        R2A_predict = sc.AnnData(X = R2A_predict)
        A2R_predict = sc.AnnData(X = A2R_predict)
        R2A_predict.obs = test_sc.ad[modal2].obs.copy()
        A2R_predict.obs = test_sc.ad['r'].obs.copy()
        R2A_predict.var = test_sc.ad[modal2].var.copy()
        A2R_predict.var = test_sc.ad['r'].var.copy()
        
        evaluator = scTranslation_eval(model='JAMIE', saved_path=self.saved_path)
        evaluator.add_data(pred=R2A_predict, truth=test_sc.ad[modal2], name=f'r2{modal2}')
        evaluator.add_data(pred=A2R_predict, truth=test_sc.ad['r'], name=f'{modal2}2r')
        evaluator.forward()
        test_time = time.time() - start_time
        self.logger.info(f"Test took {test_time:.2f} seconds")
        # Change to minutes and hours
        self.logger.info(f"Test took {test_time / 60:.2f} minutes")
        self.logger.info(f"Test took {test_time / 3600:.2f} hours")
    
    def initialize_model(self, train_sc, test_sc, train:bool=True, modal2='a') -> None:
        kwargs = {
            'output_dim': self.config.output_dim,
            'epoch_DNN': self.config.epoch_DNN,
            'min_epochs': self.config.min_epochs,
            'log_DNN': self.config.log_DNN,
            'use_early_stop': self.config.use_early_stop,
            'batch_size': self.config.batch_size,
            'pca_dim': self.config.pca_dim,
            'dist_method': self.config.dist_method,
            'loss_weights': self.config.loss_weights,
            'dropout': self.config.dropout,
        }
        kwargs_imp = {k: kwargs[k] for k in kwargs if k != 'dropout'}

        dataset = [np.concatenate([train_sc.ad['r'].X.A, test_sc.ad['r'].X.A], axis=0),
                   np.concatenate([train_sc.ad[modal2].X.A, test_sc.ad[modal2].X.A], axis=0)]
        # Integration
        size_str, hash_str = hash_kwargs(kwargs, train_sc.data_name, dataset)

        # Make dir
        self.model_filename.mkdir(parents=True, exist_ok=True)
        
        prefix = str(self.model_filename /'jm---')
        model_str = prefix + hash_str + '.h5'
        match_str = prefix + size_str + '.npy'

        # Instantiate
        mr = list(np.load(match_str, allow_pickle=True)) if os.path.exists(match_str) else None
        self.model = JAMIE(**kwargs, match_result=mr, debug=True)
        
        if not train:
            if os.path.exists(model_str):
                self.model.load_model(model_str)
                print(f'Loaded model \'{model_str}\'')
            else:
                print(f'Model \'{model_str}\' not found')

        return dataset, model_str