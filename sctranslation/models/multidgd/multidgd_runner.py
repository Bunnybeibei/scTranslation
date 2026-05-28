"""
Training and testing functionality for multiDGD models.
"""

from typing import Optional
import numpy as np
from scipy.sparse import hstack
import pandas as pd
import os
import time
import scanpy as sc
import torch

from sctranslation.transforms import add_attribution, use_hvg, filter_features, identity, ADT_CLR_Transform
from sctranslation.data import scData
from sctranslation.metrics import scTranslation_eval

from sctranslation.transforms.misc import Compose
from sctranslation.utils.logger import ConfigLogger

from .multidgd_dataloader import multiDGDDataModule, multiDGDDataset
from .multidgd_modellling import DGD as Model

from scipy.sparse import csr_matrix

from joblib import Parallel, delayed
from pathlib import Path


class multiDGDRunner:
    """A class to run multiDGD models.
    Parameters
    ----------
    config : Config object
        The multiDGD configuration.
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
        """Initialize a multiDGDRunner."""
        self.model_filename = Path(model_filename) / data_name / str(random_seed)
        self.logger = ConfigLogger(self.model_filename /f'log.txt')
        self.logger.info("Preparing to run multiDGD on dataset: %s", data_name)
        self.logger.log_config(config)
        
        """Initialize a ModelRunner"""
        self.config = config
        self.saved_path = Path(saved_path) / data_name / str(random_seed)
        self.random_seed = random_seed

    @staticmethod
    def preprocessing_pipeline_r(config):
        """Create a preprocessing pipeline for RNA data."""
        transforms = [
            add_attribution(),
            # use_hvg(n_top_genes=config.n_top_genes, retain_hvg=True)
        ]

        return Compose(*transforms)

    @staticmethod
    def preprocessing_pipeline_a(config):
        """Create a preprocessing pipeline for ATAC data."""
        transforms = [
            add_attribution(),
            filter_features(fpeaks=config.fpeaks) if config.fpeaks != 0. else identity(),
        ]
        return Compose(*transforms)
    
    @staticmethod
    def preprocessing_pipeline_p(config):
        """Create a preprocessing pipeline for ATAC data."""
        transforms = [
            add_attribution(),
            # ADT_CLR_Transform(),
        ]
        return Compose(*transforms)
    
    def train(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData,
    ) -> None:
        """Train the multiDGD model.

        Parameters
        ----------
        train_sc : AnnData
            The training dataset.
        val_sc : AnnData
            The validation dataset.
        """

        # Initialize the model
        train_set = multiDGDDataset(self.combine_multiomics_data(data=train_sc))
        val_set = multiDGDDataset(self.combine_multiomics_data(data=val_sc))
        
        self.initialize_model(train_set=train_set, val_set=val_set, model_name=train_sc.data_name)

        # Create dataloaders
        train_dataloader = multiDGDDataModule(
            ad=train_set,
            n_workers=self.config.n_workers,
            batch_size=self.config.batch_size,
        ).get_dataloader()
        validation_dataloader = multiDGDDataModule(
            ad=val_set,
            n_workers=self.config.n_workers,
            batch_size=self.config.batch_size,
        ).get_dataloader()

        # Start training
        start_time = time.time()
        self.model.train(n_epochs=self.config.max_epochs, train_loader=train_dataloader, validation_loader=validation_dataloader)
        training_time = time.time() - start_time
        self.logger.info(f"Training took {training_time:.2f} seconds")
        self.logger.info(f"Training ae took {training_time / 60:.2f} minutes")
        self.logger.info(f"Training ae took {training_time / 3600:.2f} hours")
        self.model.save()
        self.logger.info(f"Model saved to {self.model_filename}")

    def test(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData,
    ) -> None:
        """Evaluate the multiDGD model.

        Parameters
        ----------
        test_sc : AnnData
            The test dataset.
        """

        # Initialize the model to get self.model
        combine_set = self.combine_multiomics_data(data=test_sc)
        test_set = multiDGDDataset(combine_set)
        
        self.initialize_model(model_name=test_sc.data_name, test_set=test_set, train=False)

        # Create dataloaders
        test_dataloader = multiDGDDataModule(
            ad=test_set,
            n_workers=self.config.n_workers,
            batch_size=self.config.batch_size,
        ).get_dataloader()

        # Start testing
        start_time = time.time()
        self.model.test(test_loader=test_dataloader, n_epochs=20)
        # self.model.save()
        self.logger.info(f"Model saved to {self.model_filename}")
        # Empty cache
        torch.cuda.empty_cache()
        self.logger.info("Testing model...")
        # self.evaluate(data=combine_set)
        # test_rep = self.model.get_representation(split='test')
        # predict_r = self.model.decoder(torch.from_numpy(test_rep).cuda())[0]
        # predict_a = self.model.decoder(torch.from_numpy(test_rep).cuda())[1]
        predict_r, predict_a = self.model.predict_from_representation(self.model.test_rep)
        A2R_predict, R2A_predict = predict_r, predict_a
        A2R_predict = sc.AnnData(X=csr_matrix(A2R_predict.detach().cpu().numpy()),
                                 obs=test_sc.ad['r'].obs.copy(),
                                    var=test_sc.ad['r'].var.copy())
        R2A_predict = sc.AnnData(X=csr_matrix(R2A_predict.detach().cpu().numpy()),
                                    obs=test_sc.ad['a'].obs.copy() if 'a' in test_sc.ad else test_sc.ad['p'].obs.copy(),
                                        var=test_sc.ad['a'].var.copy() if 'a' in test_sc.ad else test_sc.ad['p'].var.copy())
        truth_r = test_sc.ad['r'].copy() if 'r' in test_sc.ad else test_sc.ad['p'].copy()
        truth_a = test_sc.ad['a'].copy() if 'a' in test_sc.ad else test_sc.ad['p'].copy()
        evaluator = scTranslation_eval(model='multiDGD', saved_path=self.saved_path)
        evaluator.add_data(pred=R2A_predict, truth=truth_a, name='r2a' if 'a' in test_sc.ad else 'p2a')
        evaluator.add_data(pred=A2R_predict, truth=truth_r, name='a2r' if 'a' in test_sc.ad else 'p2r')
        # evaluator.forward()
        test_time = time.time() - start_time
        self.logger.info(f"Test took {test_time:.2f} seconds")
        self.logger.info(f"Training ae took {test_time / 60:.2f} minutes")
        self.logger.info(f"Training ae took {test_time / 3600:.2f} hours")


    def initialize_model(self, train_set: multiDGDDataset=None, val_set: multiDGDDataset=None, model_name:str=None, train:bool=True, test_set:multiDGDDataset=None) -> None:
        """Initialize the multiDGD model."""
        if train:
            if not os.path.exists(self.model_filename):
                Path(self.model_filename).mkdir(parents=True, exist_ok=True)
            
            assert train_set is not None, "Training set is required for training mode."
            assert val_set is not None, "Validation set is required for training mode."

            model_params = dict(
                train_set=train_set,
                val_set=val_set, 
                test_set=None,
                covariates=[x for x in train_set.data.obs.columns if 'covariate_' in x],
                parameter_dictionary=None,
                scaling='sum',
                save_dir=self.model_filename,
                random_seed = self.random_seed,
                model_name=model_name,
                print_outputs=False
            )

            self.model = Model(**model_params)
        else:
            assert self.model_filename is not None, "Model filename is required for test mode."
            assert test_set is not None, "Test set is required for test mode."
            model_params = dict(
                data=test_set,
                save_dir=self.model_filename,
                random_seed = self.random_seed,
                model_name=model_name,
            )
                        
            self.model = Model.load(**model_params)

    def combine_multiomics_data(self, data: scData) -> multiDGDDataset:
        """Combine multiomics data for training."""
        modal2_ad = data.ad_a if data.ad_a is not None else data.ad_p
        # Connect two datasets data.data['r'](X,Y) and data.data['a'](X,Z) to new dataset (X,Y+Z)
        # Obs find columns in data.data['r'].obs but not in data.data['a'].obs
        columns_need_to_add = [x for x in modal2_ad.obs.columns if x not in data.ad_r.obs.columns]
        # Var find columns in data.data['r'].var but not in data.data['a'].var
        combine_data = sc.AnnData(
            X=hstack((data.ad_r.X, modal2_ad.X)),
            obs=pd.concat([data.ad_r.obs, modal2_ad.obs[columns_need_to_add]], axis=1),
            var=pd.concat([data.ad_r.var, modal2_ad.var], axis=0)
        )
        return combine_data
