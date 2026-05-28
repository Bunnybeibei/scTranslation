"""
Training and testing functionality for scButterfly models.
"""

from pathlib import Path
from typing import Optional
import torch.nn as nn
import pandas as pd
import time
import scanpy as sc

from sctranslation.transforms import (
    binary_data,
    filter_features,
    tfidf,
    normalize_a,
    normalize_r,
    log1p,
    use_hvg,
    ADT_CLR_Transform,
    filter_features_babel,
    annotate_babel,
    identity
)
from sctranslation.transforms.misc import Compose
from sctranslation.metrics import scTranslation_eval
from sctranslation.utils.logger import ConfigLogger

from .scButterfly.train_model import Model
from .scButterfly.train_model_cite import Model as Model_CITE
from .scbutterfly_dataloader import scButterflyDataModule

class scButterflyRunner:
    """A class to run scButterfly models.
    Parameters
    ----------
    config : Config object
        The scButterfly configuration.
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
        """Initialize a scButterflyRunner."""
        self.model_filename = Path(model_filename) / data_name / str(random_seed)
        self.logger = ConfigLogger(self.model_filename /f'log.txt')
        self.logger.info("Preparing to run scButterfly on dataset: %s", data_name)
        self.logger.log_config(config)
        
        """Initialize a ModelRunner"""
        self.config = config
        self.saved_path = Path(saved_path) / data_name / str(random_seed)
        self.random_seed = random_seed

    @staticmethod
    def preprocessing_pipeline_r(config):
        """Create a preprocessing pipeline for RNA data."""
        transforms = [
            annotate_babel(),
            # filter_features_babel('r'),
            # normalize_r(),
            # log1p(),
            # use_hvg(n_top_genes=config.n_top_genes) if config.use_hvg_flag else identity(),
        ]

        return Compose(*transforms)

    @staticmethod
    def preprocessing_pipeline_p(config):
        """Create a preprocessing pipeline for ATAC data."""
        transforms = [
            # ADT_CLR_Transform(),
        ]
        return Compose(*transforms)
    
    @staticmethod
    def preprocessing_pipeline_a(config):
        """Create a preprocessing pipeline for ATAC data."""
        transforms = [
            annotate_babel(),
            filter_features_babel('a'),
            binary_data(),
            filter_features(fpeaks=config.fpeaks) if config.fpeaks != 0. else identity(),
            tfidf(),
            normalize_a(),
        ]
        return Compose(*transforms)

    def train(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData = None,
    ) -> None:
        """Train the scButterfly model.

        Parameters
        ----------
        train_sc : AnnData
            The training dataset.
        val_sc : AnnData
            The validation dataset.
        """
        # Add configuration parameters from the training data
        self.config.RNA_input_dim = train_sc.ad["r"].shape[1]
        if 'a' in train_sc.ad:
            modal2 = 'a'
            self.config.ATAC_input_dim = train_sc.ad["a"].X.shape[1]
            self.config.chrom_list = self.get_chrom_list(
                train_sc.ad["a"], data=train_sc.data_name
            )
        elif 'p' in train_sc.ad:
            modal2 = 'p'
            self.config.ADT_input_dim = train_sc.ad["p"].X.shape[1]
        else:
            raise ValueError

        # Initialize the model
        self.initialize_model(modal2)

        # Create dataloaders
        train_dataloader = scButterflyDataModule(
            ad=train_sc,
            n_workers=self.config.n_workers,
            batch_size=self.config.batch_size,
            modal2=modal2
        ).get_dataloader()
        validation_dataloader = scButterflyDataModule(
            ad=val_sc,
            n_workers=self.config.n_workers,
            batch_size=self.config.batch_size,
            modal2=modal2
        ).get_dataloader()

        # Initialize the trainer
        trainer_params = self.initialize_trainer(
            train=True,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            modal2=modal2
        )

        self.logger.info("Starting training...")
        start_time = time.time()
        self.model.train(**trainer_params)
        training_time = time.time() - start_time
        self.logger.info(f"Training took {training_time:.2f} seconds")
        # Change to minutes and hours
        self.logger.info(f"Training took {training_time / 60:.2f} minutes")
        self.logger.info(f"Training took {training_time / 3600:.2f} hours")

    def test(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData,
    ) -> None:
        """Evaluate the scButterfly model.

        Parameters
        ----------
        test_sc : AnnData
            The test dataset.
        """
        # Add configuration parameters from the test data
        self.config.RNA_input_dim = train_sc.ad["r"].shape[1]
        if 'a' in train_sc.ad:
            modal2 = 'a'
            self.config.ATAC_input_dim = train_sc.ad["a"].X.shape[1]
            self.config.chrom_list = self.get_chrom_list(
                train_sc.ad["a"], data=train_sc.data_name
            )
        elif 'p' in train_sc.ad:
            modal2 = 'p'
            self.config.ADT_input_dim = train_sc.ad["p"].X.shape[1]
        else:
            raise ValueError

        # Initialize the model
        self.initialize_model(modal2)

        # Create dataloaders
        R_test_dataloader = scButterflyDataModule(
            ad=test_sc,
            n_workers=self.config.n_workers,
            batch_size=self.config.batch_size,
            data_type='r',
            modal2=modal2
        ).get_dataloader()
        A_test_dataloader = scButterflyDataModule(
            ad=test_sc,
            n_workers=self.config.n_workers,
            batch_size=self.config.batch_size,
            data_type=modal2,
            modal2=modal2
        ).get_dataloader()

        # Initialize the trainer
        test_params = self.initialize_trainer(
            train=False,
            R_test_dataloader=R_test_dataloader,
            A_test_dataloader=A_test_dataloader,
            RNA_data_obs=test_sc.ad["r"].obs,
            ATAC_data_obs=test_sc.ad[modal2].obs,
        )

        # Start testing
        self.logger.info("Starting testing...")
        start_time = time.time()
        pred_rna, true_rna, pred_atac, true_atac = self.model.test(**test_params)
        evaluator = scTranslation_eval(saved_path=self.saved_path)
        evaluator.add_data(pred=pred_rna, truth=true_rna, name=f'{modal2}2r')
        evaluator.add_data(pred=pred_atac, truth=true_atac, name=f'r2{modal2}')
        # evaluator.forward()
        test_time = time.time() - start_time
        self.logger.info(f"Test took {test_time:.2f} seconds")
        # Change to minutes and hours
        self.logger.info(f"Test took {test_time / 60:.2f} minutes")
        self.logger.info(f"Test took {test_time / 3600:.2f} hours")

    def initialize_model(self, modal2='a') -> None:
        """Initialize the scButterfly model."""
        if modal2 == 'a':
            model_params = dict(
                chrom_list=self.config.chrom_list,
                R_encoder_dim_list=[self.config.RNA_input_dim]
                + self.config.R_encoder_dim_list,
                A_encoder_dim_list=[
                    self.config.ATAC_input_dim,
                    32 * len(self.config.chrom_list),
                ]
                + self.config.A_encoder_dim_list,
                R_decoder_dim_list=self.config.R_decoder_dim_list
                + [self.config.RNA_input_dim],
                A_decoder_dim_list=self.config.A_decoder_dim_list
                + [32 * len(self.config.chrom_list), self.config.ATAC_input_dim],
                R_encoder_nlayer=self.config.R_encoder_nlayer,
                A_encoder_nlayer=self.config.A_encoder_nlayer,
                R_decoder_nlayer=self.config.R_decoder_nlayer,
                A_decoder_nlayer=self.config.A_decoder_nlayer,
                translator_embed_dim=self.config.translator_embed_dim,
                translator_input_dim_r=self.config.translator_input_dim_r,
                translator_input_dim_a=self.config.translator_input_dim_a,
                translator_embed_act_list=self.config.translator_embed_act_list,
                discriminator_nlayer=self.config.discriminator_nlayer,
                discriminator_dim_list_R=self.config.discriminator_dim_list_R,
                discriminator_dim_list_A=self.config.discriminator_dim_list_A,
                discriminator_act_list=self.config.discriminator_act_list,
                dropout_rate=self.config.dropout_rate,
                R_noise_rate=self.config.R_noise_rate,
                A_noise_rate=self.config.A_noise_rate,
            )

            self.model = Model(**model_params)
        elif modal2 == 'p':
            model_params = dict(
                R_encoder_nlayer = 2,
                A_encoder_nlayer = 2,
                R_decoder_nlayer = 2,
                A_decoder_nlayer = 2,
                R_encoder_dim_list = [self.config.RNA_input_dim, 256, 128],
                A_encoder_dim_list = [self.config.ADT_input_dim, 128, 128],
                R_decoder_dim_list = [128, 256, self.config.RNA_input_dim],
                A_decoder_dim_list = [128, 128, self.config.ADT_input_dim],
                R_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
                A_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
                R_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
                A_decoder_act_list = [nn.LeakyReLU(), nn.Identity()],
                translator_embed_dim = 128,
                translator_input_dim_r = 128,
                translator_input_dim_a = 128,
                translator_embed_act_list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
                discriminator_nlayer = 1,
                discriminator_dim_list_R = [128],
                discriminator_dim_list_A = [128],
                discriminator_act_list = [nn.Sigmoid()],
                dropout_rate = 0.1,
                R_noise_rate = 0.5,
                A_noise_rate = 0,
                chrom_list = [],
            )
            self.model = Model_CITE(**model_params)

    def initialize_trainer(
        self,
        train: bool,
        train_dataloader: scButterflyDataModule = None,
        validation_dataloader: scButterflyDataModule = None,
        R_test_dataloader: scButterflyDataModule = None,
        A_test_dataloader: scButterflyDataModule = None,
        ATAC_data_obs: pd.DataFrame = None,
        RNA_data_obs: pd.DataFrame = None,
        modal2: str = 'a'
    ) -> dict:
        """Initialize the scButterfly trainer."""
        if train:
            R_kl_div = 1 / self.config.RNA_input_dim * 20
            A_kl_div = 1 / self.config.ATAC_input_dim * 20 if modal2=='a' else 1 / 150
            kl_div = R_kl_div + A_kl_div
            trainer_params = dict(
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                RNA_input_dim=self.config.RNA_input_dim,
                ATAC_input_dim=self.config.ATAC_input_dim if modal2=='a' else self.config.ADT_input_dim,
                R_encoder_lr=self.config.R_encoder_lr,
                A_encoder_lr=self.config.A_encoder_lr,
                R_decoder_lr=self.config.R_decoder_lr,
                A_decoder_lr=self.config.A_decoder_lr,
                R_translator_lr=self.config.R_translator_lr,
                A_translator_lr=self.config.A_translator_lr,
                translator_lr=self.config.translator_lr,
                discriminator_lr=self.config.discriminator_lr,
                R2R_pretrain_epoch=self.config.R2R_pretrain_epoch,
                A2A_pretrain_epoch=self.config.A2A_pretrain_epoch,
                lock_encoder_and_decoder=self.config.lock_encoder_and_decoder,
                translator_epoch=self.config.translator_epoch,
                patience=self.config.patience,
                r_loss=nn.MSELoss(size_average=True),
                a_loss=nn.BCELoss(size_average=True) if modal2=='a' else nn.MSELoss(size_average=True),
                d_loss=nn.BCELoss(size_average=True),
                loss_weight=[1, 2, 1, R_kl_div, A_kl_div, kl_div],
                output_path=self.model_filename,
                seed=self.random_seed,
                kl_mean=self.config.kl_mean,
                R_pretrain_kl_warmup=self.config.R_pretrain_kl_warmup,
                A_pretrain_kl_warmup=self.config.A_pretrain_kl_warmup,
                translation_kl_warmup=self.config.translation_kl_warmup,
                load_model=self.config.load_model,
                my_logger=self.logger,
            )
            return trainer_params
        else:
            test_params = dict(
                R_test_dataloader=R_test_dataloader,
                A_test_dataloader=A_test_dataloader,
                RNA_data_obs=RNA_data_obs,
                ATAC_data_obs=ATAC_data_obs,
                model_path=self.model_filename,
                load_model=self.model_filename,
                output_path=self.saved_path,
                my_logger=self.logger,
            )
            return test_params

    def get_chrom_list(self, ATAC_data: sc.AnnData, data: str = None):
        """Get the number of peaks in each chromosome."""
        if data == "BMMC":

            chrom_list = []
            last_one = ""
            for i in range(len(ATAC_data.var.gene_id.index)):
                temp = ATAC_data.var.gene_id.index[i].split("-")[0]
                if temp[0:3] == "chr":
                    if not temp == last_one:
                        chrom_list.append(1)
                        last_one = temp
                    else:
                        chrom_list[-1] += 1
                else:
                    chrom_list[-1] += 1

        elif data == "MDS":

            chrom_list = [0 for i in range(20)]
            for i in range(len(ATAC_data.var.chrom)):
                temp = ATAC_data.var.chrom[i]
                if temp[3:] == "X":
                    chrom_list[19] += 1
                else:
                    chrom_list[int(temp[3:]) - 1] += 1
            peaks_idx = [[] for i in range(len(chrom_list))]
            for i in range(len(ATAC_data.var.chrom)):
                temp = ATAC_data.var.chrom[i]
                if temp[3:] == "X":
                    peaks_idx[19].append(i)
                else:
                    peaks_idx[int(temp[3:]) - 1].append(i)
            peaks_idx_temp = []
            for i in range(len(peaks_idx)):
                peaks_idx_temp.extend(peaks_idx[i])

            ATAC_data.X = ATAC_data.X[:, peaks_idx_temp]
            ATAC_data.var = ATAC_data.var.iloc[peaks_idx_temp, :]

        else:
            chrom_list = []
            last_one = ""
            # Add chr to chromosome names
            ATAC_data.var["chrom"] = ATAC_data.var["chrom"].apply(
                lambda x: "chr" + str(x)
            )
            for i in range(len(ATAC_data.var.chrom)):
                temp = ATAC_data.var.chrom[i]
                if temp[0:3] == "chr":
                    if not temp == last_one:
                        chrom_list.append(1)
                        last_one = temp
                    else:
                        chrom_list[-1] += 1
                else:
                    chrom_list[-1] += 1
        return chrom_list
