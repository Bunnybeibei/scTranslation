"""
Training and testing functionality for scPair models.
"""

import logging
import sys
from typing import Optional
import time
from sctranslation.transforms import (
    ADT_CLR_Transform,
    use_hvg,
    filter_features,
    identity,
)
import os
import scanpy as sc
from pathlib import Path
from sctranslation.utils.logger import ConfigLogger
from sctranslation.transforms.misc import Compose
from sctranslation.metrics import scTranslation_eval
from .scpair import *


class scPairRunner:
    """scPairRunner class for training and testing scPair models."""

    def __init__(
        self,
        config,
        model_filename: Optional[str] = None,
        data_name: Optional[str] = None,
        saved_path: Optional[str] = "",
        random_seed: Optional[int] = 0,
    ) -> None:
        """Initialize a scPairRunner."""
        self.model_filename = Path(model_filename) / data_name / str(random_seed)
        self.logger = ConfigLogger(self.model_filename / f"log.txt")
        self.logger.info("Preparing to run scPair on dataset: %s", data_name)
        self.logger.log_config(config)

        """Initialize a ModelRunner"""
        self.config = config
        self.saved_path = Path(saved_path) / data_name / str(random_seed)
        self.random_seed = random_seed

    @staticmethod
    def preprocessing_pipeline_r(config):
        """Create a preprocessing pipeline for RNA data."""
        transforms = [
            # (
            #     use_hvg(n_top_genes=config.n_top_genes)
            #     if config.use_hvg_flag
            #     else identity()
            # ),
        ]
        return Compose(*transforms)

    @staticmethod
    def preprocessing_pipeline_a(config):
        """Create a preprocessing pipeline for ATAC data."""
        transforms = [
            (
                filter_features(fpeaks=config.fpeaks)
                if config.fpeaks != 0.0
                else identity()
            ),
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
        """Train the scPair model.

        Parameters
        ----------
        train_sc : AnnData
            The training dataset.
        val_sc : AnnData
            The validation dataset.
        test_sc : AnnData
            The test dataset.
        """
        # Add configuration parameters from the training data
        if "a" in train_sc.ad.keys():
            modal2 = "a"
            modal2_name = "Peaks"
            modal2_distribution = "ber"
        elif "p" in train_sc.ad.keys():
            modal2 = "p"
            modal2_name = "Protein"
            modal2_distribution = "gau"
        train_sc.ad["r"].obs["scPair_split"] = "train"
        val_sc.ad["r"].obs["scPair_split"] = "val"
        test_sc.ad["r"].obs["scPair_split"] = "test"
        train_sc.ad[modal2].obs["scPair_split"] = "train"
        val_sc.ad[modal2].obs["scPair_split"] = "val"
        test_sc.ad[modal2].obs["scPair_split"] = "test"
        rna_adata = train_sc.ad["r"].concatenate([val_sc.ad["r"], test_sc.ad["r"]])
        atac_adata = train_sc.ad[modal2].concatenate(
            [val_sc.ad[modal2], test_sc.ad[modal2]]
        )
        adata_paired = merge_paired_data(
            [rna_adata, atac_adata],
            modality_names=["Gene_Expression", modal2_name],
            modality_distributions=["zinb", modal2_distribution],
        )
        """
        set up scPair object
        """
        if not os.path.exists(self.model_filename):
            os.makedirs(self.model_filename)
        scpair_setup = scPair_object(
            scobj=adata_paired,
            cov=None,
            modalities=(
                {"Gene_Expression": "zinb", modal2_name: "ber"}
                if modal2 == "a"
                else {"Gene_Expression": "zinb", modal2_name: "gau"}
            ),
            sample_factor_rna=self.config.sample_factor_rna,
            sample_factor_atac=self.config.sample_factor_atac,
            infer_library_size_rna=self.config.infer_library_size_rna,
            infer_library_size_atac=self.config.infer_library_size_atac,
            batchnorm=self.config.batchnorm,
            layernorm=self.config.layernorm,
            SEED=self.random_seed,
            hidden_layer=self.config.hidden_layer,
            dropout_rate=self.config.dropout_rate,
            learning_rate_prediction=self.config.learning_rate_prediction,
            max_epochs=self.config.max_epochs,
            save_path=self.model_filename,
        )

        """
        start running optimization for scPair framework
        """
        self.logger.info("Starting training...")
        start_time = time.time()
        res = scpair_setup.run()
        training_time = time.time() - start_time
        self.logger.info(f"Training took {training_time:.2f} seconds")
        self.logger.info(f"Training took {training_time / 60:.2f} minutes")
        self.logger.info(f"Training took {training_time / 3600:.2f} hours")

        # Start testing
        self.logger.info("Starting testing...")
        start_time = time.time()
        predictions = scpair_setup.predict()
        test_time = time.time() - start_time
        self.logger.info(f"Test took {test_time:.2f} seconds")
        self.logger.info(f"Test took {test_time / 60:.2f} minutes")
        self.logger.info(f"Test took {test_time / 3600:.2f} hours")

        print("predictions: ", predictions[f"{modal2_name}_test"])
        print("predictions: ", predictions["Gene_Expression_test"])
        # Save predictions
        np.save(
            self.model_filename / f"{modal2_name}_test.npy",
            predictions[f"{modal2_name}_test"],
        )
        np.save(
            self.model_filename / "gene_expression_test.npy",
            predictions["Gene_Expression_test"],
        )

    def test(
        self,
        train_sc: sc.AnnData,
        val_sc: sc.AnnData,
        test_sc: sc.AnnData,
    ) -> None:
        """Evaluate the scPair model.

        Parameters
        ----------
        train_sc : AnnData
            The training dataset.
        val_sc : AnnData
            The validation dataset.
        test_sc : AnnData
            The test dataset.
        """
        if "a" in train_sc.ad.keys():
            modal2 = "a"
            modal2_name = "Peaks"
        elif "p" in train_sc.ad.keys():
            modal2 = "p"
            modal2_name = "Protein"
        # Add configuration parameters from the test data
        pred_atac = np.load(self.model_filename / f"{modal2_name}_test.npy")
        pred_r = np.load(self.model_filename / "gene_expression_test.npy")

        pred_r = sc.AnnData(
            X=pred_r,
            var=test_sc.ad["r"].var,
            obs=test_sc.ad["r"].obs,
        )
        pred_atac = sc.AnnData(
            X=pred_atac,
            var=test_sc.ad[modal2].var,
            obs=test_sc.ad[modal2].obs,
        )
        evaluator = scTranslation_eval(model="scpair", saved_path=self.saved_path)
        evaluator.add_data(pred=pred_r, truth=test_sc.ad["r"], name=f"{modal2}2r")
        evaluator.add_data(pred=pred_atac, truth=test_sc.ad[modal2], name=f"r2{modal2}")
        # print("std pcc: ", np.std(pcc_list))
        # print("max pcc: ", np.max(pcc_list))
