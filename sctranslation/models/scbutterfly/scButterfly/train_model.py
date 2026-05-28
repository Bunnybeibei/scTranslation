import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sctranslation.metrics import scTranslation_eval
from tqdm import tqdm
from .model_component import *
from .model_utlis import *
from .calculate_cluster import *
from .draw_cluster import *
from .data_processing import *
from .logger import *
from pathlib import Path


class Model:
    def __init__(
        self,
        chrom_list: list,
        R_encoder_dim_list: list,
        A_encoder_dim_list: list,
        R_decoder_dim_list: list,
        A_decoder_dim_list: list,
        R_encoder_nlayer: int = 2,
        A_encoder_nlayer: int = 2,
        R_decoder_nlayer: int = 2,
        A_decoder_nlayer: int = 2,
        R_encoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_encoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_decoder_act_list: list = [nn.LeakyReLU(), nn.Sigmoid()],
        translator_embed_dim: int = 128,
        translator_input_dim_r: int = 128,
        translator_input_dim_a: int = 128,
        translator_embed_act_list: list = [
            nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
        ],
        discriminator_nlayer: int = 1,
        discriminator_dim_list_R: list = [128],
        discriminator_dim_list_A: list = [128],
        discriminator_act_list: list = [nn.Sigmoid()],
        dropout_rate: float = 0.1,
        R_noise_rate: float = 0.5,
        A_noise_rate: float = 0.3,
    ):
        """
        Main model. Some parameters need information about data, please see in Tutorial.

        Parameters
        ----------

        chrom_list: list
            list of peaks count for each chromosomes.

        R_encoder_dim_list: list
            dimension list of RNA encoder, length equal to R_encoder_nlayer + 1, the first equal to RNA data dimension, the last equal to embedding dimension.

        A_encoder_dim_list: list
            dimension list of ATAC encoder, length equal to A_encoder_nlayer + 1, the first equal to RNA data dimension, the last equal to embedding dimension.

        R_decoder_dim_list: list
            dimension list of RNA decoder, length equal to R_decoder_nlayer + 1, the last equal to embedding dimension, the first equal to RNA data dimension.

        A_decoder_dim_list: list
            dimension list of ATAC decoder, length equal to A_decoder_nlayer + 1, the last equal to embedding dimension, the first equal to RNA data dimension.

        R_encoder_nlayer: int
            layer counts of RNA encoder, default 2.

        A_encoder_nlayer: int
            layer counts of ATAC encoder, default 2.

        R_decoder_nlayer: int
            layer counts of RNA decoder, default 2.

        A_decoder_nlayer: int
            layer counts of ATAC decoder, default 2.

        R_encoder_act_list: list
            activation list of RNA encoder, length equal to R_encoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].

        A_encoder_act_list: list
            activation list of ATAC encoder, length equal to A_encoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].

        R_decoder_act_list: list
            activation list of RNA decoder, length equal to R_decoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].

        A_decoder_act_list: list
            activation list of ATAC decoder, length equal to A_decoder_nlayer, default [nn.LeakyReLU(), nn.Sigmoid()].

        translator_embed_dim: int
            dimension of embedding space for translator, default 128.

        translator_input_dim_r: int
            dimension of input from RNA encoder for translator, default 128.

        translator_input_dim_a: int
            dimension of input from ATAC encoder for translator, default 128.

        translator_embed_act_list: list
            activation list for translator, involving [mean_activation, log_var_activation, decoder_activation], default [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()].

        discriminator_nlayer: int
            layer counts of discriminator, default 1.

        discriminator_dim_list_R: list
            dimension list of discriminator, length equal to discriminator_nlayer, the first equal to translator_input_dim_R, default [128].

        discriminator_dim_list_A: list
            dimension list of discriminator, length equal to discriminator_nlayer, the first equal to translator_input_dim_A, default [128].

        discriminator_act_list: list
            activation list of discriminator, length equal to  discriminator_nlayer, default [nn.Sigmoid()].

        dropout_rate: float
            rate of dropout for network, default 0.1.

        R_noise_rate: float
            rate of set part of RNA input data to 0, default 0.5.

        A_noise_rate: float
            rate of set part of ATAC input data to 0, default 0.3.

        """

        self.RNA_encoder = NetBlock(
            nlayer=R_encoder_nlayer,
            dim_list=R_encoder_dim_list,
            act_list=R_encoder_act_list,
            dropout_rate=dropout_rate,
            noise_rate=R_noise_rate,
        )

        self.ATAC_encoder = Split_Chrom_Encoder_block(
            nlayer=A_encoder_nlayer,
            dim_list=A_encoder_dim_list,
            act_list=A_encoder_act_list,
            chrom_list=chrom_list,
            dropout_rate=dropout_rate,
            noise_rate=A_noise_rate,
        )

        self.RNA_decoder = NetBlock(
            nlayer=R_decoder_nlayer,
            dim_list=R_decoder_dim_list,
            act_list=R_decoder_act_list,
            dropout_rate=dropout_rate,
            noise_rate=0,
        )

        self.ATAC_decoder = Split_Chrom_Decoder_block(
            nlayer=A_decoder_nlayer,
            dim_list=A_decoder_dim_list,
            act_list=A_decoder_act_list,
            chrom_list=chrom_list,
            dropout_rate=dropout_rate,
            noise_rate=0,
        )

        self.R_translator = Single_Translator(
            translator_input_dim=translator_input_dim_r,
            translator_embed_dim=translator_embed_dim,
            translator_embed_act_list=translator_embed_act_list,
        )

        self.A_translator = Single_Translator(
            translator_input_dim=translator_input_dim_a,
            translator_embed_dim=translator_embed_dim,
            translator_embed_act_list=translator_embed_act_list,
        )

        self.translator = Translator(
            translator_input_dim_r=translator_input_dim_r,
            translator_input_dim_a=translator_input_dim_a,
            translator_embed_dim=translator_embed_dim,
            translator_embed_act_list=translator_embed_act_list,
        )

        discriminator_dim_list_R.append(1)
        discriminator_dim_list_A.append(1)
        self.discriminator_R = NetBlock(
            nlayer=discriminator_nlayer,
            dim_list=discriminator_dim_list_R,
            act_list=discriminator_act_list,
            dropout_rate=0,
            noise_rate=0,
        )

        self.discriminator_A = NetBlock(
            nlayer=discriminator_nlayer,
            dim_list=discriminator_dim_list_A,
            act_list=discriminator_act_list,
            dropout_rate=0,
            noise_rate=0,
        )

        # use GPU for training if cuda is available
        if torch.cuda.is_available():
            self.RNA_encoder = self.RNA_encoder.cuda()
            self.RNA_decoder = self.RNA_decoder.cuda()
            self.ATAC_encoder = self.ATAC_encoder.cuda()
            self.ATAC_decoder = self.ATAC_decoder.cuda()
            self.R_translator = self.R_translator.cuda()
            self.A_translator = self.A_translator.cuda()
            self.translator = self.translator.cuda()
            self.discriminator_R = self.discriminator_R.cuda()
            self.discriminator_A = self.discriminator_A.cuda()

        self.is_train_finished = False

    def set_train(self):
        self.RNA_encoder.train()
        self.RNA_decoder.train()
        self.ATAC_encoder.train()
        self.ATAC_decoder.train()
        self.R_translator.train()
        self.A_translator.train()
        self.translator.train()
        self.discriminator_R.train()
        self.discriminator_A.train()

    def set_eval(self):
        self.RNA_encoder.eval()
        self.RNA_decoder.eval()
        self.ATAC_encoder.eval()
        self.ATAC_decoder.eval()
        self.R_translator.eval()
        self.A_translator.eval()
        self.translator.eval()
        self.discriminator_R.eval()
        self.discriminator_A.eval()

    def forward_R2R(self, RNA_input, r_loss, kl_div_w, forward_type):
        latent_layer, mu, d = self.R_translator(
            self.RNA_encoder(RNA_input), forward_type
        )
        predict_RNA = self.RNA_decoder(latent_layer)
        reconstruct_loss = r_loss(predict_RNA, RNA_input)
        kl_div_r = -0.5 * torch.mean(1 + d - mu.pow(2) - d.exp())
        loss = reconstruct_loss + kl_div_w * kl_div_r
        return loss, reconstruct_loss, kl_div_r

    def forward_A2A(self, ATAC_input, a_loss, kl_div_w, forward_type):
        latent_layer, mu, d = self.A_translator(
            self.ATAC_encoder(ATAC_input), forward_type
        )
        predict_ATAC = self.ATAC_decoder(latent_layer)
        reconstruct_loss = a_loss(predict_ATAC, ATAC_input)
        kl_div_a = -0.5 * torch.mean(1 + d - mu.pow(2) - d.exp())
        loss = reconstruct_loss + kl_div_w * kl_div_a
        return loss, reconstruct_loss, kl_div_a

    def forward_translator(
        self,
        batch_samples,
        RNA_input_dim,
        ATAC_input_dim,
        a_loss,
        r_loss,
        loss_weight,
        forward_type,
        kl_div_mean=False,
    ):

        RNA_input, ATAC_input = torch.split(
            batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1
        )

        # forward generator

        R2 = self.RNA_encoder(RNA_input)
        A2 = self.ATAC_encoder(ATAC_input)
        if forward_type == "train":
            R2R, R2A, mu_r, sigma_r = self.translator.train_model(R2, "RNA")
            A2R, A2A, mu_a, sigma_a = self.translator.train_model(A2, "ATAC")
        elif forward_type == "test":
            R2R, R2A, mu_r, sigma_r = self.translator.test_model(R2, "RNA")
            A2R, A2A, mu_a, sigma_a = self.translator.test_model(A2, "ATAC")

        R2R = self.RNA_decoder(R2R)
        R2A = self.ATAC_decoder(R2A)
        A2R = self.RNA_decoder(A2R)
        A2A = self.ATAC_decoder(A2A)

        # reconstruct loss
        lossR2R = r_loss(R2R, RNA_input)
        lossA2R = r_loss(A2R, RNA_input)
        lossR2A = a_loss(R2A, ATAC_input)
        lossA2A = a_loss(A2A, ATAC_input)

        # kl divergence
        if kl_div_mean:
            kl_div_r = -0.5 * torch.mean(1 + sigma_r - mu_r.pow(2) - sigma_r.exp())
            kl_div_a = -0.5 * torch.mean(1 + sigma_a - mu_a.pow(2) - sigma_a.exp())
        else:
            kl_div_r = torch.clamp(
                -0.5 * torch.sum(1 + sigma_r - mu_r.pow(2) - sigma_r.exp()), 0, 10000
            )
            kl_div_a = torch.clamp(
                -0.5 * torch.sum(1 + sigma_a - mu_a.pow(2) - sigma_a.exp()), 0, 10000
            )

        # calculate the loss
        r_loss_w, a_loss_w, d_loss_w, kl_div_R, kl_div_A, kl_div_w = loss_weight
        reconstruct_loss = r_loss_w * (lossR2R + lossA2R) + a_loss_w * (
            lossR2A + lossA2A
        )

        kl_div = kl_div_r + kl_div_a

        loss_g = kl_div_w * kl_div + reconstruct_loss

        return reconstruct_loss, kl_div, loss_g

    def forward_discriminator(
        self, batch_samples, RNA_input_dim, ATAC_input_dim, d_loss, forward_type
    ):

        RNA_input, ATAC_input = torch.split(
            batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1
        )

        # forward of generator
        R2 = self.RNA_encoder(RNA_input)
        A2 = self.ATAC_encoder(ATAC_input)
        if forward_type == "train":
            R2R, R2A, mu_r, sigma_r = self.translator.train_model(R2, "RNA")
            A2R, A2A, mu_a, sigma_a = self.translator.train_model(A2, "ATAC")
        elif forward_type == "test":
            R2R, R2A, mu_r, sigma_r = self.translator.test_model(R2, "RNA")
            A2R, A2A, mu_a, sigma_a = self.translator.test_model(A2, "ATAC")

        batch_size = batch_samples.shape[0]

        # 1 menas a real data 0 menas a generated data, here use a soft label
        temp1 = np.random.rand(batch_size)
        temp = [0 for item in temp1]
        for i in range(len(temp1)):
            if temp1[i] > 0.8:
                temp[i] = temp1[i]
            elif temp1[i] <= 0.8 and temp1[i] > 0.5:
                temp[i] = 0.8
            elif temp1[i] <= 0.5 and temp1[i] > 0.2:
                temp[i] = 0.2
            else:
                temp[i] = temp1[i]

        input_data_a = torch.stack(
            [A2[i] if temp[i] > 0.5 else R2A[i] for i in range(batch_size)], dim=0
        )
        input_data_r = torch.stack(
            [R2[i] if temp[i] > 0.5 else A2R[i] for i in range(batch_size)], dim=0
        )

        predict_atac = self.discriminator_A(input_data_a)
        predict_rna = self.discriminator_R(input_data_r)

        loss1 = d_loss(
            predict_atac.reshape(batch_size), torch.tensor(temp).cuda().float()
        )
        loss2 = d_loss(
            predict_rna.reshape(batch_size), torch.tensor(temp).cuda().float()
        )
        return loss1 + loss2

    def save_model_dict(self, output_path):
        torch.save(self.RNA_encoder.state_dict(), output_path /"model/RNA_encoder.pt")
        torch.save(
            self.ATAC_encoder.state_dict(), output_path /"model/ATAC_encoder.pt"
        )
        torch.save(self.RNA_decoder.state_dict(), output_path / "model/RNA_decoder.pt")
        torch.save(
            self.ATAC_decoder.state_dict(), output_path / "model/ATAC_decoder.pt"
        )
        torch.save(
            self.R_translator.state_dict(), output_path / "model/R_translator.pt"
        )
        torch.save(
            self.A_translator.state_dict(), output_path / "model/A_translator.pt"
        )
        torch.save(self.translator.state_dict(), output_path / "model/translator.pt")
        torch.save(
            self.discriminator_A.state_dict(), output_path / "model/discriminator_A.pt"
        )
        torch.save(
            self.discriminator_R.state_dict(), output_path / "model/discriminator_R.pt"
        )

    def train(
        self,
        loss_weight: list,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        RNA_input_dim: int,
        ATAC_input_dim: int,
        R_encoder_lr: float = 0.001,
        A_encoder_lr: float = 0.001,
        R_decoder_lr: float = 0.001,
        A_decoder_lr: float = 0.001,
        R_translator_lr: float = 0.001,
        A_translator_lr: float = 0.001,
        translator_lr: float = 0.001,
        discriminator_lr: float = 0.005,
        R2R_pretrain_epoch: int = 100,
        A2A_pretrain_epoch: int = 100,
        lock_encoder_and_decoder: bool = False,
        translator_epoch: int = 200,
        patience: int = 50,
        r_loss=nn.MSELoss(size_average=True),
        a_loss=nn.BCELoss(size_average=True),
        d_loss=nn.BCELoss(size_average=True),
        output_path: str = None,
        seed: int = 19193,
        kl_mean: bool = True,
        R_pretrain_kl_warmup: int = 50,
        A_pretrain_kl_warmup: int = 50,
        translation_kl_warmup: int = 50,
        load_model: str = None,
        my_logger: logging.Logger = None,
    ):
        """
        Training for model. Some parameters need information about data, please see in Tutorial.

        Parameters
        ----------
        loss_weight: list
            list of loss weight for [r_loss, a_loss, d_loss, kl_div_R, kl_div_A, kl_div_all].

        train_dataloader: DataLoader
            dataloader for training data.

        validation_dataloader: DataLoader
            dataloader for validation data.

        RNA_input_dim: int
            dimension of RNA data.

        ATAC_input_dim: int
            dimension of ATAC data.

        R_encoder_lr: float
            learning rate of RNA encoder, default 0.001.

        A_encoder_lr: float
            learning rate of ATAC encoder, default 0.001.

        R_decoder_lr: float
            learning rate of RNA decoder, default 0.001.

        A_decoder_lr: float
            learning rate of ATAC decoder, default 0.001.

        R_translator_lr: float
            learning rate of RNA pretrain translator, default 0.001.

        A_translator_lr: float
            learning rate of ATAC pretrain translator, default 0.001.

        translator_lr: float
            learning rate of translator, default 0.001.

        discriminator_lr: float
            learning rate of discriminator, default 0.005.

        R2R_pretrain_epoch: int
            max epoch for pretrain RNA autoencoder, default 100.

        A2A_pretrain_epoch: int
            max epoch for pretrain ATAC autoencoder, default 100.

        lock_encoder_and_decoder: bool
            lock the pretrained encoder and decoder or not, default False.

        translator_epoch: int
            max epoch for train translator, default 200.

        patience: int
            patience for loss on validation, default 50.

        r_loss
            loss function for RNA reconstruction, default nn.MSELoss(size_average=True).

        a_loss
            loss function for ATAC reconstruction, default nn.BCELoss(size_average=True).

        d_loss
            loss function for discriminator, default nn.BCELoss(size_average=True).

        output_path: str
            file path for model output, default None.

        seed: int
            set up the random seed, default 19193.

        kl_mean: bool
            size average for kl divergence or not, default True.

        R_pretrain_kl_warmup: int
            epoch of linear weight warm up for kl divergence in RNA pretrain, default 50.

        A_pretrain_kl_warmup: int
            epoch of linear weight warm up for kl divergence in ATAC pretrain, default 50.

        translation_kl_warmup: int
            epoch of linear weight warm up for kl divergence in translator pretrain, default 50.

        load_model: str
            the path for loading model if needed, else set it None, default None.

        """

        self.is_train_finished = False

        if output_path is None:
            output_path = "."

        if not load_model is None:
            my_logger.info(
                "load pretrained model from path: " + str(load_model) + "/model/"
            )
            self.RNA_encoder.load_state_dict(
                torch.load(load_model + "/model/RNA_encoder.pt")
            )
            self.ATAC_encoder.load_state_dict(
                torch.load(load_model + "/model/ATAC_encoder.pt")
            )
            self.RNA_decoder.load_state_dict(
                torch.load(load_model + "/model/RNA_decoder.pt")
            )
            self.ATAC_decoder.load_state_dict(
                torch.load(load_model + "/model/ATAC_decoder.pt")
            )
            self.translator.load_state_dict(
                torch.load(load_model + "/model/translator.pt")
            )
            self.discriminator_A.load_state_dict(
                torch.load(load_model + "/model/discriminator_A.pt")
            )
            self.discriminator_R.load_state_dict(
                torch.load(load_model + "/model/discriminator_R.pt")
            )

        if not seed is None:
            setup_seed(seed)

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.optimizer_R_encoder = torch.optim.Adam(
            self.RNA_encoder.parameters(), lr=R_encoder_lr
        )
        self.optimizer_A_encoder = torch.optim.Adam(
            self.ATAC_encoder.parameters(), lr=A_encoder_lr, weight_decay=0
        )
        self.optimizer_R_decoder = torch.optim.Adam(
            self.RNA_decoder.parameters(), lr=R_decoder_lr
        )
        self.optimizer_A_decoder = torch.optim.Adam(
            self.ATAC_decoder.parameters(), lr=A_decoder_lr, weight_decay=0
        )
        self.optimizer_R_translator = torch.optim.Adam(
            self.R_translator.parameters(), lr=R_translator_lr
        )
        self.optimizer_A_translator = torch.optim.Adam(
            self.A_translator.parameters(), lr=A_translator_lr
        )
        self.optimizer_translator = torch.optim.Adam(
            self.translator.parameters(), lr=translator_lr
        )
        self.optimizer_discriminator_A = torch.optim.SGD(
            self.discriminator_A.parameters(), lr=discriminator_lr
        )
        self.optimizer_discriminator_R = torch.optim.SGD(
            self.discriminator_R.parameters(), lr=discriminator_lr
        )

        """ eraly stop for model """
        self.early_stopping_R2R = EarlyStopping(patience=patience, verbose=False)
        self.early_stopping_A2A = EarlyStopping(patience=patience, verbose=False)
        self.early_stopping_all = EarlyStopping(patience=patience, verbose=False)

        if not os.path.exists(output_path /"model"):
            path_to_create = output_path / "model"
            # Parent directory is created if it does not exist.
            path_to_create.mkdir(parents=True, exist_ok=True)

        """ pretrain for RNA and ATAC """
        my_logger.info("RNA pretraining ...")
        pretrain_r_loss, pretrain_r_kl, pretrain_r_loss_val, pretrain_r_kl_val = (
            [],
            [],
            [],
            [],
        )
        with tqdm(total=R2R_pretrain_epoch, ncols=100) as pbar:
            pbar.set_description("RNA pretrain")
            for epoch in range(R2R_pretrain_epoch):
                (
                    pretrain_r_loss_,
                    pretrain_r_kl_,
                    pretrain_r_loss_val_,
                    pretrain_r_kl_val_,
                ) = ([], [], [], [])
                self.set_train()
                for idx, batch_samples in enumerate(self.train_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(
                        batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1
                    )

                    """ pretrain for RNA """
                    weight_temp = loss_weight.copy()
                    if epoch < R_pretrain_kl_warmup:
                        weight_temp[3] = loss_weight[3] * epoch / R_pretrain_kl_warmup

                    loss, reconstruct_loss, kl_div_r = self.forward_R2R(
                        RNA_input, r_loss, weight_temp[3], "train"
                    )
                    self.optimizer_R_encoder.zero_grad()
                    self.optimizer_R_decoder.zero_grad()
                    self.optimizer_R_translator.zero_grad()
                    loss.backward()
                    self.optimizer_R_encoder.step()
                    self.optimizer_R_decoder.step()
                    self.optimizer_R_translator.step()

                    pretrain_r_loss_.append(reconstruct_loss.item())
                    pretrain_r_kl_.append(kl_div_r.item())

                self.set_eval()
                for idx, batch_samples in enumerate(self.validation_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(
                        batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1
                    )

                    loss, reconstruct_loss, kl_div_r = self.forward_R2R(
                        RNA_input, r_loss, weight_temp[3], "test"
                    )

                    pretrain_r_loss_val_.append(reconstruct_loss.item())
                    pretrain_r_kl_val_.append(kl_div_r.item())

                pretrain_r_loss.append(np.mean(pretrain_r_loss_))
                pretrain_r_kl.append(np.mean(pretrain_r_kl_))
                pretrain_r_loss_val.append(np.mean(pretrain_r_loss_val_))
                pretrain_r_kl_val.append(np.mean(pretrain_r_kl_val_))

                self.early_stopping_R2R(
                    np.mean(pretrain_r_loss_val_), self, output_path
                )
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train="{:.4f}".format(np.mean(pretrain_r_loss_val_)),
                    val="{:.4f}".format(np.mean(pretrain_r_loss_)),
                )

                if self.early_stopping_R2R.early_stop:
                    my_logger.info(
                        "RNA pretraining early stop, validation loss does not improve in "
                        + str(patience)
                        + " epoches!"
                    )
                    self.RNA_encoder.load_state_dict(
                        torch.load(output_path / "model/RNA_encoder.pt")
                    )
                    self.RNA_decoder.load_state_dict(
                        torch.load(output_path / "model/RNA_decoder.pt")
                    )
                    self.R_translator.load_state_dict(
                        torch.load(output_path / "model/R_translator.pt")
                    )
                    break

        pretrain_a_loss, pretrain_a_kl, pretrain_a_loss_val, pretrain_a_kl_val = (
            [],
            [],
            [],
            [],
        )
        my_logger.info("ATAC pretraining ...")
        with tqdm(total=A2A_pretrain_epoch, ncols=100) as pbar:
            pbar.set_description("ATAC pretrain")
            for epoch in range(A2A_pretrain_epoch):
                (
                    pretrain_a_loss_,
                    pretrain_a_kl_,
                    pretrain_a_loss_val_,
                    pretrain_a_kl_val_,
                ) = ([], [], [], [])
                self.set_train()
                for idx, batch_samples in enumerate(self.train_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(
                        batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1
                    )

                    """ pretrain for ATAC """
                    weight_temp = loss_weight.copy()
                    if epoch < A_pretrain_kl_warmup:
                        weight_temp[4] = loss_weight[4] * epoch / A_pretrain_kl_warmup

                    loss, reconstruct_loss, kl_div_a = self.forward_A2A(
                        ATAC_input, a_loss, weight_temp[4], "train"
                    )
                    self.optimizer_A_encoder.zero_grad()
                    self.optimizer_A_decoder.zero_grad()
                    self.optimizer_A_translator.zero_grad()
                    loss.backward()
                    self.optimizer_A_encoder.step()
                    self.optimizer_A_decoder.step()
                    self.optimizer_A_translator.step()

                    pretrain_a_loss_.append(reconstruct_loss.item())
                    pretrain_a_kl_.append(kl_div_a.item())

                self.set_eval()
                for idx, batch_samples in enumerate(self.validation_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(
                        batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1
                    )

                    loss, reconstruct_loss, kl_div_a = self.forward_A2A(
                        ATAC_input, a_loss, weight_temp[4], "test"
                    )

                    pretrain_a_loss_val_.append(reconstruct_loss.item())
                    pretrain_a_kl_val_.append(kl_div_a.item())

                pretrain_a_loss.append(np.mean(pretrain_a_loss_))
                pretrain_a_kl.append(np.mean(pretrain_a_kl_))
                pretrain_a_loss_val.append(np.mean(pretrain_a_loss_val_))
                pretrain_a_kl_val.append(np.mean(pretrain_a_kl_val_))

                self.early_stopping_A2A(
                    np.mean(pretrain_a_loss_val_), self, output_path
                )
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train="{:.4f}".format(np.mean(pretrain_a_loss_val_)),
                    val="{:.4f}".format(np.mean(pretrain_a_loss_)),
                )

                if self.early_stopping_A2A.early_stop:
                    my_logger.info(
                        "ATAC pretraining early stop, validation loss does not improve in "
                        + str(patience)
                        + " epoches!"
                    )
                    self.ATAC_encoder.load_state_dict(
                        torch.load(output_path / "model/ATAC_encoder.pt")
                    )
                    self.ATAC_decoder.load_state_dict(
                        torch.load(output_path / "model/ATAC_decoder.pt")
                    )
                    self.A_translator.load_state_dict(
                        torch.load(output_path / "model/A_translator.pt")
                    )
                    break

        """ train for translator and discriminator """
        (
            train_loss,
            train_kl,
            train_discriminator,
            train_loss_val,
            train_kl_val,
            train_discriminator_val,
        ) = ([], [], [], [], [], [])
        my_logger.info("Integrative training ...")
        with tqdm(total=translator_epoch, ncols=100) as pbar:
            pbar.set_description("Integrative training")
            for epoch in range(translator_epoch):
                (
                    train_loss_,
                    train_kl_,
                    train_discriminator_,
                    train_loss_val_,
                    train_kl_val_,
                    train_discriminator_val_,
                ) = ([], [], [], [], [], [])
                self.set_train()
                for idx, batch_samples in enumerate(self.train_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(
                        batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1
                    )

                    """ train for discriminator """
                    loss_d = self.forward_discriminator(
                        batch_samples, RNA_input_dim, ATAC_input_dim, d_loss, "train"
                    )
                    self.optimizer_discriminator_R.zero_grad()
                    self.optimizer_discriminator_A.zero_grad()
                    loss_d.backward()
                    self.optimizer_discriminator_R.step()
                    self.optimizer_discriminator_A.step()

                    """ train for generator """
                    weight_temp = loss_weight.copy()
                    if epoch < translation_kl_warmup:
                        weight_temp[5] = loss_weight[5] * epoch / translation_kl_warmup
                    loss_d = self.forward_discriminator(
                        batch_samples, RNA_input_dim, ATAC_input_dim, d_loss, "train"
                    )
                    reconstruct_loss, kl_div, loss_g = self.forward_translator(
                        batch_samples,
                        RNA_input_dim,
                        ATAC_input_dim,
                        a_loss,
                        r_loss,
                        weight_temp,
                        "train",
                        kl_mean,
                    )

                    if loss_d.item() < 1.35:
                        loss_g -= loss_weight[2] * loss_d

                    self.optimizer_translator.zero_grad()
                    if not lock_encoder_and_decoder:
                        self.optimizer_R_encoder.zero_grad()
                        self.optimizer_A_encoder.zero_grad()
                        self.optimizer_R_decoder.zero_grad()
                        self.optimizer_A_decoder.zero_grad()
                    loss_g.backward()
                    self.optimizer_translator.step()
                    if not lock_encoder_and_decoder:
                        self.optimizer_R_encoder.step()
                        self.optimizer_A_encoder.step()
                        self.optimizer_R_decoder.step()
                        self.optimizer_A_decoder.step()

                    train_loss_.append(reconstruct_loss.item())
                    train_kl_.append(kl_div.item())
                    train_discriminator_.append(loss_d.item())

                self.set_eval()
                for idx, batch_samples in enumerate(self.validation_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(
                        batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1
                    )

                    """ test for discriminator """
                    loss_d = self.forward_discriminator(
                        batch_samples, RNA_input_dim, ATAC_input_dim, d_loss, "test"
                    )

                    """ test for generator """
                    loss_d = self.forward_discriminator(
                        batch_samples, RNA_input_dim, ATAC_input_dim, d_loss, "test"
                    )
                    reconstruct_loss, kl_div, loss_g = self.forward_translator(
                        batch_samples,
                        RNA_input_dim,
                        ATAC_input_dim,
                        a_loss,
                        r_loss,
                        weight_temp,
                        "train",
                        kl_mean,
                    )
                    loss_g -= loss_weight[2] * loss_d

                    train_loss_val_.append(reconstruct_loss.item())
                    train_kl_val_.append(kl_div.item())
                    train_discriminator_val_.append(loss_d.item())

                train_loss.append(np.mean(train_loss_))
                train_kl.append(np.mean(train_kl_))
                train_discriminator.append(np.mean(train_discriminator_))
                train_loss_val.append(np.mean(train_loss_val_))
                train_kl_val.append(np.mean(train_kl_val_))
                train_discriminator_val.append(np.mean(train_discriminator_val_))
                self.early_stopping_all(np.mean(train_loss_val_), self, output_path)

                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train="{:.4f}".format(np.mean(train_loss_val_)),
                    val="{:.4f}".format(np.mean(train_loss_)),
                )

                if self.early_stopping_all.early_stop:
                    my_logger.info(
                        "Integrative training early stop, validation loss does not improve in "
                        + str(patience)
                        + " epoches!"
                    )
                    self.RNA_encoder.load_state_dict(
                        torch.load(output_path / "model/RNA_encoder.pt")
                    )
                    self.ATAC_encoder.load_state_dict(
                        torch.load(output_path / "model/ATAC_encoder.pt")
                    )
                    self.RNA_decoder.load_state_dict(
                        torch.load(output_path / "model/RNA_decoder.pt")
                    )
                    self.ATAC_decoder.load_state_dict(
                        torch.load(output_path / "model/ATAC_decoder.pt")
                    )
                    self.translator.load_state_dict(
                        torch.load(output_path / "model/translator.pt")
                    )
                    self.discriminator_A.load_state_dict(
                        torch.load(output_path / "model/discriminator_A.pt")
                    )
                    self.discriminator_R.load_state_dict(
                        torch.load(output_path / "model/discriminator_R.pt")
                    )
                    break

        self.save_model_dict(output_path)

        self.is_train_finished = True

        record_loss_log(
            pretrain_r_loss,
            pretrain_r_kl,
            pretrain_r_loss_val,
            pretrain_r_kl_val,
            pretrain_a_loss,
            pretrain_a_kl,
            pretrain_a_loss_val,
            pretrain_a_kl_val,
            train_loss,
            train_kl,
            train_discriminator,
            train_loss_val,
            train_kl_val,
            train_discriminator_val,
            output_path,
        )

    def test(
        self,
        R_test_dataloader=None,
        A_test_dataloader=None,
        ATAC_data_obs=None,
        RNA_data_obs=None,
        model_path: str = None,
        load_model: bool = True,
        output_path: str = None,
        my_logger: logging.Logger = None,
    ):
        """
        Test for model.

        Parameters
        ----------
        R_test_dataloader : DataLoader
            RNA test data loader.
        A_test_dataloader : DataLoader
            ATAC test data loader.
        ATAC_data_obs : dict
            ATAC data observation information.
        RNA_data_obs : dict
            RNA data observation information.
        model_path : str
            Pre-trained model path.
        load_model : bool
            Whether to load the pre-trained model, default True.
        output_path : str
            Model output file path, default None.
        test_cluster : bool
            Whether to test cluster indices, default True.
        test_figure : bool
            Whether to test t-SNE figures, default True.
        test_expression : bool
            Whether to test expression indices, default True.
        test_generator : bool
            Whether to test generator indices, default True.
        output_data : bool
            Whether to output predicted data to files, default False.
        return_predict : bool
            Whether to return predicted results, if True, returns (A2R_predict, R2A_predict), default False.

        """

        if output_path is None:
            output_path = "."  # If output_path is None, set it to the current directory

        """ Load model from model_path if needed """
        if load_model:
            my_logger.info(
                "Loading trained model from path: " + str(model_path) + "/model"
            )
            self.RNA_encoder.load_state_dict(
                torch.load(model_path /"model/RNA_encoder.pt")
            )
            self.ATAC_encoder.load_state_dict(
                torch.load(model_path / "model/ATAC_encoder.pt")
            )
            self.RNA_decoder.load_state_dict(
                torch.load(model_path / "model/RNA_decoder.pt")
            )
            self.ATAC_decoder.load_state_dict(
                torch.load(model_path / "model/ATAC_decoder.pt")
            )
            self.translator.load_state_dict(
                torch.load(model_path / "model/translator.pt")
            )

        """ Load data """
        self.R_test_dataloader = R_test_dataloader
        self.A_test_dataloader = A_test_dataloader

        self.set_eval()
        my_logger.info("Getting predictions ...")
        """ Record the predicted data """
        R2A_predict = []
        A2R_predict = []
        with torch.no_grad():
            with tqdm(total=len(self.R_test_dataloader), ncols=100) as pbar:
                pbar.set_description("RNA to ATAC predicting...")
                for idx, batch_samples in enumerate(self.R_test_dataloader):
                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    R2 = self.RNA_encoder(batch_samples)
                    R2R, R2A, mu_r, sigma_r = self.translator.test_model(R2, "RNA")
                    R2A = self.ATAC_decoder(R2A)

                    R2A_predict.append(R2A.cpu())

                    time.sleep(0.01)
                    pbar.update(1)

        with torch.no_grad():
            with tqdm(total=len(self.A_test_dataloader), ncols=100) as pbar:
                pbar.set_description("ATAC to RNA predicting...")
                for idx, batch_samples in enumerate(self.A_test_dataloader):
                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    A2 = self.ATAC_encoder(batch_samples)
                    A2R, A2A, mu_a, sigma_a = self.translator.test_model(A2, "ATAC")
                    A2R = self.RNA_decoder(A2R)

                    A2R_predict.append(A2R.cpu())

                    time.sleep(0.01)
                    pbar.update(1)

        pred_atac = tensor2adata(R2A_predict)
        pred_rna = tensor2adata(A2R_predict)

        pred_atac.obs = ATAC_data_obs
        pred_rna.obs = RNA_data_obs
        
        true_atac = sc.AnnData(X=csr_matrix(self.A_test_dataloader.dataset.data))
        true_rna = sc.AnnData(X=csr_matrix(self.R_test_dataloader.dataset.data))
        true_atac.obs = ATAC_data_obs
        true_rna.obs = RNA_data_obs
        
        return pred_rna, true_rna, pred_atac, true_atac