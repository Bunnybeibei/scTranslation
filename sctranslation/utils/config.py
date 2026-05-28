"""
generate config  for different models
"""

import yaml
import torch.nn as nn


class Config:
    def __init__(self, config_file, model_name):
        self.model_name = model_name
        # load config file
        with open(config_file) as f_in:
            self.config = yaml.safe_load(f_in)
        self.check_config_type()
        # print config
        print(self.config)

    def check_config_type(self):
        config_types = dict(
            n_peaks=int,
            min_mz=float,
            max_mz=float,
            min_intensity=float,
            remove_precursor_tol=float,
            max_charge=int,
            precursor_mass_tol=float,
            isotope_error_range=lambda min_max: (int(min_max[0]), int(min_max[1])),
            warmup_iters=int,
            max_iters=int,
            num_sanity_val_steps=int,
            learning_rate=float,
            weight_decay=float,
            max_epochs=int,
            save_weights_only=bool,
            model_save_folder_path=str,
            logger_save_path=str,
            val_check_interval=int,
            check_val_every_n_epoch=int,
            n_workers=int,
            save_top_k=int,
            devices=int,
        )
        for k, t in config_types.items():
            try:
                if self.config[k] is not None:
                    self.config[k] = t(self.config[k])
            except (TypeError, ValueError) as e:
                print("Incorrect type for configuration value %s: %s", k, e)
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")
        # check model config type
        for key, value in self.config.items():
            setattr(self, key, value)
        self.check_model_config_type(self.model_name)

    def check_model_config_type(self, model_name):
        if model_name == "scButterfly":
            self.check_scButterfly_config_type()
        elif model_name == "multiDGD":
            self.check_multiDGD_config_type()
        elif model_name == "JAMIE":
            self.check_JAMIE_config_type()
        elif model_name == "BABEL":
            self.check_BABEL_config_type()
        elif model_name == "scPair":
            self.check_scPair_config_type()
        elif model_name == "scDiffusionX":
            self.check_scDiffusionX_config_type()

    def check_scButterfly_config_type(self):
        config_types = dict(
            # Model parameters
            R_encoder_nlayer=int,
            A_encoder_nlayer=int,
            R_decoder_nlayer=int,
            A_decoder_nlayer=int,
            R_encoder_dim_list=list,
            A_encoder_dim_list=list,
            R_decoder_dim_list=list,
            A_decoder_dim_list=list,
            R_encoder_act_list=list,
            A_encoder_act_list=list,
            R_decoder_act_list=list,
            A_decoder_act_list=list,
            translator_embed_dim=int,
            translator_input_dim_r=int,
            translator_input_dim_a=int,
            translator_embed_act_list=list,
            discriminator_nlayer=int,
            discriminator_dim_list_R=list,
            discriminator_dim_list_A=list,
            discriminator_act_list=list,
            dropout_rate=float,
            R_noise_rate=float,
            A_noise_rate=float,
            # Training parameters
            R_encoder_lr=float,
            A_encoder_lr=float,
            R_decoder_lr=float,
            A_decoder_lr=float,
            R_translator_lr=float,
            A_translator_lr=float,
            translator_lr=float,
            discriminator_lr=float,
            R2R_pretrain_epoch=int,
            A2A_pretrain_epoch=int,
            lock_encoder_and_decoder=bool,
            translator_epoch=int,
            patience=int,
            kl_mean=bool,
            R_pretrain_kl_warmup=int,
            A_pretrain_kl_warmup=int,
            translation_kl_warmup=int,
            load_model=str,  # set to None if not loading
            batch_size=int,
            use_hvg_flag=bool,
            fpeaks=float,
        )

        # Activation functions
        activation_functions = {
            "LeakyReLU": nn.LeakyReLU(),
            "Sigmoid": nn.Sigmoid(),
        }

        for k, t in config_types.items():
            try:
                if self.config["scButterfly"][k] is not None:
                    if t == list:
                        # If the key is an activation function, convert the string to the actual function
                        if k in [
                            "R_encoder_act_list",
                            "A_encoder_act_list",
                            "R_decoder_act_list",
                            "A_decoder_act_list",
                            "translator_embed_act_list",
                            "discriminator_act_list",
                        ]:
                            self.config["scButterfly"][k] = [
                                activation_functions.get(str(act), act)
                                for act in self.config["scButterfly"][k].split(",")
                            ]
                        else:  # Otherwise, convert the string to a list of integers
                            self.config["scButterfly"][k] = [
                                int(i)
                                for i in self.config["scButterfly"][k].split(",")
                                if len(i) > 0
                            ]

                    else:
                        self.config["scButterfly"][k] = t(self.config["scButterfly"][k])

            except (TypeError, ValueError) as e:
                print(f"Incorrect type for configuration value {k}: {e}")
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")

        for key, value in self.config["scButterfly"].items():
            setattr(self, key, value)

    def check_multiDGD_config_type(self):
        """
          n_top_genes: 100
        batch_size: 128
        max_epochs: 2 # 2--> 500
        """
        config_types = dict(
            # Model parameters
            fpeaks=float,
            n_top_genes=int,
            batch_size=int,
            max_epochs=int,
            n_jobs=int,
        )

        for k, t in config_types.items():
            try:
                if self.config["multiDGD"][k] is not None:
                    self.config["multiDGD"][k] = t(self.config["multiDGD"][k])
            except (TypeError, ValueError) as e:
                print(f"Incorrect type for configuration value {k}: {e}")
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")
        
        for key, value in self.config["multiDGD"].items():
            setattr(self, key, value)
            
    def check_JAMIE_config_type(self):

        config_types = dict(
            # Model parameters
            fpeaks=float,
            n_top_genes=int,
            batch_size=int,
            dist_method=str,
            dropout=float,
            epoch_DNN=int,
            log_DNN=int,
            loss_weights=list,
            min_epochs=int,
            output_dim=int,
            pca_dim=list,
            use_early_stop=bool,
        )
        
        for k, t in config_types.items():
            try:
                if self.config["JAMIE"][k] is not None:
                    self.config["JAMIE"][k] = t(self.config["JAMIE"][k])
            except (TypeError, ValueError) as e:
                print(f"Incorrect type for configuration value {k}: {e}")
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")

        for key, value in self.config["JAMIE"].items():
            setattr(self, key, value)
            
    def check_BABEL_config_type(self):
        config_types = dict(
            # Model parameters
            n_top_genes=int,
            fpeaks=float,
            nofilter=bool,
            linear=bool,
            clustermethod=str,
            validcluster=int,
            testcluster=int,
            naive=bool,
            hidden=list,
            pretrain=str,
            lossweight=list,
            optim=str,
            lr=list,
            batchsize=list,
            earlystop=int,
            device=int,
            ext=str,
            use_hvg_flag=bool,
        )

        for k, t in config_types.items():
            try:
                if self.config["BABEL"][k] is not None:
                    self.config["BABEL"][k] = t(self.config["BABEL"][k])
            except (TypeError, ValueError) as e:
                print(f"Incorrect type for configuration value {k}: {e}")
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")

        for key, value in self.config["BABEL"].items():
            setattr(self, key, value)
            
    def check_scPair_config_type(self):
        config_types = dict(
            # Model parameters
            fpeaks=float,
            use_hvg_flag=bool,
            n_top_genes=int,
            batchnorm=bool,
            dropout_rate=float,
            hidden_layer=list,
            infer_library_size_atac=bool,
            infer_library_size_rna=bool,
            layernorm=bool,
            learning_rate_prediction=float,
            max_epochs=int,
            sample_factor_atac=bool,
            sample_factor_rna=bool,
        )

        for k, t in config_types.items():
            try:
                if self.config["scPair"][k] is not None:
                    self.config["scPair"][k] = t(self.config["scPair"][k])
            except (TypeError, ValueError) as e:
                print(f"Incorrect type for configuration value {k}: {e}")
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")

        for key, value in self.config["scPair"].items():
            setattr(self, key, value)
    
    def check_scDiffusionX_config_type(self):

        config_types = dict(
            # Model parameters
            use_hvg_flag=bool,
            n_top_genes=int,
            fpeaks=float,
            data_dir=str,
            schedule_sampler=str,
            lr=float,
            t_lr=float,
            weight_decay=float,
            lr_anneal_steps=int,
            batch_size=int,
            num_workers=int,
            microbatch=int,
            ema_rate=str,
            log_interval=int,
            devices=int,
            save_interval=int,
            resume_checkpoint=str,
            use_fp16=bool,
            fp16_scale_growth=float,
            use_db=bool,
            sample_fn=str,
            frame_gap=int,
            num_class=int,
            condition=str,
            rna_dim=list,
            atac_dim=list,
            num_channels=int,
            num_res_blocks=int,
            num_heads=int,
            num_heads_upsample=int,
            num_head_channels=int,
            cross_attention_resolutions=str,
            cross_attention_windows=str,
            cross_attention_shift=bool,
            channel_mult=str,  # Assuming this is a string
            dropout=float,
            class_cond=bool,
            use_checkpoint=bool,
            use_scale_shift_norm=bool,
            resblock_updown=bool,
            learn_sigma=bool,
            diffusion_steps=int,
            noise_schedule=str,  # Assuming this is a string
        )
        
        for k, t in config_types.items():
            try:
                if self.config["scDiffusionX"][k] is not None:
                    self.config["scDiffusionX"][k] = t(self.config["scDiffusionX"][k])
            except (TypeError, ValueError) as e:
                print(f"Incorrect type for configuration value {k}: {e}")
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")

        for key, value in self.config["scDiffusionX"].items():
            setattr(self, key, value)
            
    def to_dict(self, k_list=None):
        """将对象转换回字典"""
        return {key: value.to_dict() if isinstance(value, Config) else value for key, value in self.__dict__.items() if k_list is None or key in k_list}