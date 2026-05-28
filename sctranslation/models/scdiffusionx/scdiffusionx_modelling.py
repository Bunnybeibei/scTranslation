import os
from pathlib import Path
import uuid
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from .scdiffusionx_dataloader import RNAseqLoader
from .Autoencoder.models.base.encoder_model import EncoderModel
from .DiffusionBackbone.multimodal_unet import MultimodalUNet
from .DiffusionBackbone import multimodal_gaussian_diffusion as gd
from .DiffusionBackbone.multimodal_respace import SpacedDiffusion, space_timesteps

import numpy as np
from pathlib import Path
import torch.distributed as dist
from scvi.distributions import NegativeBinomial
from torch.distributions import Poisson, Bernoulli, MultivariateNormal

from .scdiffusionx_dataloader import RNAseqLoader
from .DiffusionBackbone import dist_util
from .DiffusionBackbone.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
from .Autoencoder.models.base.encoder_model import EncoderModel
from sklearn.preprocessing import LabelEncoder
from torch.distributions import Normal
import yaml
from scipy.sparse import csr_matrix

# Some general settings for the run
os.environ["WANDB__SERVICE_WAIT"] = "300"
torch.autograd.set_detect_anomaly(True)

def get_array(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, csr_matrix):
        return x.toarray()
    else:
        raise ValueError(f"Unsupported type: {type(x)}")

class EncoderEstimator:
    """Class for training and using the cfgen model."""
    
    def __init__(self, args, train_sc=None, val_sc=None, test_sc=None, modal2='a', n_cat=None):
        """
        Initialize encoder Estimator.

        Args:
            args (Args): Configuration hyperparameters for the model.
        """
        # args is a dictionary containing the configuration hyperparameters 
        self.args = args
        
        # date and time to name run 
        self.unique_id = str(uuid.uuid4())
        
        # Initialize training directory         
        TRAINING_FOLDER = Path(self.args.training_config.chekpoint_path).resolve()
        self.training_dir = TRAINING_FOLDER / self.args.logger.project
        print("Create the training folders...")
        self.training_dir.mkdir(parents=True, exist_ok=True)

        # Set device for training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Initialize data module...")
        self.init_datamodule(train_sc, val_sc, test_sc, modal2=modal2, n_cat=n_cat)  # Initialize the data module  
        self.get_fixed_rna_model_params()  # Initialize the data derived model parameters 
        self.init_trainer()
        
        print("Initialize model...")
        self.init_model()  # Initialize the model

    def init_datamodule(self, train_sc=None, val_sc=None, test_sc=None, modal2='a', n_cat=None):
        """
        Initialization of the data module.
        """        
        # Initialize the dataset using RNAseqLoader
        self.train_data = RNAseqLoader(data_r=train_sc.ad['r'],
                                    data_modal2=train_sc.ad[modal2],
                                    modal2 = modal2,
                                    layer_key=self.args.dataset.layer_key,
                                    covariate_keys=self.args.dataset.covariate_keys,
                                    subsample_frac=self.args.dataset.subsample_frac, 
                                    encoder_type=self.args.dataset.encoder_type,
                                    multimodal=self.args.dataset.multimodal, 
                                    is_binarized=self.args.dataset.is_binarized)
        
        self.n_cat = n_cat
  
        self.valid_data = RNAseqLoader(
                                data_r=val_sc.ad['r'],
                                data_modal2=val_sc.ad[modal2],
                                modal2=modal2,
                                layer_key=self.args.dataset.layer_key,
                                covariate_keys=self.args.dataset.covariate_keys,
                                subsample_frac=self.args.dataset.subsample_frac,
                                encoder_type=self.args.dataset.encoder_type,
                                multimodal=self.args.dataset.multimodal,
                                is_binarized=self.args.dataset.is_binarized)
        self.dataset = self.train_data
        # Initialize the data loaders for training and validation
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=True,
                                                            num_workers=4, 
                                                            drop_last=True)
        
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=False,
                                                            num_workers=4, 
                                                            drop_last=True)
    
    def get_fixed_rna_model_params(self):
        """Set the model parameters extracted from the data loader object.
        """
        if not self.dataset.multimodal:
            # Single modality: get the gene dimension from the dataset
            self.gene_dim = self.dataset.X.shape[1] 
        else:
            # Multimodal: get the gene dimensions for each modality
            self.gene_dim = {mod: self.dataset.X[mod].shape[1] for mod in self.dataset.X}

    def init_trainer(self):
        """
        Initialize Trainer.
        """
        # Callbacks for saving checkpoints 
        checkpoint_callback = ModelCheckpoint(dirpath=self.training_dir / "checkpoints", 
                                                **self.args.checkpoints.to_dict())
        callbacks = [checkpoint_callback]
        
        # Early stopping callbacks
        if self.args.training_config.use_early_stopping:
            early_stopping_callbacks = EarlyStopping(**self.args.early_stopping.to_dict())
            callbacks.append(early_stopping_callbacks)
        
        # Logger settings 
        self.logger = WandbLogger(save_dir=self.training_dir,
                                    name=self.unique_id, 
                                    **self.args.logger.to_dict())
        
        # Initialize the PyTorch Lightning trainer with the specified callbacks and logger
        self.trainer_generative = Trainer(callbacks=callbacks, 
                                          default_root_dir=self.training_dir, 
                                          logger=self.logger,
                                          **self.args.trainer.to_dict())

    def init_model(self):
        """Initialize the encoder model.
        """
        # Initialize the model using the provided arguments and data-derived parameters
        self.encoder_model = EncoderModel(in_dim=self.gene_dim,
                                          n_cat=self.n_cat,
                                          conditioning_covariate=self.args.dataset.theta_covariate, 
                                          encoder_type=self.args.dataset.encoder_type,
                                          **self.args.encoder.to_dict())
        print("Encoder architecture", self.encoder_model)

    def train(self):
        """
        Train the generative model using the provided trainer.
        """
        # Train the model using the training and validation data loaders
        self.trainer_generative.fit(
            self.encoder_model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader)
    
    def test(self):
        """
        Test the generative model.
        """
        # Test the model using the validation data loader
        self.trainer_generative.test(
            self.encoder_model,
            dataloaders=self.valid_dataloader)
        
def create_model(
    rna_dim,
    atac_dim,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    cross_attention_resolutions="2,4,8",
    cross_attention_windows="1,4,8",
    cross_attention_shift=True,
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    use_fp16=False,
    resblock_updown=True,
    num_class=None,
):
    
    image_size = rna_dim[-1] 
    channel_mult = (4, 2, 1)

    cross_attention_resolutions = [int(i) for i in cross_attention_resolutions.split(',')]
    cross_attention_windows = [int(i) for i in cross_attention_windows.split(',')]

    return MultimodalUNet(
        rna_dim=rna_dim,
        atac_dim=atac_dim,
        model_channels=num_channels,
        video_out_channels=rna_dim[-1],
        audio_out_channels=atac_dim[-1],
        num_res_blocks=num_res_blocks,
        cross_attention_resolutions=cross_attention_resolutions,
        cross_attention_windows=cross_attention_windows,
        cross_attention_shift=cross_attention_shift,

        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=num_class,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
    )
def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
      

def create_model_and_diffusion(
    rna_dim,
    atac_dim,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    cross_attention_resolutions,
    cross_attention_windows,
    cross_attention_shift,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    class_cond=False,
    num_class=None,
):
    model = create_model(
        rna_dim=rna_dim,
        atac_dim=atac_dim,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        cross_attention_resolutions=cross_attention_resolutions,
        cross_attention_windows=cross_attention_windows,
        cross_attention_shift=cross_attention_shift,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        num_class=num_class,
       
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

def get_size_factor(type_index, covariate_keys = "cell_type", size_factor_statistics:dict=None, device=torch.device('cuda')):
        covariate_indices = {}
        covariate_indices[covariate_keys] = type_index

        mean_size_factor, sd_size_factor = size_factor_statistics["mean"][covariate_keys], size_factor_statistics["sd"][covariate_keys]
        mean_size_factor, sd_size_factor = mean_size_factor[covariate_indices[covariate_keys]], sd_size_factor[covariate_indices[covariate_keys]]
        size_factor_dist = Normal(loc=mean_size_factor, scale=sd_size_factor)
        log_size_factor = size_factor_dist.sample().view(-1, 1)
        size_factor = torch.exp(log_size_factor).to(device)
        return {"rna": size_factor}

@torch.no_grad()
def generate(multimodal_model, multimodal_diffusion, adata_rna=None, adata_modal2=None, modal2=None, model_path=None, args=None, saved_path=None):

    if args.use_fp16:
        multimodal_model.convert_to_fp16()
    multimodal_model.eval()

    model_name = model_path.split('/')[-1]
    args.clip_denoised = True

    groups= 0
    multimodal_save_path = os.path.join(saved_path, model_name, 'original')
    audio_save_path = os.path.join(saved_path)
    img_save_path = os.path.join(saved_path)
    if dist.get_rank() == 0:
        os.makedirs(multimodal_save_path, exist_ok=True)
        os.makedirs(audio_save_path, exist_ok=True)
        os.makedirs(img_save_path, exist_ok=True)

    modality2 = {'a': 'atac', 'p': 'adt'}.get(modal2)
    if modality2 is None:
        raise ValueError(
            f"Unsupported modality: {modal2}. Expected 'a' (ATAC) or 'p' (ADT)."
        )

    mdata = {'rna': adata_rna, modality2: adata_modal2}

    dataset = RNAseqLoader(data_r=adata_rna,
                           data_modal2=mdata[modality2],
                           modal2=modal2,
                           layer_key='X_counts',
                           covariate_keys=[args.condition],
                           subsample_frac=1,
                           encoder_type='learnt_autoencoder',
                           multimodal=True,
                           is_binarized=True)

    size_factor_statistics = {"mean": dataset.log_size_factor_mu,
                              "sd": dataset.log_size_factor_sd}

    labels = mdata['rna'].obs[args.condition].values
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    classes_all = label_encoder.transform(labels)
    n_cat = int(label_encoder.classes_.shape[0])

    with open(args.encoder_config, 'r') as file:
        autoencoder_args = yaml.safe_load(file)

    in_dim = {modality2: mdata[modality2].shape[1], 'rna': mdata['rna'].shape[1]}
    encoder_model = EncoderModel(in_dim=in_dim,
                                 n_cat=n_cat,
                                 conditioning_covariate=args.condition,
                                 encoder_type='learnt_autoencoder',
                                 **autoencoder_args)

    weight = torch.load(args.ae_path, weights_only=False)["state_dict"]
    new_weight = {}
    for k in encoder_model.state_dict().keys():
        if k not in weight.keys():
            print('Remove key {} from encoder_model.state_dict()'.format(k))
        else:
            new_weight[k] = weight[k]
    encoder_model.load_state_dict(new_weight)
    encoder_model.to(dist_util.dev())

    batch = {}
    gt_rna = torch.tensor(get_array(mdata['rna'].X), device=dist_util.dev())
    gt_modal2 = torch.tensor(get_array(mdata[modality2].X), device=dist_util.dev())
    batch["X_norm"] = {'rna': gt_rna, modality2: gt_modal2}
    z = encoder_model.encode(batch)
    # Conditioning comes from the *source* modality (everything except gen_mode).
    noise_init = z[next(s for s in z.keys() if s != args.gen_mode)]

    npzfile = np.load('/'.join(args.ae_path.split('/')[:-2]) + '/norm_factor.npz')
    if args.gen_mode == 'rna':
        # Source is the second modality -> use modal-2 std (modality-aware key
        # if present, falling back to the legacy ``atac_std``).
        modal2_key = f"{modality2}_std"
        std = npzfile[modal2_key] if modal2_key in npzfile.files else npzfile["atac_std"]
    else:
        # Source is RNA.
        std = npzfile['rna_std']
    noise_init = noise_init / torch.tensor(std, device=noise_init.device)

    videos = []
    audios = []
    all_labels = []

    # while groups * args.batch_size *  dist.get_world_size()< args.all_save_num: 
    sample_num = noise_init.shape[0]
    num_iteration = int(sample_num/args.batch_size)+1
    args.gen_times = 5
    for i in list(range(num_iteration))*args.gen_times:

        model_kwargs = {}

        x_T_init = noise_init[i*args.batch_size:(i+1)*args.batch_size]
        if len(x_T_init) == 0:
            continue
        
        if args.gen_mode == 'rna':
            model_kwargs["audio"] = x_T_init.unsqueeze(1).to(dist_util.dev())
        else:
            model_kwargs["video"] = x_T_init.unsqueeze(1).to(dist_util.dev())
        if args.class_cond:
            classes = classes_all[i*args.batch_size:(i+1)*args.batch_size]  # generated random cell type
            classes = torch.tensor(classes, device=dist_util.dev(), dtype=torch.int)
            model_kwargs["label"] = classes

        shape = {"video":(args.batch_size if i!=num_iteration-1 else x_T_init.shape[0], *args.rna_dim), \
                "audio":(args.batch_size if i!=num_iteration-1 else x_T_init.shape[0], *args.atac_dim)
            }
        if args.sample_fn == 'dpm_solver':
            # sample_fn = multimodal_dpm_solver
            # sample = sample_fn(shape = shape, \
            #     model_fn = multimodal_model, steps=args.timestep_respacing)

            dpm_solver = multimodal_DPM_Solver(model=multimodal_model, \
                alphas_cumprod=torch.tensor(multimodal_diffusion.alphas_cumprod, dtype=torch.float32))
            x_T = {"video":torch.randn(shape["video"]).to(dist_util.dev()), \
                    "audio":torch.randn(shape["audio"]).to(dist_util.dev())}
            sample = dpm_solver.sample(
                x_T,
                steps=20,
                order=3,
                skip_type="logSNR",
                method="singlestep",
            )

        elif args.sample_fn == 'dpm_solver++':
            dpm_solver = multimodal_DPM_Solver(model=multimodal_model, \
                alphas_cumprod=torch.tensor(multimodal_diffusion.alphas_cumprod, dtype=torch.float32), \
                    predict_x0=True, thresholding=True)
            
            x_T = {"video":torch.randn(shape["video"]).to(dist_util.dev()), \
                    "audio":torch.randn(shape["audio"]).to(dist_util.dev())}
            sample = dpm_solver.sample(
                x_T,
                steps=20,
                order=2,
                skip_type="logSNR",
                method="adaptive",
            )
        else:
            sample_fn = (
                multimodal_diffusion.conditional_p_sample_loop if  args.sample_fn=="ddpm" else multimodal_diffusion.ddim_sample_loop
            )

            sample = sample_fn(
                multimodal_model,
                shape = shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                # noise=x_T_init,
                # gen_mode=args.gen_mode,
                use_fp16 = args.use_fp16,
                class_scale=args.classifier_scale
            )

        video = sample["video"]
        audio = sample["audio"]              

        all_videos = video.detach().cpu().numpy()
        all_audios = audio.detach().cpu().numpy()

        if args.class_cond:
            all_labels.append(classes.cpu().numpy())
            
        videos.append(all_videos)
        audios.append(all_audios)

        groups += 1

        dist.barrier()
        torch.cuda.empty_cache()

    rna_seq = np.concatenate(videos)
    atac_seq = np.concatenate(audios)
    type_index = np.concatenate(all_labels) if all_labels != [] else np.zeros(atac_seq.shape[0])
        
    video_output_path = os.path.join(img_save_path, f"RNA_{dist.get_rank()}.npz")
    audio_output_path = os.path.join(audio_save_path, f"{modality2.upper()}_{dist.get_rank()}.npz")
    
    # np.savez(video_output_path,data=rna_seq,label=type_index)
    # np.savez(audio_output_path,data=atac_seq,label=type_index)
    
    # load norm factor for encoder (modality-aware key with legacy fallback)
    npzfile = np.load('/'.join(args.ae_path.split('/')[:-2]) + '/norm_factor.npz')
    rna_std = npzfile['rna_std']
    modal2_std_key = f"{modality2}_std"
    modal2_std = npzfile[modal2_std_key] if modal2_std_key in npzfile.files else npzfile["atac_std"]
    z = {
        'rna': torch.tensor(rna_seq * rna_std).squeeze(1).cuda(),
        modality2: torch.tensor(atac_seq * modal2_std).squeeze(1).cuda(),
    }

    # get size factor and decode
    size_factor = get_size_factor(type_index = torch.tensor(type_index,dtype=torch.int), covariate_keys=args.condition, size_factor_statistics=size_factor_statistics)
    mu_hat = encoder_model.decode(z, size_factor)

    sample = {}  # containing final samples 
    for mod in mu_hat:
        if mod=="rna":  
            distr = NegativeBinomial(mu=mu_hat[mod], theta=torch.exp(encoder_model.theta).cuda())
        elif mod == 'atac':  # if mod is atac
            if not encoder_model.is_binarized:
                distr = Poisson(rate=mu_hat[mod])
            else:
                distr = Bernoulli(probs=mu_hat[mod])
        elif mod == 'adt':
            # cov is map a one-dimensional vector to a diagonal covariance matrix
            device = mu_hat[mod]['logvar'].device
            cov = torch.diag_embed(mu_hat[mod]['logvar'].exp().to(device))
            distr = MultivariateNormal(loc=mu_hat[mod]['mu'].to(device), covariance_matrix=cov)
        sample[mod] = distr.sample() 

    rna = sample['rna'].detach().cpu().numpy()
    modal2_out = sample[modality2].detach().cpu().numpy()

    rna = rna.reshape(args.gen_times, -1, rna.shape[1]).mean(axis=0)
    modal2_out = modal2_out.reshape(args.gen_times, -1, modal2_out.shape[1]).mean(axis=0)
    return rna, modal2_out
