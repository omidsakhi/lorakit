import contextlib
from functools import partial
import gc
import itertools
import json
import math
import os
from pathlib import Path
import shutil
from typing import List
import torch
from tqdm import tqdm
from lorakit.callbacks import BaseCallback, LossCallback, ProgressCallback
from lorakit.datasets import collate_fn, DreamBoothDataset
from lorakit.jobs import BaseJob
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
import logging
import transformers
import diffusers
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, UNet2DConditionModel, DDIMScheduler
from transformers import AutoTokenizer, PretrainedConfig
from huggingface_hub import hf_hub_download
from peft import LoraConfig, set_peft_model_state_dict
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from peft.utils import get_peft_model_state_dict
from diffusers.loaders import LoraLoaderMixin
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
import torch.nn.functional as F
import time

def _flush():
    torch.cuda.empty_cache()
    gc.collect()

# copied from train_xl.py
def _tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

# copied from train_xl.py
def _encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = _tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def _get_sigmas(accelerator, noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def _compute_time_ids(original_size, crops_coords_top_left, resolution, device, dtype):
    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings (when `train_text_encoder` is not True)
    # pooled text embeddings
    # time ids
    # This function computes time IDs for Stable Diffusion XL (SDXL) training
    # It's adapted from the _get_add_time_ids method in StableDiffusionXLPipeline
    
    # Set the target size to a fixed resolution (presumably defined elsewhere)
    target_size = [resolution, resolution]
    
    # Combine original size, crop coordinates, and target size into a single list
    
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    
    # Convert the list to a PyTorch tensor and add a batch dimension

    add_time_ids = torch.tensor([add_time_ids])
    
        
    # Move the tensor to the appropriate device and cast to the correct dtype
    add_time_ids = add_time_ids.to(device=device, dtype=dtype)
    
    return add_time_ids

def _save_model_hook(accelerator, unet_type, text_encoder_1_type, text_encoder_2_type, models, weights, output_folder):
    if accelerator.is_main_process:
        unet_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None
        for model in models:
            if isinstance(model, unet_type):
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            elif isinstance(model, text_encoder_1_type):
                text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)
                )
            elif isinstance(model, text_encoder_2_type):
                text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)
                )
            else:
                raise ValueError(f"Unexpected model type: {model.__class__}")
            
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()
        
        StableDiffusionXLPipeline.save_lora_weights(
            output_folder,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )

class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.model = model.to(device)
        self.decay = decay
        self.device = device
        self.ema_weights = {}
        self.update(model)

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name not in self.ema_weights:
                        self.ema_weights[name] = param.data.clone().to(self.device)
                    else:
                        self.ema_weights[name] = self.decay * self.ema_weights[name] + (1 - self.decay) * param.data.to(self.device)

    def apply(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.ema_weights[name])

class TrainJob(BaseJob):

    def __init__(self, config, version, name, root_folder):
        super().__init__(version, name, root_folder)

        self._device = config.get('device', 'cpu')
        if torch.cuda.is_available():
            if self._device == 'cpu':
                print("Warning: CUDA is available, but CPU is being used. Consider using a GPU for faster training.")
        else:
            if self._device != 'cpu':
                raise RuntimeError("CUDA is not available, but GPU device was specified. Please use a machine with CUDA support or specify 'cpu' as the device.")
        
        print(f"Using device: {self._device}")
        
        self._allow_tf32 = config.get('allow_tf32', False)
        
        if self._allow_tf32:
            print("Using TF32 for faster training on Ampere GPUs")
            # Enable TF32 for faster training on Ampere GPUs,
            # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices            
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self._matmul_precision = config.get('matmul_precision', None)
        if self._matmul_precision:
            print(f"Using matmul precision: {self._matmul_precision}")
            torch.set_float32_matmul_precision(self._matmul_precision)        
        
        self._cudnn_benchmark = config.get('cudnn_benchmark', None)
        if self._cudnn_benchmark:
            print("Using cuDNN benchmark")
            torch.backends.cudnn.benchmark = self._cudnn_benchmark

        self._logging_path = Path(self._experiment_folder, "logs")

        self._train_config = config.get('train', None)
        if self._train_config is None:
            raise ValueError("train is required")
        
        self._snr_gamma = self._train_config.get('snr_gamma', None)

        self._do_edm_style_training = self._train_config.get('do_edm_style_training', False)
        if self._do_edm_style_training and self._snr_gamma is not None:
            raise ValueError("Cannot specify both do_edm_style_training and snr_gamma")
        
        dtype_str = self._train_config.get('dtype', None)
        if dtype_str is None:
            raise ValueError("dtype is required")
        dtype_str = dtype_str.lower()
        if dtype_str == 'bfloat16' or dtype_str == 'bf16':
            self._dtype = torch.bfloat16
            self._mixed_precision = 'bf16'
        elif dtype_str == 'fp16' or dtype_str == 'float16':
            self._dtype = torch.float16
            self._mixed_precision = 'fp16'
        elif dtype_str == 'float32' or dtype_str == 'fp32':
            self._dtype = torch.float32
            self._mixed_precision = 'no'
        else:
            raise ValueError("Invalid dtype. Supported dtypes are bf16, fp16, fp32, and float32.")

        self._gradient_accumulation_steps = self._train_config.get('gradient_accumulation_steps', 1)
        if self._gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")        
        
        self._train_seed = self._train_config.get('seed', None)
        if self._train_seed is not None:
            print(f"Using seed: {self._train_seed}")
        else:
            print("No seed specified, using random seed")

        self._model_config = config.get('model', None)
        if self._model_config is None:
            raise ValueError("model is required")
        
        self._model_name_or_path = self._model_config.get('name_or_path', None)
        if self._model_name_or_path is None:
            raise ValueError("name_or_path is required for model")
        
        self._local_files_only = self._model_config.get('local_files_only', None)
        if self._local_files_only is not None:
            print(f"Using local files only: {self._local_files_only}")
        
        self._revision = self._model_config.get('revision', None)
        if self._revision is not None:
            print(f"Using revision: {self._revision}")

        self._variant = self._model_config.get('variant', None)
        if self._variant is not None:
            print(f"Using variant: {self._variant}")

        self._gradient_checkpointing = self._train_config.get('gradient_checkpointing', False)
        if self._gradient_checkpointing:
            print("Using gradient checkpointing")

        self._train_text_encoder = self._train_config.get('train_text_encoder', False)
        if self._train_text_encoder:
            print("Text encoder will be fine-tuned")
        
        self._unet_lora = self._train_config.get('lora', self._train_config.get('unet_lora', None))
        if self._unet_lora is None:
            raise ValueError("unet_lora is required")
        
        self._unet_lora_rank = self._unet_lora.get('r', self._unet_lora.get('rank'))
        if self._unet_lora_rank is None:
            raise ValueError("unet_lora_rank is required")
        
        self._unet_lora_alpha = self._unet_lora.get('lora_alpha', self._unet_lora_rank)        
        
        self._unet_lora_init_weights = self._unet_lora.get('init_lora_weights', "gaussian")
        
        self._unet_lora_target_modules = self._unet_lora.get('target_modules', ["to_k", "to_q", "to_v", "to_out.0"])
        
        self._unet_lora_use_dora = self._unet_lora.get('use_dora', False)

        if self._train_text_encoder:
            self._text_encoder_lora = self._train_config.get('text_encoder_lora', None)
            if self._text_encoder_lora is None:
                raise ValueError("text_encoder_lora is required")
            
            self._text_encoder_lora_rank = self._text_encoder_lora.get('r', self._text_encoder_lora.get('rank'))
            if self._text_encoder_lora_rank is None:
                raise ValueError("text_encoder_lora_rank is required")
            
            self._text_encoder_lora_alpha = self._text_encoder_lora.get('lora_alpha', self._text_encoder_lora_rank)        
            
            self._text_encoder_lora_init_weights = self._text_encoder_lora.get('init_lora_weights', "gaussian")
            
            self._text_encoder_lora_target_modules = self._text_encoder_lora.get('target_modules', ["q_proj", "k_proj", "v_proj", "out_proj"])
            
            self._text_encoder_lora_use_dora = self._text_encoder_lora.get('use_dora', False)

        self._batch_size = self._train_config.get('batch_size', 1)
        self._optimizer_name = self._train_config.get('optimizer', None)
        if self._optimizer_name is None:
            raise ValueError("optimizer is required")
        self._optimizer_name = self._optimizer_name.lower()
        if self._optimizer_name not in ['adamw', 'adamw8bit' 'adamwschedulefree', 'prodigy']:
            raise ValueError("Invalid optimizer. Supported optimizers are adamw and adamwschedulefree.")
        
        self._optimizer_params = self._train_config.get('optimizer_params', None)
        if self._optimizer_params is None:
            raise ValueError("optimizer_params is required")

        self._resolution = self._train_config.get('resolution', None)
        if not isinstance(self._resolution, int):
            raise ValueError("Invalid resolution. resolution must be an integer")

        self._dataset_folder = self._train_config.get('dataset_folder', None)
        if self._dataset_folder is None:
            raise ValueError("dataset_folder is required")
        
        self._num_workers = self._train_config.get('num_workers', 1)
        if self._num_workers > 1:
            print(f"Using {self._num_workers} workers for data loading")

        self._with_prior_preservation = self._train_config.get('with_prior_preservation', False)
        if self._with_prior_preservation:
            print("Using prior preservation")
        
        self._prior_loss_weight = self._train_config.get('prior_loss_weight', 1.0)
        if self._prior_loss_weight < 0.0:
            raise ValueError("prior_loss_weight must be >= 0.0")
        
        self._lr_scheduler = self._train_config.get('lr_scheduler', None)
        if self._lr_scheduler is None:
            raise ValueError("lr_scheduler is required")
        
        self._lr_scheduler_params = self._train_config.get('lr_scheduler_params', None)
        if self._lr_scheduler_params is None:
            raise ValueError("lr_scheduler_params is required")
        
        self._class_prompt = config.get('class_prompt', None)
        if self._class_prompt is None:
            raise ValueError("class_prompt is required")

        self._instant_prompt = config.get('instant_prompt', None)
        if self._instant_prompt is None:
            raise ValueError("instant_prompt is required")
        
        self._save_every = self._train_config.get('save_every', 0)

        self._max_grad_norm = self._train_config.get('max_grad_norm', None)
        if self._max_grad_norm is not None and self._max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be >= 0.0 or None")
        
        self._max_train_steps = self._train_config.get('max_train_steps', None)
        if self._max_train_steps is not None:
            if not isinstance(self._max_train_steps, int) or self._max_train_steps < 0:
                raise ValueError("max_train_steps must be a non-negative integer or None")
        
        self._lr_warmup_steps = self._train_config.get('lr_warmup_steps', 1)

        self._skip_pre_train_sample = self._train_config.get('skip_pre_train_sample', False)
        self._disable_sampling = self._train_config.get('disable_sampling', False)

        self._checkpoints_total_limit = self._train_config.get('checkpoints_total_limit', None)  
        self._sample_config = config.get('sample', None)
        if self._sample_config is not None:
            self._sample_every = self._sample_config.get('sample_every', 0)
            if self._sample_every < 0:
                raise ValueError("sample_every must be >= 0")
            self._sample_prompts = self._sample_config.get('prompts', None)
            if self._sample_prompts is None:
                raise ValueError("sample_prompts is required")
            self._sample_neg = self._sample_config.get('neg', None)
            if self._sample_neg is None:
                raise ValueError("sample_neg is required")
            if self._sample_neg is not None:
                if len(self._sample_neg) != len(self._sample_prompts):
                    raise ValueError("neg and prompts must have the same length")
            self._sample_seed = self._sample_config.get('seed', 42)
            self._sample_guidance_scale = self._sample_config.get('guidance_scale', 7)
            self._sample_steps = self._sample_config.get('steps', 20)
            self._walk_seed = self._sample_config.get('walk_seed', False)
            if self._walk_seed:
                print("Walking seed")
        
        self._resume_from_checkpoint = self._train_config.get('resume_from_checkpoint', None)
        if self._resume_from_checkpoint is not None:
            print(f"Resuming from checkpoint: {self._resume_from_checkpoint}")

        # EMA configuration
        self._use_ema = self._train_config.get('use_ema', False)
        if self._use_ema:
            self._ema_decay = self._train_config.get('ema_decay', 0.9999)
            print(f"Using EMA with decay {self._ema_decay}")

        self._loss_decay = self._train_config.get('loss_decay', 0.995)

    def _get_optimizer(self, params_to_optimize):
        if self._optimizer_name == "adamw":
            from torch.optim import AdamW            
            optimizer = AdamW(params_to_optimize,**self._optimizer_params)
        elif self._optimizer_name == "adamw8bit":
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("Please install bitsandbytes to use AdamW8bit optimizer")
            optimizer = bnb.optim.AdamW8bit(params_to_optimize, **self._optimizer_params)
        elif self._optimizer_name == "adamwschedulefree":
            try:
                from schedulefree import AdamWScheduleFree
            except ImportError:
                raise ImportError("Please install schedulefree to use AdamWScheduleFree optimizer")
            optimizer = AdamWScheduleFree(params_to_optimize, **self._optimizer_params)
        elif self._optimizer_name == "prodigy":
            try:
                from prodigy import ProdigyOptimizer
            except ImportError:
                raise ImportError("Please install prodigy to use Prodigy optimizer")
            lr = self._optimizer_params.get('lr', None)
            if lr is None:
                raise ValueError("lr is required for Prodigy optimizer")
            if lr < 0.1:
                raise ValueError("lr is better to be set around 1.0")
            optimizer = ProdigyOptimizer(params_to_optimize, lr=lr, **self._optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self._optimizer_name}")
        
        return optimizer

    def _get_text_encoder(self, subfolder: str):
        text_encoder_config = PretrainedConfig.from_pretrained(
            self._model_name_or_path, subfolder=subfolder, revision=self._revision, torch_dtype=self._dtype, local_files_only=self._local_files_only
        )
        model_class = text_encoder_config.architectures[0]

        cls = None
        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            cls = CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            cls = CLIPTextModelWithProjection
        elif model_class == "T5EncoderModel":
            from transformers import T5EncoderModel

            cls = T5EncoderModel
        else:
            raise ValueError(f"{model_class} is not supported.")   
        
        return cls.from_pretrained(
            self._model_name_or_path, subfolder=subfolder, revision=self._revision, torch_dtype=self._dtype
        )
                 
    def _get_tokenizer(self, subfolder: str):
        return AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            subfolder=subfolder,
            revision=self._revision,
            use_fast=False,
            torch_dtype=self._dtype,
            local_files_only=self._local_files_only
        )
    
    def _get_noise_scheduler(self):
        model_index_filename = "model_index.json"
        if Path(self._model_name_or_path).is_dir():
            model_index = Path(self._model_name_or_path) / model_index_filename
        else:
            model_index = hf_hub_download(
                repo_id=self._model_name_or_path, filename=model_index_filename, revision=self._revision
            )

        with open(model_index, "r") as f:
            scheduler_type = json.load(f)["scheduler"][1]        
        if "EDM" in scheduler_type:
            from diffusers import EDMEulerScheduler
            noise_scheduler = EDMEulerScheduler.from_pretrained(self._model_name_or_path, subfolder="scheduler", local_files_only=self._local_files_only)
            print("Using EDMEulerScheduler")
            self._do_edm_style_training = True
            is_edm_scheduler = True
        elif self._do_edm_style_training:
            from diffusers import EulerDiscreteScheduler
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(
                self._model_name_or_path, subfolder="scheduler", local_files_only=self._local_files_only
            )
            print("Using EulerDiscreteScheduler")
            is_edm_scheduler = False
        else:
            from diffusers import DDPMScheduler
            noise_scheduler = DDPMScheduler.from_pretrained(self._model_name_or_path, subfolder="scheduler", local_files_only=self._local_files_only)
            print("Using DDPMScheduler")
            is_edm_scheduler = False
        if self._do_edm_style_training:
            print("Performing EDM-style training")            
        
        return noise_scheduler, is_edm_scheduler
    
    def _get_vae(self):
        # The VAE is always in float32 to avoid NaN losses.
        return AutoencoderKL.from_pretrained(self._model_name_or_path, subfolder="vae", revision=self._revision, local_files_only=self._local_files_only)
    
    def _get_latents(self, vae):
        latents_mean = latents_std = None
        if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
            latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
            latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)
        return latents_mean, latents_std

    def _get_unet(self):
        return UNet2DConditionModel.from_pretrained(self._model_name_or_path, subfolder="unet", torch_dtype=self._dtype, local_files_only=self._local_files_only)
    
    def _cast_training_params_to_fp32(self, unet, text_encoder_1, text_encoder_2):
        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if self._dtype == torch.float16 or self._dtype == torch.bfloat16:
            models = [unet]
            if self._train_text_encoder:
                models.extend([text_encoder_1, text_encoder_2])                    
                cast_training_params(models, dtype=torch.float32)
        
    def _generate_samples(self, accelerator, global_step, vae, unet, text_encoder_one, text_encoder_two):
        if text_encoder_one is None:
            text_encoder_one = self._get_text_encoder("text_encoder")

        if text_encoder_two is None:
            text_encoder_two = self._get_text_encoder("text_encoder_2")
        sample_scheduler = DDIMScheduler.from_pretrained(self._model_name_or_path, subfolder="scheduler", local_files_only=self._local_files_only)
        with torch.no_grad():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self._model_name_or_path,
                vae = vae,
                text_encoder = accelerator.unwrap_model(text_encoder_one),
                text_encoder_2 = accelerator.unwrap_model(text_encoder_two),
                unet = accelerator.unwrap_model(unet),
                revision = self._revision,
                variant = self._variant,
                torch_dtype = self._dtype,
                use_safetensors = True,
                local_files_only = self._local_files_only
            )                                
            pipeline.set_progress_bar_config(disable=True)            
            pipeline = pipeline.to(accelerator.device)                 
            # Currently the context determination is a bit hand-wavy. We can improve it in the future if there's a better
            # way to condition it. Reference: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
            inference_ctx = (
                contextlib.nullcontext() #if "playground" in self.model_name_or_path else torch.cuda.amp.autocast(accelerator.device)
            )
            samples_dir = Path(self._experiment_folder) / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            with inference_ctx:
                if self._sample_neg:
                    sample_prompts = zip(self._sample_prompts, self._sample_neg)
                else:
                    sample_prompts = zip(self._sample_prompts, [None]*len(self._sample_prompts))
                for i, (p, n) in enumerate(sample_prompts):                    
                    generator = torch.Generator(device=accelerator.device).manual_seed(self._sample_seed + i if self._walk_seed else 0) if self._sample_seed else None
                    image = pipeline(
                        prompt = p,
                        negative_prompt = n,
                        height = self._resolution,
                        width = self._resolution,
                        guidance_scale = self._sample_guidance_scale,
                        num_inference_steps = self._sample_steps,
                        generator = generator
                    ).images[0]                            
                    image_path = samples_dir / f"{int(time.time())}_{global_step:010d}_{i}.jpg"
                    image.save(image_path)        
        del pipeline
        del sample_scheduler
        _flush()

    def run(self, callbacks: List[BaseCallback] = []):
        accelerator_project_config = ProjectConfiguration(project_dir=self._experiment_folder, logging_dir=self._logging_path)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            mixed_precision=self._mixed_precision,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )

        def unwrap_model(model):
            """
            Unwraps a model from its accelerator and compilation wrappers.
            
            This function does two things:
            1. It uses the accelerator to unwrap the model, removing any distributed
                or parallel processing wrappers.
            2. It checks if the model is a compiled module (e.g., using torch.compile),
                and if so, it returns the original module instead of the compiled version.

            Args:
                model: The model to unwrap.

            Returns:
                The unwrapped model, free from accelerator and compilation wrappers.
            """
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        if accelerator.is_local_main_process:

            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()

            Path(self._experiment_folder, parents=True, exist_ok=True)
            Path(self._logging_path, parents=True, exist_ok=True)

        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        
        if self._train_seed is not None:
            set_seed(self._train_seed)

        print ("Loading noise scheduler")
        noise_scheduler, is_edm_scheduler = self._get_noise_scheduler()

        print("Loading text encoder 1")
        tokenizer_1 = self._get_tokenizer("tokenizer")
        text_encoder_1 = self._get_text_encoder("text_encoder")
        text_encoder_1.requires_grad_(False)
        text_encoder_1.to(accelerator.device)

        print("Loading text encoder 2")
        tokenizer_2 = self._get_tokenizer("tokenizer_2")
        text_encoder_2 = self._get_text_encoder("text_encoder_2")
        text_encoder_2.requires_grad_(False)
        text_encoder_2.to(accelerator.device)

        print("Loading VAE")
        vae = self._get_vae()
        vae.requires_grad_(False)
        vae.to(accelerator.device, dtype=torch.float32)
        latents_mean, latents_std = self._get_latents(vae)        

        print("Loading unet")
        unet = self._get_unet()
        unet.requires_grad_(False)
        unet.to(accelerator.device)

        if self._gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if self._train_text_encoder:
                text_encoder_1.gradient_checkpointing_enable()
                text_encoder_2.gradient_checkpointing_enable()
        
        print("Adding LoRA adapters")
        unet_lora_config = LoraConfig(
            r=self._unet_lora_rank,
            use_dora=self._unet_lora_use_dora,
            lora_alpha=self._unet_lora_alpha,
            init_lora_weights=self._unet_lora_init_weights,
            target_modules=self._unet_lora_target_modules,
        )
        unet.add_adapter(unet_lora_config)
        
        if self._train_text_encoder:
            print("Adding LoRA adapters for text encoders")
            text_encoder_lora_config = LoraConfig(
                r=self._text_encoder_lora_rank,
                use_dora=self._text_encoder_lora_use_dora,
                lora_alpha=self._text_encoder_lora_alpha,
                init_lora_weights=self._text_encoder_lora_init_weights,
                target_modules=self._text_encoder_lora_target_modules,
            )
            text_encoder_1.add_adapter(text_encoder_lora_config)
            text_encoder_2.add_adapter(text_encoder_lora_config)

        def load_model_hook(models, input_dir):
            unet_to_load = None
            text_encoder_one_to_load = None
            text_encoder_two_to_load = None
            unet_type = type(unwrap_model(unet))
            text_encoder_1_type = type(unwrap_model(text_encoder_1))
            text_encoder_2_type = type(unwrap_model(text_encoder_2))

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, unet_type  ):
                    unet_to_load = model
                elif isinstance(model, text_encoder_1_type):
                    text_encoder_one_to_load = model
                elif isinstance(model, text_encoder_2_type):
                    text_encoder_two_to_load = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

            unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}            
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            incompatible_keys = set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    print(f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            if self._train_text_encoder:
                # Do we need to call `scale_lora_layers()` here?
                _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_to_load)

                _set_state_dict_into_text_encoder(
                    lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_to_load
                )

            self._cast_training_params_to_fp32(unet_to_load, text_encoder_one_to_load, text_encoder_two_to_load)

        unet_type = type(unwrap_model(unet))
        text_encoder_1_type = type(unwrap_model(text_encoder_1))
        text_encoder_2_type = type(unwrap_model(text_encoder_2))

        accelerator.register_save_state_pre_hook(partial(_save_model_hook, accelerator, unet_type, text_encoder_1_type, text_encoder_2_type))
        accelerator.register_load_state_pre_hook(load_model_hook)

        self._cast_training_params_to_fp32(unet, text_encoder_1, text_encoder_2)

        unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
        unet_lora_parameters_with_lr = {"params": unet_lora_parameters}
        if self._train_text_encoder:
            text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_1.parameters()))
            text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_2.parameters()))
            text_lora_parameters_one_with_lr = {
                "params": text_lora_parameters_one,
            }
            text_lora_parameters_two_with_lr = {
                "params": text_lora_parameters_two,
            }
            params_to_optimize = [
                unet_lora_parameters_with_lr,
                text_lora_parameters_one_with_lr,
                text_lora_parameters_two_with_lr,                
            ]
            params_to_clip = itertools.chain(unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)
        else:
            params_to_optimize = [unet_lora_parameters_with_lr]
            params_to_clip = unet_lora_parameters
        
        print("Loading optimizer")
        optimizer = self._get_optimizer(params_to_optimize)

        print("Loading datasets")
        train_dataset = DreamBoothDataset(self._dataset_folder, self._instant_prompt, self._class_prompt, resolution=self._resolution)
        train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self._num_workers, pin_memory=True)

        if not self._train_text_encoder:
            tokenizers = [tokenizer_1, tokenizer_2]
            text_encoders = [text_encoder_1, text_encoder_2]

            def compute_text_embeddings(prompt, text_encoders, tokenizers):
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = _encode_prompt(text_encoders, tokenizers, prompt)
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                return prompt_embeds, pooled_prompt_embeds

        # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
        # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
        # the redundant encoding.
        if not self._train_text_encoder:
            instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
                self._instant_prompt, text_encoders, tokenizers
            )

        # Handle class prompt for prior-preservation.
        if self._with_prior_preservation:
            if not self._train_text_encoder:
                class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
                    self._class_prompt, text_encoders, tokenizers
                )

        # Clear the memory here
        if not self._train_text_encoder and not train_dataset.custom_instance_prompts:
            del tokenizers, text_encoders
            _flush()

        # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
        # pack the statically computed variables appropriately here. This is so that we don't
        # have to pass them to the dataloader.

        if not train_dataset.custom_instance_prompts:
            if not self._train_text_encoder:
                prompt_embeds = instance_prompt_hidden_states
                unet_add_text_embeds = instance_pooled_prompt_embeds
                if self._with_prior_preservation:
                    prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                    unet_add_text_embeds = torch.cat([unet_add_text_embeds, class_pooled_prompt_embeds], dim=0)
            # if we're optmizing the text encoder (both if instance prompt is used for all images or custom prompts) we need to tokenize and encode the
            # batch prompts on all training steps
            else:
                tokens_one = _tokenize_prompt(tokenizer_1, self._instance_prompt)
                tokens_two = _tokenize_prompt(tokenizer_2, self._instance_prompt)
                if self._with_prior_preservation:
                    class_tokens_one = _tokenize_prompt(tokenizer_1, self._class_prompt)
                    class_tokens_two = _tokenize_prompt(tokenizer_2, self._class_prompt)
                    tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                    tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self._gradient_accumulation_steps)
        if self._max_train_steps is None:
            self._max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        print ('Loading lr scheduler')
        lr_scheduler = get_scheduler(
            self._lr_scheduler,
            optimizer=optimizer,
            **self._lr_scheduler_params            
        )

        # Prepare everything with our `accelerator`.
        if self._train_text_encoder:
            unet, text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self._gradient_accumulation_steps)
        if overrode_max_train_steps:
            self._max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(self._max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            tracker_name = (
                "dreambooth-lora-sd-xl"
                if "playground" not in self._model_name_or_path
                else "dreambooth-lora-playground"
            )
            accelerator.init_trackers(tracker_name, config=vars(self))

        print("Training ...")        

        global_step = 0
        first_epoch = 1
        ema_loss = None

        # Potentially load in the weights and states from a previous save
        if self._resume_from_checkpoint:
            if self._resume_from_checkpoint != "latest":
                path = os.path.basename(self._resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self._experiment_folder)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("_")[-1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{self._resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self._resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(self._experiment_folder, path))
                global_step = int(path.split("_")[-1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch

        else:
            initial_global_step = 0

        if not self._disable_sampling and not self._skip_pre_train_sample:
            self._generate_samples(accelerator, global_step, vae, unet, text_encoder_1, text_encoder_2)        
        
        progress_bar = tqdm(
            range(0, self._max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, self.num_train_epochs):
            self.set_to_train(accelerator, text_encoder_1, text_encoder_2, unet, optimizer)

            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    prompts = batch["prompts"]

                    # encode batch prompts when custom prompts are provided for each image -
                    if train_dataset.custom_instance_prompts:
                        if not self._train_text_encoder:
                            prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                                prompts, text_encoders, tokenizers
                            )
                        else:
                            tokens_one = _tokenize_prompt(tokenizer_1, prompts)
                            tokens_two = _tokenize_prompt(tokenizer_2, prompts)

                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()

                    if latents_mean is None and latents_std is None:
                        model_input = model_input * vae.config.scaling_factor
                        model_input = model_input.to(self._dtype)
                    else:
                        latents_mean = latents_mean.to(device=model_input.device, dtype=model_input.dtype)
                        latents_std = latents_std.to(device=model_input.device, dtype=model_input.dtype)
                        model_input = (model_input - latents_mean) * vae.config.scaling_factor / latents_std
                        model_input = model_input.to(dtype=self._dtype)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]

                    # Sample a random timestep for each image
                    if not self._do_edm_style_training:
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                        )
                        timesteps = timesteps.long()
                    else:
                        # in EDM formulation, the model is conditioned on the pre-conditioned noise levels
                        # instead of discrete timesteps, so here we sample indices to get the noise levels
                        # from `scheduler.timesteps`
                        indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                        timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                    # For EDM-style training, we first obtain the sigmas based on the continuous timesteps.
                    # We then precondition the final model inputs based on these sigmas instead of the timesteps.
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    if self._do_edm_style_training:
                        sigmas = _get_sigmas(accelerator, noise_scheduler, timesteps, len(noisy_model_input.shape), noisy_model_input.dtype)
                        if is_edm_scheduler:
                            inp_noisy_latents = noise_scheduler.precondition_inputs(noisy_model_input, sigmas)
                        else:
                            inp_noisy_latents = noisy_model_input / ((sigmas**2 + 1) ** 0.5)

                    # time ids
                    add_time_ids = torch.cat(
                        [
                            _compute_time_ids(original_size=s, crops_coords_top_left=c, resolution=self._resolution, device=accelerator.device, dtype=self._dtype)
                            for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                        ]
                    )

                    # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                    if not train_dataset.custom_instance_prompts:
                        elems_to_repeat_text_embeds = bsz // 2 if self._with_prior_preservation else bsz
                    else:
                        elems_to_repeat_text_embeds = 1

                    # Predict the noise residual
                    if not self._train_text_encoder:
                        unet_added_conditions = {
                            "time_ids": add_time_ids,
                            "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                        }
                        prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                        model_pred = unet(
                            inp_noisy_latents if self._do_edm_style_training else noisy_model_input,
                            timesteps,
                            prompt_embeds_input,
                            added_cond_kwargs=unet_added_conditions,
                            return_dict=False,
                        )[0]
                    else:
                        unet_added_conditions = {"time_ids": add_time_ids}
                        prompt_embeds, pooled_prompt_embeds = _encode_prompt(
                            text_encoders=[text_encoder_1, text_encoder_2],
                            tokenizers=None,
                            prompt=None,
                            text_input_ids_list=[tokens_one, tokens_two],
                        )
                        unet_added_conditions.update(
                            {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                        )
                        prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                        model_pred = unet(
                            inp_noisy_latents if self._do_edm_style_training else noisy_model_input,
                            timesteps,
                            prompt_embeds_input,
                            added_cond_kwargs=unet_added_conditions,
                            return_dict=False,
                        )[0]

                    weighting = None
                    if self._do_edm_style_training:
                        # Similar to the input preconditioning, the model predictions are also preconditioned
                        # on noised model inputs (before preconditioning) and the sigmas.
                        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                        if is_edm_scheduler:
                            model_pred = noise_scheduler.precondition_outputs(noisy_model_input, model_pred, sigmas)
                        else:
                            if noise_scheduler.config.prediction_type == "epsilon":
                                model_pred = model_pred * (-sigmas) + noisy_model_input
                            elif noise_scheduler.config.prediction_type == "v_prediction":
                                model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                                    noisy_model_input / (sigmas**2 + 1)
                                )
                        # We are not doing weighting here because it tends result in numerical problems.
                        # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
                        # There might be other alternatives for weighting as well:
                        # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
                        if not is_edm_scheduler:
                            weighting = (sigmas**-2.0).float()

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = model_input if self._do_edm_style_training else noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = (
                            model_input
                            if self._do_edm_style_training
                            else noise_scheduler.get_velocity(model_input, noise, timesteps)
                        )
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if self._with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute prior loss
                        if weighting is not None:
                            prior_loss = torch.mean(
                                (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                                    target_prior.shape[0], -1
                                ),
                                1,
                            )
                            prior_loss = prior_loss.mean()
                        else:
                            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    if self._snr_gamma is None:
                        if weighting is not None:
                            loss = torch.mean(
                                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                                    target.shape[0], -1
                                ),
                                1,
                            )
                            loss = loss.mean()
                        else:
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        base_weight = (
                            torch.stack([snr, self._snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )

                        if noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective needs to be floored to an SNR weight of one.
                            mse_loss_weights = base_weight + 1
                        else:
                            # Epsilon and sample both use the same loss weights.
                            mse_loss_weights = base_weight

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    if self._with_prior_preservation:
                        # Add the prior loss to the instance loss.
                        loss = loss + self._prior_loss_weight * prior_loss

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and self._max_grad_norm is not None:
                        accelerator.clip_grad_norm_(params_to_clip, self._max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    progress_callback = next((cb for cb in callbacks if isinstance(cb, ProgressCallback)), None)
                    if progress_callback:
                        progress_callback.set_progress(global_step / self._max_train_steps)

                if ema_loss is None:
                    ema_loss = loss.detach().item()
                else:
                    ema_loss = self._loss_decay * ema_loss + (1 - self._loss_decay) * loss.detach().item()  
                
                loss_callback = next((cb for cb in callbacks if isinstance(cb, LossCallback)), None)
                if loss_callback:
                    loss_callback.set_loss(ema_loss)

                logs = {"loss": ema_loss, "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process :
                    if self._save_every > 0 and global_step > 0 and global_step % self._save_every == 0:
                        if self._checkpoints_total_limit is not None:
                            checkpoints = os.listdir(self._experiment_folder)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= self._checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - self._checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                print(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(self._experiment_folder, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(self._experiment_folder, f"checkpoint_{self._name}_{self._version}_{global_step:010d}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

                    if not self._disable_sampling and self._sample_config is not None and global_step % self._sample_every == 0:                           
                        self._generate_samples(accelerator, global_step, vae, unet, text_encoder_1, text_encoder_2)

                if global_step >= self._max_train_steps:
                    print("Max training steps reached")
                    break

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = unwrap_model(unet)
            unet = unet.to(torch.float32)
            unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

            if self._train_text_encoder:
                text_encoder_1 = unwrap_model(text_encoder_1)
                text_encoder_lora_layers = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(text_encoder_1.to(torch.float32))
                )
                text_encoder_2 = unwrap_model(text_encoder_2)
                text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(text_encoder_2.to(torch.float32))
                )
            else:
                text_encoder_lora_layers = None
                text_encoder_2_lora_layers = None

            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=self._experiment_folder,
                unet_lora_layers=unet_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers,
            )

        accelerator.end_training()

    def set_to_train(self, accelerator, text_encoder_1, text_encoder_2, unet, optimizer):
        unet.train()
        optimizer.train()
        if self._train_text_encoder:
            text_encoder_1.train()
            text_encoder_2.train()

                # set top parameter requires_grad = True for gradient checkpointing works
            accelerator.unwrap_model(text_encoder_1).text_model.embeddings.requires_grad_(True)
            accelerator.unwrap_model(text_encoder_2).text_model.embeddings.requires_grad_(True)
