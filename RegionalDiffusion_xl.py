import inspect
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import xformers
import xformers.ops

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import DPMSolverMultistepScheduler, KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from matrix import matrixdealer, keyconverter  # 假设 matrix.py 在同级目录

# 定义常量
TOKENSCON = 77
TOKENS = 75

if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from RegionalDiffusion_xl import RegionalDiffusionXLPipeline

        >>> pipe = RegionalDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

# ==============================================================================
# 1. 辅助函数 (Attention Helpers)
# ==============================================================================

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def _memory_efficient_attention_xformers(module, query, key, value):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
    hidden_states = module.batch_to_head_dim(hidden_states)
    return hidden_states

def main_forward_diffusers(module, hidden_states, encoder_hidden_states, divide, userpp=False, tokens=[], width=64, height=64, step=0, isxl=False, inhr=None):
    context = encoder_hidden_states
    query = module.to_q(hidden_states)
    key = module.to_k(context)
    value = module.to_v(context)
    
    query = module.head_to_batch_dim(query)
    key = module.head_to_batch_dim(key)
    value = module.head_to_batch_dim(value)
    
    hidden_states = _memory_efficient_attention_xformers(module, query, key, value)
    hidden_states = hidden_states.to(query.dtype)
    
    hidden_states = module.to_out[0](hidden_states)
    hidden_states = module.to_out[1](hidden_states)
    return hidden_states

def split_dims(x_t, height, width, self=None):
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    if x_t > latent_h * latent_w and hasattr(self, "nei_multi"):
        latent_h, latent_w = self.nei_multi[1], self.nei_multi[0]
        while latent_h * latent_w != x_t:
            latent_h, latent_w = latent_h // 2, latent_w // 2
    return latent_h, latent_w

def repeat_div(x, y):
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x

# ==============================================================================
# 2. 核心 Hook 逻辑 (含区域 IP-Adapter 支持)
# ==============================================================================

def hook_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "Attention":
            module.forward = hook_forward(self, module)

def hook_forward(self, module):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x = hidden_states
        context = encoder_hidden_states
        
        height = self.h
        width = self.w
        x_t = x.size()[1]
        
        # 简单计算 latent 尺寸
        scale = round(math.sqrt(height * width / x_t))
        latent_h = round(height / scale)
        latent_w = round(width / scale)
        
        # 修正计算误差
        ha, wa = x_t % latent_h, x_t % latent_w
        if ha == 0: latent_w = int(x_t / latent_h)
        elif wa == 0: latent_h = int(x_t / latent_w)

        contexts = context.clone()

        # 获取区域 IP-Adapter Embeddings (如果存在)
        # 格式: List[Tensor], 每个Tensor形状 [Batch, Seq, Dim]
        region_ip_embeds_list = None
        if hasattr(self, "_cross_attention_kwargs") and self._cross_attention_kwargs is not None:
            region_ip_embeds_list = self._cross_attention_kwargs.get("region_ip_embeds", None)
            
        def matsepcalc(x, contexts, pn, divide):
            h_states = []
            (latent_h, latent_w) = split_dims(x.size()[1], height, width, self)
            latent_out = latent_w
            latent_in = latent_h

            tll = self.pt # 区域 token 范围
            i = 0
            outb = None

            # --- Base (Background) Region ---
            if self.usebase:
                # 1. 截取 Base Prompt 的文本 Context
                context_base = contexts[:, tll[i][0] * TOKENSCON : tll[i][1] * TOKENSCON, :]
                
                # ControlNet 兼容
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context_base = torch.cat([context_base, contexts[:, -cnet_ext:, :]], dim=1)
                
                # 2. [创新点] 注入 Base 区域的 IP-Adapter (Index 0)
                if region_ip_embeds_list is not None and len(region_ip_embeds_list) > i:
                    # 拼接: [Batch, Text_Seq + Image_Seq, Dim]
                    # 注意：IP-Adapter 通常是通过 CrossAttn 下发。如果 module.to_k 维度匹配即可。
                    # SDXL IP-Adapter 输出 dim 2048 (OpenCLIP) 或 1280 (ViT-BigG)，需确保 project 过。
                    # 我们假设传入的 embeds 已经是 project 好的 (在 pipeline __call__ 里处理)。
                    ip_emb_base = region_ip_embeds_list[i].to(context_base.device, context_base.dtype)
                    # 扩展 batch 维度以匹配 context (CFG split 后)
                    if ip_emb_base.shape[0] != context_base.shape[0]:
                         ip_emb_base = ip_emb_base.repeat(context_base.shape[0] // ip_emb_base.shape[0], 1, 1)
                    
                    context_base = torch.cat([context_base, ip_emb_base], dim=1)

                i = i + 1
                
                # Base 前向
                out = main_forward_diffusers(module, x, context_base, divide, userpp=True, isxl=self.isxl)
                outb = out.clone().reshape(out.size()[0], latent_h, latent_w, out.size()[2])

            # --- Split Regions ---
            sumout = 0
            for drow in self.split_ratio:
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    # 1. 截取当前区域的 Text Context
                    context_region = contexts[:, tll[i][0] * TOKENSCON : tll[i][1] * TOKENSCON, :]
                    
                    cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                    if cnet_ext > 0:
                        context_region = torch.cat([context_region, contexts[:, -cnet_ext:, :]], dim=1)

                    # 2. [创新点] 注入当前 Split 区域的 IP-Adapter
                    # i 已经递增，对应当前 region index
                    if region_ip_embeds_list is not None:
                        # 安全获取 index，防止越界
                        ip_idx = i if i < len(region_ip_embeds_list) else -1
                        ip_emb_region = region_ip_embeds_list[ip_idx].to(context_region.device, context_region.dtype)
                        
                        if ip_emb_region.shape[0] != context_region.shape[0]:
                             ip_emb_region = ip_emb_region.repeat(context_region.shape[0] // ip_emb_region.shape[0], 1, 1)
                        
                        context_region = torch.cat([context_region, ip_emb_region], dim=1)

                    # 递增索引 (处理 breaks)
                    i = i + 1 + dcell.breaks

                    # Region 前向
                    out = main_forward_diffusers(module, x, context_region, divide, userpp=self.pn, isxl=self.isxl)
                    out = out.reshape(out.size()[0], latent_h, latent_w, out.size()[2])

                    # 3. 空间掩码与融合 (Spatial Masking & Blending)
                    addout = 0
                    addin = 0
                    sumin = sumin + int(latent_in * dcell.end) - int(latent_in * dcell.start)
                    
                    # 修正边界
                    if dcell.end >= 0.999:
                        addin = sumin - latent_in
                        sumout = sumout + int(latent_out * drow.end) - int(latent_out * drow.start)
                        if drow.end >= 0.999:
                            addout = sumout - latent_out
                    
                    # Crop out the region
                    out = out[:, int(latent_h * drow.start) + addout : int(latent_h * drow.end),
                              int(latent_w * dcell.start) + addin : int(latent_w * dcell.end), :]
                    
                    # 与 Base 混合 (Base Ratio)
                    if self.usebase and outb is not None:
                        outb_t = outb[:, int(latent_h * drow.start) + addout : int(latent_h * drow.end),
                                      int(latent_w * dcell.start) + addin : int(latent_w * dcell.end), :].clone()
                        out = out * (1 - dcell.base) + outb_t * dcell.base
            
                    v_states.append(out)

                output_x = torch.cat(v_states, dim=2)
                h_states.append(output_x)
            
            output_x = torch.cat(h_states, dim=1)
            output_x = output_x.reshape(x.size()[0], x.size()[1], x.size()[2])
            return output_x

        # Batch Splitting for CFG (Cond / Uncond)
        if x.size()[0] == 1 * self.batch_size:
            output_x = matsepcalc(x, contexts, self.pn, 1)
        else:
            if self.isvanilla:
                nx, px = x.chunk(2)
                conn, conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp, conn = contexts.chunk(2)
            
            # 分别计算正负向
            opx = matsepcalc(px, conp, True, 2)
            onx = matsepcalc(nx, conn, False, 2)
            
            if self.isvanilla:
                output_x = torch.cat([onx, opx])
            else:
                output_x = torch.cat([opx, onx])

        self.pn = not self.pn
        self.count = 0
        return output_x

    return forward

# ==============================================================================
# 3. Pipeline 定义
# ==============================================================================

class RegionalDiffusionXLPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", 
        "image_encoder", "feature_extractor"
    ]
    _callback_tensor_inputs = [
        "latents", "prompt_embeds", "negative_prompt_embeds", 
        "add_text_embeds", "add_time_ids", 
        "negative_pooled_prompt_embeds", "negative_add_time_ids"
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DPMSolverMultistepScheduler,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        
        # 核心 Hook 注入
        hook_forwards(self, self.unet)
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    # 新增: 辅助 IP-Adapter 编码
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)
            return image_embeds, uncond_image_embeds

    def encode_prompt(self, prompt, prompt_2=None, device=None, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None, negative_prompt_2=None, prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None, lora_scale=None, clip_skip=None):
        # 简化版实现，保留原逻辑
        device = device or self._execution_device
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder: scale_lora_layers(self.text_encoder, lora_scale) if USE_PEFT_BACKEND else adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            if self.text_encoder_2: scale_lora_layers(self.text_encoder_2, lora_scale) if USE_PEFT_BACKEND else adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]

            for prompt_item, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt_item = self.maybe_convert_prompt(prompt_item, tokenizer)
                
                # Split prompt by 'BREAK' (Regional Logic)
                regional_prompt_list = prompt_item[0].split('BREAK')
                regional_prompt_embeds = []
                
                for sub_prompt in regional_prompt_list:
                    text_inputs = tokenizer(sub_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                    text_input_ids = text_inputs.input_ids
                    prompt_embeds_curr = text_encoder(text_input_ids.to(device), output_hidden_states=True)
                    pooled_prompt_embeds = prompt_embeds_curr[0]
                    if clip_skip is None:
                        prompt_embeds_curr = prompt_embeds_curr.hidden_states[-2]
                    else:
                        prompt_embeds_curr = prompt_embeds_curr.hidden_states[-(clip_skip + 2)]
                    regional_prompt_embeds.append(prompt_embeds_curr)
                
                prompt_embeds = torch.cat(regional_prompt_embeds, dim=1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            # 计算区域数量供 Hook 使用
            self.region_num = prompt_embeds.shape[1] // TOKENSCON
            # print(f"Detected {self.region_num} regions from prompt.")

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            
            uncond_tokens = [negative_prompt, negative_prompt_2]
            negative_prompt_embeds_list = []
            
            for negative_prompt_item, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                # Negative prompt also needs to be repeated for regions to match shape
                regional_neg_list = []
                # Use calculated region_num
                r_num = getattr(self, 'region_num', 1) 
                
                for i in range(r_num):
                    uncond_input = tokenizer(negative_prompt_item, padding="max_length", max_length=TOKENSCON, truncation=True, return_tensors="pt")
                    neg_emb = text_encoder(uncond_input.input_ids.to(device), output_hidden_states=True)
                    negative_pooled_prompt_embeds = neg_emb[0]
                    neg_emb = neg_emb.hidden_states[-2]
                    regional_neg_list.append(neg_emb)
                
                negative_prompt_embeds = torch.concat(regional_neg_list, dim=1)
                negative_prompt_embeds_list.append(negative_prompt_embeds)
            
            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        # BS expansion
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)
            
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = (self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(f"Model expects {expected_add_embed_dim} add_embed_dim, got {passed_add_embed_dim}")
        return torch.tensor([add_time_ids], dtype=dtype)

    def get_guidance_scale_embedding(self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32) -> torch.FloatTensor:
        assert len(w.shape) == 1
        w = w * 1000.0
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1: emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def regional_info(self, prompts):
        ppl = prompts.split('BREAK')
        targets = [p.split(",")[-1] for p in ppl[:]]
        pt, ppt = [], []
        padd = 0
        for pp in targets:
            pp = pp.split(" ")
            pp = [p for p in pp if p != ""]
            tokensnum = len(pp)
            pt.append([padd, tokensnum // TOKENS + 1 + padd])
            ppt.append(tokensnum)
            padd = tokensnum // TOKENS + 1 + padd
        self.pt = pt
        self.ppt = ppt

    def torch_fix_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

    # ==============================================================================
    # 4. Main Inference (__call__)
    # ==============================================================================
    @property
    def guidance_scale(self): return self._guidance_scale
    @property
    def do_classifier_free_guidance(self): return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    @property
    def cross_attention_kwargs(self): return self._cross_attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        split_ratio: str,
        base_ratio: Optional[float] = None,
        base_prompt: Optional[str] = None,
        batch_size: Optional[int] = 1,
        prompt: Union[str, List[str]] = None,
        seed: Optional[int] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        
        # --- NEW: SDEdit / IP-Adapter Args ---
        image: Optional[torch.Tensor] = None,           # SDEdit 输入图
        strength: float = 0.8,                          # SDEdit 强度
        region_ip_images: Optional[List[Any]] = None,   # 区域 IP 图片列表
        region_ip_weights: Optional[List[float]] = None,# 区域 IP 权重
        enable_region_ip: bool = False,                 # 开关
        # -------------------------------------

        **kwargs,
    ):
        # 1. Init & Config
        self.usebase = True if base_ratio is not None and base_prompt is not None else False
        self.base_ratio = base_ratio
        self.base_prompt = base_prompt
        self.split_ratio = split_ratio
        
        # Prompt Composition
        self.prompt = prompt if base_prompt is None else base_prompt + ' BREAK ' + prompt
        self.original_prompt = self.prompt
        
        self.h = height or self.default_sample_size * self.vae_scale_factor
        self.w = width or self.default_sample_size * self.vae_scale_factor
        
        # Hooks Flags
        self.pn = True
        self.eq = True
        self.isvanilla = True
        self.count = 0
        self.isxl = True
        self.batch_size = batch_size
        
        # Initialize Regional Info (pt, ppt)
        self.regional_info(self.prompt)
        
        # Matrix Utils (Ensure matrix.py is present)
        keyconverter(self, self.split_ratio, self.usebase)
        matrixdealer(self, self.split_ratio, self.base_ratio)
        
        if seed is not None and seed > 0:
            self.torch_fix_seed(seed=seed)

        # 2. Dimensions & Checks
        original_size = original_size or (self.h, self.w)
        target_size = target_size or (self.h, self.w)

        self.check_inputs(
            prompt, prompt_2, self.h, self.w, None, negative_prompt, negative_prompt_2,
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
            ip_adapter_image, ip_adapter_image_embeds, callback_on_step_end_tensor_inputs
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs or {}
        self._denoising_end = denoising_end
        self._interrupt = False
        
        device = self._execution_device

        # 3. Encode Prompts
        lora_scale = self._cross_attention_kwargs.get("scale", None)
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=self.original_prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare Timesteps (SDEdit aware)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        
        if image is not None:
            # Img2Img Mode
            t_start = int(len(timesteps) * strength)
            t_start = min(t_start, len(timesteps) - 1)
            timesteps = timesteps[len(timesteps) - t_start :]
            num_inference_steps = len(timesteps)

        # 5. IP-Adapter Pre-processing for Regions
        if enable_region_ip and region_ip_images is not None:
            if not getattr(self, "image_encoder", None):
                logger.warning("enable_region_ip is True but 'image_encoder' is not found. Skipping.")
            else:
                # 获取区域数量 (从 prompt_embeds 推断)
                region_num = getattr(self, "region_num", len(region_ip_images))
                
                # 编码图像
                ip_embeds_list = []
                for i in range(region_num):
                    # 循环使用图片，防止越界
                    img = region_ip_images[i] if i < len(region_ip_images) else region_ip_images[-1]
                    # 获取 hidden_states [-2] 层作为 Token 输入
                    ip_emb, _ = self.encode_image(
                        img, device, num_images_per_prompt, output_hidden_states=True
                    )
                    ip_embeds_list.append(ip_emb)
                
                # 注入 kwargs 供 Hook 使用
                self._cross_attention_kwargs["region_ip_embeds"] = ip_embeds_list
                # region_ip_weights 如果需要可以在 Hook 里加权逻辑，目前 Hook 实现为 Concat
        
        # 6. Prepare Latents (SDEdit or Txt2Img)
        num_channels_latents = self.unet.config.in_channels
        
        if image is not None:
            # SDEdit Mode
            if not isinstance(image, torch.Tensor):
                image = self.image_processor.preprocess(image)
            
            image = image.to(device=device, dtype=self.vae.dtype)
            init_latents = self.vae.encode(image).latent_dist.sample(generator)
            init_latents = init_latents * self.vae.config.scaling_factor
            
            if init_latents.shape[0] != batch_size * num_images_per_prompt:
                init_latents = init_latents.repeat(batch_size * num_images_per_prompt, 1, 1, 1)

            noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
            
            # SDEdit add_noise
            # DPMSolver usually takes scalar or matching batch size
            latent_timestep = timesteps[:1].repeat(init_latents.shape[0])
            latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = latents.to(dtype=prompt_embeds.dtype)
        else:
            # Standard Mode
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                self.h,
                self.w,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # 7. Prepare Added Time IDs (SDXL)
        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim if self.text_encoder_2 else int(pooled_prompt_embeds.shape[-1])

        add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, prompt_embeds.dtype, text_encoder_projection_dim)
        
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(negative_original_size, negative_crops_coords_top_left, negative_target_size, prompt_embeds.dtype, text_encoder_projection_dim)
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # Global IP-Adapter (Legacy support for full image IP-Adapter)
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        else:
            image_embeds = None

        # 8. Denoising Loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        
        # Guidance Scale Embedding (SDXL Requirement)
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt: continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                
                # Forward Pass (Hook will handle regional logic)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # Step
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    # Callback logic omitted for brevity, keeping original structure
                    pass

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE: xm.mark_step()

        # 9. Decode
        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            if needs_upcasting: self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            if self.watermark is not None: image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict: return (image,)
        return StableDiffusionXLPipelineOutput(images=image)
