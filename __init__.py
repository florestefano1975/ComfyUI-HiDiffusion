# ComfyUI-HiDiffusion
# Created by AI Wiz Art (Stefano Flore)
# Version: 1.3
# https://stefanoflore.it
# https://ai-wiz.art

from .hidiffusion import apply_hidiffusion, remove_hidiffusion
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, AutoPipelineForText2Image, DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import DPMSolverSinglestepScheduler
from diffusers import KDPM2DiscreteScheduler
from diffusers import KDPM2AncestralDiscreteScheduler
from diffusers import EulerDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import HeunDiscreteScheduler
from diffusers import LMSDiscreteScheduler
from diffusers import UniPCMultistepScheduler
from diffusers import DDIMScheduler
import torch
import numpy as np
import comfy.sd
import folder_paths as comfy_paths
import os

# ========================================================
# IMAGE TO TENSOR
# ========================================================

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# ========================================================
# SCHEDULERS
# ========================================================

scheduler_list = [
    "DDIM",
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Karras",
    "DPM++ SDE",
    "DPM++ SDE Karras",
    "DPM2",
    "DPM2 Karras",
    "DPM2 a",
    "DPM2 a Karras",
    "Euler",
    "Euler a",
    "Heun",
    "LMS",
    "LMS Karras",
    "UniPC",
]

def get_sheduler(name):
    scheduler = False
    if name == "DPM++ 2M":
        scheduler = DPMSolverMultistepScheduler()
    elif name == "DPM++ 2M Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
    elif name == "DPM++ 2M SDE":
        scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ 2M SDE Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ SDE":
        scheduler = DPMSolverSinglestepScheduler()
    elif name == "DPM++ SDE Karras":
        scheduler = DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif name == "DPM2":
        scheduler = KDPM2DiscreteScheduler()
    elif name == "DPM2 Karras":
        scheduler = KDPM2DiscreteScheduler(use_karras_sigmas=True)
    elif name == "DPM2 a":
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif name == "DPM2 a Karras":
        scheduler = KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True)
    elif name == "Euler":
        scheduler = EulerDiscreteScheduler()
    elif name == "Euler a":
        scheduler = EulerAncestralDiscreteScheduler()
    elif name == "Heun":
        scheduler = HeunDiscreteScheduler()
    elif name == "LMS":
        scheduler = LMSDiscreteScheduler()
    elif name == "LMS Karras":
        scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
    elif name == "UniPC":
        scheduler = UniPCMultistepScheduler()
    return scheduler

# ========================================================
# SDXL
# ========================================================

class HiDiffusionSDXL:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (comfy_paths.get_filename_list("checkpoints"),),
                "apply_raunet": ("BOOLEAN", {"default": True},),
                "apply_window_attn": ("BOOLEAN", {"default": True},),
                "optimizations": ("BOOLEAN", {"default": True},),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Standing tall amidst the ruins, a stone golem awakens, vines and flowers sprouting from the crevices in its body."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, ugly, duplicate, poorly drawn face, deformed, mosaic, artifacts, bad limbs"
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1,
                    "max": 99,
                    "step": 0.1
                }),
                "eta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01
                }),
                "scheduler": (scheduler_list,),
                "width": ("INT", {
                    "default": 2048,
                    "min": 0,
                    "max": 9999999
                }),
                "height": ("INT", {
                    "default": 2048,
                    "min": 0,
                    "max": 9999999
                }),
                "seed": ("INT", {"forceInput": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "hi_diff_sdxl"
    CATEGORY = "AI WizArt/HiDiffusion"

    def hi_diff_sdxl(self, positive_prompt="", negative_prompt="", guidance_scale=7.5, width=2048, height=2048, eta=1.0, seed=False, ckpt_name="", apply_raunet=True, apply_window_attn=True, optimizations=True, scheduler="", steps=50):
        ckpt_path = comfy_paths.get_full_path("checkpoints", ckpt_name)
        scheduler_apply = get_sheduler(scheduler)
        if scheduler_apply == False:
            pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, torch_dtype=torch.float16).to("cuda")
        else:
            pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, scheduler=scheduler_apply, torch_dtype=torch.float16).to("cuda")
        if optimizations:
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
        apply_hidiffusion(pipe, apply_raunet, apply_window_attn, "SDXL")
        image = pipe(prompt=positive_prompt, num_inference_steps=steps, guidance_scale=guidance_scale, height=height, width=width, eta=eta, negative_prompt=negative_prompt).images[0]
        output_t = pil2tensor(image)
        return (output_t,)

# ========================================================
# XL Turbo
# ========================================================

class HiDiffusionSDXLTurbo:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "seed": ("INT", {"forceInput": False}),
            },
            "required": {
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "In the depths of a mystical forest, a robotic owl with night vision lenses for eyes watches over the nocturnal creatures."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0,
                    "max": 99,
                    "step": 0.1
                }),
                "inference_steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 99
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 9999999
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 9999999
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "hi_diff_sdxlturbo"
    CATEGORY = "AI WizArt/HiDiffusion"

    def hi_diff_sdxlturbo(self, positive_prompt="", guidance_scale=7.5, width=1024, height=1024, eta=1.0, inference_steps=4, seed=False):
        pretrain_model = "stabilityai/sdxl-turbo"
        pipe = AutoPipelineForText2Image.from_pretrained(pretrain_model, torch_dtype=torch.float16, variant="fp16").to('cuda')
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        apply_hidiffusion(pipe, True, True, "SDXLTurbo")
        image = pipe(prompt=positive_prompt, num_inference_steps=inference_steps, height=height, width=width, guidance_scale=guidance_scale).images[0]
        output_t = pil2tensor(image)
        return (output_t,)

# ========================================================
# SD 2.1
# ========================================================

class HiDiffusionSD21:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "seed": ("INT", {"forceInput": False}),
            },
            "required": {
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "An adorable happy brown border collie sitting on a bed, high detail."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "ugly, tiling, out of frame, poorly drawn face, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, artifacts, bad proportions."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1,
                    "max": 99,
                    "step": 0.1
                }),
                "eta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 9999999
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 9999999
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "hi_diff_sd21"
    CATEGORY = "AI WizArt/HiDiffusion"

    def hi_diff_sd21(self, positive_prompt="", negative_prompt="", guidance_scale=7.5, width=1024, height=1024, eta=1.0, seed=False):
        pretrain_model = "stabilityai/stable-diffusion-2-1-base"
        scheduler = DDIMScheduler.from_pretrained(pretrain_model, subfolder="scheduler")
        pipe = DiffusionPipeline.from_pretrained(pretrain_model, scheduler = scheduler, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        apply_hidiffusion(pipe, True, True, "SD2.1")
        image = pipe(prompt=positive_prompt, guidance_scale=guidance_scale, height=height, width=width, eta=eta, negative_prompt=negative_prompt).images[0]
        output_t = pil2tensor(image)
        return (output_t,)

# ========================================================
# SD 1.5
# ========================================================

class HiDiffusionSD15:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "seed": ("INT", {"forceInput": False}),
            },
            "required": {
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "thick strokes, bright colors, an exotic fox, cute, chibi kawaii. detailed fur, hyperdetailed , big reflective eyes, fairytale, artstation,centered composition, perfect composition, centered, vibrant colors, muted colors, high detailed, 8k."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "ugly, tiling, poorly drawn face, out of frame, disfigured, deformed, blurry, bad anatomy, blurred."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1,
                    "max": 99,
                    "step": 0.1
                }),
                "eta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 9999999
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 9999999
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "hi_diff_sd15"
    CATEGORY = "AI WizArt/HiDiffusion"

    def hi_diff_sd15(self, positive_prompt="", negative_prompt="", guidance_scale=7.5, width=1024, height=1024, eta=1.0, seed=False):
        pretrain_model = "runwayml/stable-diffusion-v1-5"
        scheduler = DDIMScheduler.from_pretrained(pretrain_model, subfolder="scheduler")
        pipe = DiffusionPipeline.from_pretrained(pretrain_model, scheduler = scheduler, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        apply_hidiffusion(pipe, True, True, "SD1.5")
        image = pipe(prompt=positive_prompt, guidance_scale=guidance_scale, height=height, width=width, eta=eta, negative_prompt=negative_prompt).images[0]
        output_t = pil2tensor(image)
        return (output_t,)

# ========================================================
# NODE MAPPING
# ========================================================

NODE_CLASS_MAPPINGS = {
    "HiDiffusionSDXL": HiDiffusionSDXL,
    "HiDiffusionSDXLTurbo": HiDiffusionSDXLTurbo,
    "HiDiffusionSD21": HiDiffusionSD21,
    "HiDiffusionSD15": HiDiffusionSD15,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDiffusionSDXL": "HiDiffusion SDXL",
    "HiDiffusionSDXLTurbo": "HiDiffusion SDXL Turbo",
    "HiDiffusionSD21": "HiDiffusion SD 2.1",
    "HiDiffusionSD15": "HiDiffusion SD 1.5",
}
