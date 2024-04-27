# ComfyUI-HiDiffusion
# Created by AI Wiz Art (Stefano Flore)
# Version: 1.0
# https://stefanoflore.it
# https://ai-wiz.art

from hidiffusion import apply_hidiffusion, remove_hidiffusion
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoPipelineForText2Image, DiffusionPipeline
import torch
import numpy as np

# ========================================================
# IMAGE TO TENSOR
# ========================================================

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

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
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Standing tall amidst the ruins, a stone golem awakens, vines and flowers sprouting from the crevices in its body."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, ugly, duplicate, poorly drawn face, deformed, mosaic, artifacts, bad limbs"
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
                    "default": 2048,
                    "min": 0,
                    "max": 9999999
                }),
                "height": ("INT", {
                    "default": 2048,
                    "min": 0,
                    "max": 9999999
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "hi_diff_sdxl"
    CATEGORY = "AI WizArt/HiDiffusion"

    def hi_diff_sdxl(self, positive_prompt="", negative_prompt="", guidance_scale=7.5, width=2048, height=2048, eta=1.0):
        pretrain_model = "stabilityai/stable-diffusion-xl-base-1.0"
        scheduler = DDIMScheduler.from_pretrained(pretrain_model, subfolder="scheduler")
        pipe = StableDiffusionXLPipeline.from_pretrained(pretrain_model, scheduler = scheduler, torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        apply_hidiffusion(pipe)
        image = pipe(prompt=positive_prompt, guidance_scale=guidance_scale, height=height, width=width, eta=eta, negative_prompt=negative_prompt).images[0]
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

    def hi_diff_sdxlturbo(self, positive_prompt="", guidance_scale=7.5, width=1024, height=1024, eta=1.0, inference_steps=4):
        pretrain_model = "stabilityai/sdxl-turbo"
        pipe = AutoPipelineForText2Image.from_pretrained(pretrain_model, torch_dtype=torch.float16, variant="fp16").to('cuda')
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        apply_hidiffusion(pipe)
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

    def hi_diff_sd21(self, positive_prompt="", negative_prompt="", guidance_scale=7.5, width=1024, height=1024, eta=1.0):
        pretrain_model = "stabilityai/stable-diffusion-2-1-base"
        scheduler = DDIMScheduler.from_pretrained(pretrain_model, subfolder="scheduler")
        pipe = DiffusionPipeline.from_pretrained(pretrain_model, scheduler = scheduler, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        apply_hidiffusion(pipe)
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

    def hi_diff_sd15(self, positive_prompt="", negative_prompt="", guidance_scale=7.5, width=1024, height=1024, eta=1.0):
        pretrain_model = "runwayml/stable-diffusion-v1-5"
        scheduler = DDIMScheduler.from_pretrained(pretrain_model, subfolder="scheduler")
        pipe = DiffusionPipeline.from_pretrained(pretrain_model, scheduler = scheduler, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        apply_hidiffusion(pipe)
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
    "HiDiffusionSD15": HiDiffusionSD15
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDiffusionSDXL": "HiDiffusion SDXL",
    "HiDiffusionSDXLTurbo": "HiDiffusion SDXL Turbo",
    "HiDiffusionSD21": "HiDiffusion SD 2.1",
    "HiDiffusionSD15": "HiDiffusion SD 1.5"
}
