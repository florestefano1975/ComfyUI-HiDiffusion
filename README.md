## ComfyUI-HiDiffusion

Custom node for the use of HiDiffusion technology.

![ComfyUI-HiDiffusion nodes](/assets/overview.png)

**Please refer to the original project for details and specifications.**

## !!WARNINGS!!

Several bugs and malfunctions of ComfyUI were found after the nodes were installed. I recommend testing them beforehand in a separate installation to avoid problems and program crashes.

The requirements.txt file was removed to avoid installation of dependencies that create conflicts. In case the nodes do not start correctly when launching ComfyUI, refer to the console errors to manually install the missing dependencies.

## SDXL Node Updates

The node dedicated to SDXL models has more options:

- **ckpt_name**
- **apply_raunet**
- **apply_window_attn**
- **optimizations**
- **steps**
- **guidance_scale**
- **scheduler**

I will run tests to be able to implement the same number of options in the other nodes as well.

![ComfyUI-HiDiffusion SDXL node](/assets/sdxl-2.png)

![ComfyUI-HiDiffusion SDXL node](/assets/sdxl-3.png)

## Docs

- HiDiffusion project page: https://hidiffusion.github.io/
- HiDiffusion repository: https://github.com/megvii-research/HiDiffusion/
- Diffusers: https://huggingface.co/docs/diffusers/v0.27.2/en/api/schedulers/overview

## Other projects

- [ComfyUI Portrait Master](https://github.com/florestefano1975/comfyui-portrait-master/)
- [ComfyUI Prompt Composer](https://github.com/florestefano1975/comfyui-prompt-composer/)
- [ComfyUI StabilityAI Suite](https://github.com/florestefano1975/ComfyUI-StabilityAI-Suite/)

**_If this project is useful to you and you like it, please consider a small donation to the author._**

➡️ https://ko-fi.com/stefanoflore75