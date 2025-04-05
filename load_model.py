from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
import torch


def load_model(model_name='sd14', device='cuda'):
    """Загрузка модели Stable Diffusion"""
    if model_name == 'sd14':
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
             torch_dtype=torch.float16, 
            cache_dir='./cache'
            )
    elif model_name == 'sd21':
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16, 
            cache_dir='./cache'
        )
    elif model_name == 'sd21-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir='./cache'
        )
    elif model_name == 'sdxl':
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16",
            cache_dir='./cache'
        )
    elif model_name == 'sdxl-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
             "stabilityai/sdxl-turbo", 
             torch_dtype=torch.float16, 
             variant="fp16",
             cache_dir='./cache'
         )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe.to(device)
    return pipe, device


def run_model(model_name, pipe, prompt, seed, num_denoising_steps, device='cuda'):
    """Запуск модели для генерации изображения"""
    if model_name in ['sd14', 'sd21', 'sdxl']:
        image = pipe(
            prompt=prompt, 
            num_inference_steps=num_denoising_steps, 
            generator=torch.Generator(device=device).manual_seed(seed)
        ).images[0]
      
    elif model_name in ['sd21-turbo', 'sdxl-turbo']:
        image = pipe(
            prompt=prompt, 
            num_inference_steps=num_denoising_steps,
            guidance_scale=0.0,
            generator=torch.Generator(device=device).manual_seed(seed)
        ).images[0]
            
    return image