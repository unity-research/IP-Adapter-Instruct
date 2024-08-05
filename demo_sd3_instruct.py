
import torch
print(torch.__version__)
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
import datetime

#from ip_adapter.ip_adapter_resample_input import IPAdapterPlusDual
from ip_adapter.ip_adapter_instruct import IPAdapter_sd3_Instruct
#from diffusers import DiffusionPipeline
from ip_adapter.pipeline_stable_diffusion_sd3_extra_cfg import StableDiffusion3PipelineExtraCFG
base_model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
#vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
#ip_ckpt = "models/ip_adapter12_33000.bin"
ip_ckpt = r'models/model_sd3_instruct_7_70000.bin'
#ip_ckpt = "models/ip-adapter-plus_sdxl_vit-h.bin"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
#vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load SD pipeline
pipe = None
pipe = StableDiffusion3PipelineExtraCFG.from_pretrained(
    base_model_path,
    feature_extractor=None,
    safety_checker=None,
    torch_dtype=torch.float16,
    
)
pipe.enable_model_cpu_offload()

image = Image.open("test_images/knight - Copy.png")
image.resize((512, 512))
#ip_ckpt= None
ip_model = IPAdapter_sd3_Instruct(pipe, image_encoder_path, ip_ckpt, device,num_tokens=16 )

# only image prompt
images = ip_model.generate(prompt="firefighter woman at the beach",negative_prompt="",pil_image=image, num_samples=1, num_inference_steps=30, seed=443228,scale=0.8,query="take the composition from the image")

#images = pipe(
#    "A cat holding a sign that says hello world",
#    negative_prompt="",
#    num_inference_steps=28,
#    guidance_scale=7.0,
#).images

grid = image_grid(images, 1, 1)
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_folder = './grid_test'
save_path = f"{save_folder}/grid_{timestamp}.png"

# Ensure the folder exists
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Save the grid
print("Saving images grid...")
grid.save(save_path)
print(f"Images grid saved to {save_path}")