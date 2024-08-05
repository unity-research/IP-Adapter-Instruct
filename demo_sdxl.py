
import torch
print(torch.__version__)
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL,DiffusionPipeline
from PIL import Image
import datetime
from ip_adapter.pipeline_stable_diffusion_sdxl_extra_cfg import StableDiffusionXLPipelineExtraCFG

from ip_adapter.ip_adapter_instruct import IPAdapterInstructSDXL
#from ip_adapter import IPAdapter

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
#vae_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "models/model_sdxl_3_30000.bin"
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
pipe = StableDiffusionXLPipelineExtraCFG.from_pretrained(
    base_model_path,
    scheduler=noise_scheduler,
   ## vae=vae,
    torch_dtype=torch.float16,
    feature_extractor=None,
    safety_checker=None
)

image = Image.open("./test_images/knight.png")
image.resize((512, 512))

ip_model = IPAdapterInstructSDXL(pipe, image_encoder_path, ip_ckpt, device,dtypein=torch.float16,num_tokens=16)
query="use the composition from the image"
instruct_scale = 4.0
if "composition" in query or "background" in query or "object" in query:
    instruct_scale = 6.0
# only image prompt
images = ip_model.generate(prompt="a photo of a firefighter woman",pil_image=image, num_samples=2, num_inference_steps=25, seed=5333332,query=query,scale=0.9,simple_cfg_mode=False,guidance_scale=7.0,instruct_guidance_scale=instruct_scale,image_guidance_scale=0.5)
grid = image_grid(images, 1, 2)
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_folder = 'D:\\gitprojects\\IP-Adapter-Instruct\\grid_test'
save_path = f"{save_folder}\\grid_{timestamp}.png"

# Ensure the folder exists
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Save the grid
print("Saving images grid...")
grid.save(save_path)
print(f"Images grid saved to {save_path}")