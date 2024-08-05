
import torch
print(torch.__version__)
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
import datetime
from ip_adapter.pipeline_stable_diffusion_extra_cfg import StableDiffusionPipelineCFG

from ip_adapter.ip_adapter_instruct import IPAdapterInstruct

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "models/model_12_40000.bin"
#ip_ckpt = "models/ip-adapter-plus_sd15.bin"
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
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load SD pipeline
pipe = StableDiffusionPipelineCFG.from_pretrained(
    base_model_path,
    scheduler=noise_scheduler,
    vae=vae,
    torch_dtype=torch.float16,
    feature_extractor=None,
    safety_checker=None
)

image = Image.open("./test_images/river.png")
image.resize((512, 768))

ip_model = IPAdapterInstruct(pipe, image_encoder_path, ip_ckpt, device,dtypein=torch.float16,num_tokens=16)

# only image prompt
images = ip_model.generate(prompt="a politician woman",pil_image=image, num_samples=4, seed=52222246, query="use the face",scale=0.8,guidance_scale=5.0,instruct_guidance_scale=2.0,image_guidance_scale=1.0,width=512,height=768)
grid = image_grid(images, 1, 4)
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