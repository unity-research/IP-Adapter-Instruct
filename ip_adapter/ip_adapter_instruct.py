import os
from typing import List
import torch.nn as nn

import torch
from PIL import Image
from safetensors import safe_open
from .ip_joint_attention import IPJointAttnProcessor2_0 , JointAttnProcessor2_0
from .joint_attention_block_modified import JointTransformerBlock_IP
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection,AutoImageProcessor,AutoModel,CLIPTokenizer,CLIPTextModel
from .utils import is_torch2_available,get_generator

#if is_torch2_available():
# from .attention_processor import (
#     AttnProcessor2_0 as AttnProcessor,
# )
# from .attention_processor import (
#     CNAttnProcessor2_0 as CNAttnProcessor,
# )
# from .attention_processor import (
#     IPAttnProcessor2_0 as IPAttnProcessor,
# )
#else:
#    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler_Instruct import ResamplerInstruct,MLP,ResamplerInstructBigger
from ip_adapter.ip_adapter import IPAdapter
from .resampler_SD3 import ResamplerSD3,ResamplerSD3_Instruct

class IPAdapterInstruct(IPAdapter):
    """IP-Adapter with fine-grained features"""
    def __init__ (self, sd_pipe, image_encoder_path,ip_ckpt, device, num_tokens=4,dtypein=torch.float32):
        #super().__init__()

        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        if ip_ckpt is not None:
            self.set_ip_adapter()

        # load image encoder
        #self.image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base").to(self.device, dtype=torch.float16)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device, dtype=dtypein)
        #self.clip_image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.clip_image_processor = CLIPImageProcessor()

        # image proj model
        self.image_proj_model = self.init_proj()
        print(ip_ckpt)
        if ip_ckpt is not None:
            self.load_ip_adapter()

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}, "mlp_proj": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        print(state_dict["image_proj"].keys(),state_dict["ip_adapter"].keys())
        print(self.image_proj_model.load_state_dict(state_dict["image_proj"],strict=False))
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        print(ip_layers.load_state_dict(state_dict["ip_adapter"],strict=False))


    def init_proj(self):
        image_proj_model = ResamplerInstructBigger(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim_image_embeds=self.image_encoder.config.hidden_size,
            embedding_dim_instruct_embeds=self.pipe.text_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
            ff_mult_secondary=2,
        ).to(self.device, dtype=torch.float16)

  
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None,instruct_embeds=None,negative_instruct_embeds=None,prompt_embeds=None,negative_prompt_embeds=None,instruct_embeds_everything=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        print(clip_image_embeds.dtype,instruct_embeds.dtype)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds,instruct_embeds,prompt_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds,negative_instruct_embeds,negative_prompt_embeds)
        img_only_prompt_embeds = self.image_proj_model(clip_image_embeds,negative_prompt_embeds,negative_prompt_embeds)
        img_prompt_everything_cond = self.image_proj_model(clip_image_embeds,instruct_embeds_everything,prompt_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds,img_only_prompt_embeds,img_prompt_everything_cond

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        query=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=6.0,
        image_guidance_scale=1.0,
        instruct_guidance_scale=6.0,
        num_inference_steps=30,
        width=512,
        height=512,
        auto_scale=True,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        print("prompt = ",prompt)
        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt_ids_instruction = self.pipe.tokenizer(
                query, max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            instruct_embeds = self.pipe.text_encoder(prompt_ids_instruction, return_dict=True)[0]
            prompt_ids_instruction_negative = self.pipe.tokenizer(
                "", max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            #TEMPORARY UNTIL IVE TRAINED IT TO CONDITION ON IMAGE ONLY WITH NO INSTRUCTION
            prompt_ids_instruction_everything_temp = self.pipe.tokenizer(
                "everything", max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            instruct_embeds_everything = self.pipe.text_encoder(prompt_ids_instruction_everything_temp, return_dict=True)[0]

            negative_instruct_embeds = self.pipe.text_encoder(prompt_ids_instruction_negative, return_dict=True)[0]  

            prompt_embeds_ = prompt_embeds_[0].unsqueeze(0)
            negative_prompt_embeds_ = negative_prompt_embeds_[0].unsqueeze(0)



            image_prompt_embeds, uncond_image_prompt_embeds,img_only_prompt_embeds,img_prompt_everything_cond = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds,instruct_embeds=instruct_embeds,negative_instruct_embeds=negative_instruct_embeds,prompt_embeds=prompt_embeds_,negative_prompt_embeds=negative_prompt_embeds_,instruct_embeds_everything=instruct_embeds_everything
            )

            prompt_embeds_ = prompt_embeds_.repeat(num_samples, 1,1)
            negative_prompt_embeds_ = negative_prompt_embeds_.repeat(num_samples, 1,1)
        
            
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        img_only_prompt_embeds = img_only_prompt_embeds.repeat(1, num_samples, 1)
        img_only_prompt_embeds = img_only_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        img_prompt_everything_cond = img_prompt_everything_cond.repeat(1, num_samples, 1)
        img_prompt_everything_cond = img_prompt_everything_cond.view(bs_embed * num_samples, seq_len, -1)

        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        img_only_prompt_embeds = torch.cat([negative_prompt_embeds_, img_only_prompt_embeds], dim=1)
        img_prompt_everything_cond = torch.cat([negative_prompt_embeds_, img_prompt_everything_cond], dim=1)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        simple_cfg_mode = False
        if "style" in query or "colour" in query or "everything" in query or "color" in query or "face" in query or "facial" in query or "colour" in query:
            simple_cfg_mode = True

        #replace colour with color
        if "colour" in query:
            query = query.replace("colour","color")
            #llm disrespect generating the training queries ngl
        if auto_scale:
            if "composition" in query or "pose" in query:
                scale = scale - 0.3
            #else:
            #    scale = 0.9
        self.set_scale(scale)
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            img_only_prompt_embeds=img_only_prompt_embeds,
            img_prompt_everything_cond=img_prompt_everything_cond,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            instruct_guidance_scale=instruct_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            width=width,
            height=height,
            simple_cfg_mode=simple_cfg_mode,
            **kwargs,
        ).images

        return images
    



class IPAdapterInstructSDXL(IPAdapterInstruct):

    def init_proj(self):
        image_proj_model = ResamplerInstructBigger(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim_image_embeds=self.image_encoder.config.hidden_size,
            embedding_dim_instruct_embeds=self.pipe.text_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        query=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        guidance_scale=5.0,
        image_guidance_scale=1.0,
        instruct_guidance_scale=5.0,
        num_inference_steps=30,
        simple_cfg_mode=False,
        auto_scale=False,
        **kwargs,
    ):

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        print("prompt = ",prompt)
        with torch.inference_mode():
            (
                prompt_embeds_,
                negative_prompt_embeds_,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_ids_prompt = self.pipe.tokenizer(
                prompt, max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            negative_prompt_ids = self.pipe.tokenizer(
                "", max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            prompt_ids_instruction = self.pipe.tokenizer(
                query, max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            instruct_embeds = self.pipe.text_encoder(prompt_ids_instruction, return_dict=True)[0]
            prompt_ids_instruction_negative = self.pipe.tokenizer(
                "", max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            negative_instruct_embeds = self.pipe.text_encoder(prompt_ids_instruction_negative, return_dict=True)[0]  

            prompt_embeds_ = prompt_embeds_[0].unsqueeze(0)
            negative_prompt_embeds_ = negative_prompt_embeds_[0].unsqueeze(0)
            prompt_ids_instruction_everything_temp = self.pipe.tokenizer(
                "everything", max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            instruct_embeds_everything = self.pipe.text_encoder(prompt_ids_instruction_everything_temp, return_dict=True)[0]

            print("hidden sizes",self.pipe.text_encoder.config.hidden_size,self.pipe.text_encoder_2.config.hidden_size)
            small_prompt_embeds =  self.pipe.text_encoder(prompt_ids_prompt, return_dict=True)[0]
            small_negative_prompt_embeds = self.pipe.text_encoder(negative_prompt_ids, return_dict=True)[0]
            image_prompt_embeds, uncond_image_prompt_embeds,img_only_prompt_embeds,img_prompt_everything_cond = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds,instruct_embeds=instruct_embeds,negative_instruct_embeds=negative_instruct_embeds,prompt_embeds=small_prompt_embeds,negative_prompt_embeds=small_negative_prompt_embeds,instruct_embeds_everything=instruct_embeds_everything
            )

            prompt_embeds_ = prompt_embeds_.repeat(num_samples, 1,1)
            negative_prompt_embeds_ = negative_prompt_embeds_.repeat(num_samples, 1,1)
        
            
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        img_only_prompt_embeds = img_only_prompt_embeds.repeat(1, num_samples, 1)
        img_only_prompt_embeds = img_only_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        img_prompt_everything_cond = img_prompt_everything_cond.repeat(1, num_samples, 1)
        img_prompt_everything_cond = img_prompt_everything_cond.view(bs_embed * num_samples, seq_len, -1)

        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        img_only_prompt_embeds = torch.cat([negative_prompt_embeds_, img_only_prompt_embeds], dim=1)
        img_prompt_everything_cond = torch.cat([negative_prompt_embeds_, img_prompt_everything_cond], dim=1)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        #simple_cfg_mode = False
        if "style" in query or "colour" in query or "everything" in query or "color" in query or "face" in query or "facial" in query or "colour" in query:
            simple_cfg_mode = True

        #replace colour with color
        if "colour" in query:
            query = query.replace("colour","color")
            #llm disrespect generating the training queries ngl

        if auto_scale:
            if "composition" in query:
                scale = 0.7
            #else:
            #    scale = 0.9
        self.set_scale(scale)


        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            pooled_prompt_embeds=pooled_prompt_embeds,
            img_only_prompt_embeds=img_only_prompt_embeds,
            img_prompt_everything_cond=img_prompt_everything_cond,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            instruct_guidance_scale=instruct_guidance_scale,
            generator=generator,
            simple_cfg_mode=simple_cfg_mode,
            **kwargs,
        ).images

        return images
    

def replace_transformer_blocks(original_model):
    original_model.transformer_blocks = nn.ModuleList(
        [
            JointTransformerBlock_IP(
                dim=original_model.inner_dim,
                num_attention_heads=original_model.config.num_attention_heads,
                attention_head_dim=original_model.inner_dim,
                context_pre_only=i == original_model.config.num_layers - 1,
            )
            for i in range(original_model.config.num_layers)
        ]
    )
    return original_model


class IPAdapter_sd3:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        #self.dino_image_encoder = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device, dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()
        #self.dino_image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

        # image proj model
        self.image_proj_model = self.init_proj()
        if self.ip_ckpt is not None:
            self.load_ip_adapter()





    def init_proj(self):
        image_proj_model = ResamplerSD3(
                dim=1280, #cross attention dim
                depth=4, # a bit bigger
                dim_head=64,
                heads=20,
                num_queries=self.num_tokens,#num tokens
                embedding_dim=self.image_encoder.config.hidden_size,
                output_dim=4096,
                ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self, n=2):
        attn_procs = {}
        transformer_sd = self.pipe.transformer.state_dict()

        old_state_dict = self.pipe.transformer.state_dict()
        self.pipe.transformer = replace_transformer_blocks(self.pipe.transformer)
        self.pipe.transformer.load_state_dict(old_state_dict, strict=True)

        for i, name in enumerate(self.pipe.transformer.attn_processors.keys()):
            layer_name = name.split(".processor")[0]
            hidden_size = self.pipe.transformer.config.joint_attention_dim
            
            weights = {
                "to_k_ip.weight": transformer_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": transformer_sd[layer_name + ".to_v.weight"],
                "add_k_proj_ip.weight": transformer_sd[layer_name + ".add_k_proj.weight"],
                "add_v_proj_ip.weight": transformer_sd[layer_name + ".add_v_proj.weight"],
            }


            if i % n == 0 or i == 0:  # Apply IPJointAttnProcessor2_0 to every nth block
                attn_procs[name] = IPJointAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=self.pipe.transformer.config.caption_projection_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                    num_heads=12,
                ).to(self.device, dtype=torch.float16)
                attn_procs[name].load_state_dict(weights,strict=False)


            else:  # Use JointAttnProcessor2_0 for other blocks
                attn_procs[name] = JointAttnProcessor2_0(
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)

        self.pipe.transformer.set_attn_processor(attn_procs)
        self.pipe.transformer = self.pipe.transformer.to(self.device, dtype=torch.float16)




    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        print(self.image_proj_model.load_state_dict(state_dict["image_proj"],strict=False))
        ip_layers = torch.nn.ModuleList(self.pipe.transformer.attn_processors.values())
        print(ip_layers.load_state_dict(state_dict["ip_adapter"],strict=False)) #for testing



        

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)

        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        #dino_inputs = self.dino_image_processor(images=pil_image, return_tensors="pt").to(self.device, dtype=torch.float16)
        #dino_image_embeds = self.dino_image_encoder(**dino_inputs)[0]
       # all_image_embeds = torch.cat([clip_image_embeds, dino_image_embeds], dim=-1)

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_dino_embeds = torch.zeros_like(clip_image_embeds)
        #all_uncond = torch.cat([uncond_clip_image_embeds, uncond_dino_embeds], dim=-1)

        uncond_image_prompt_embeds = self.image_proj_model(uncond_dino_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.transformer.attn_processors.values():
            if isinstance(attn_processor, IPJointAttnProcessor2_0):
                attn_processor.scale = scale

    def generate(
            self,
            pil_image=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            **kwargs,
        ):
            self.set_scale(scale)

            if pil_image is not None:
                num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
            else:
                num_prompts = clip_image_embeds.size(0)

            if prompt is None:
                prompt = "best quality, high quality"
            if negative_prompt is None:
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            if not isinstance(prompt, List):
                prompt = [prompt] * num_prompts
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_prompts

            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=pil_image, clip_image_embeds=clip_image_embeds
            )
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)


            print("imge prompt embeds = ",image_prompt_embeds.shape)
            with torch.inference_mode():
                prompt_embeds_, negative_prompt_embeds_,pooled,negative_pooled = self.pipe.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt,
                    prompt_3=prompt,
                    device=self.device,
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1).to(self.device, dtype=torch.float16) 
                negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1).to(self.device, dtype=torch.float16) 

            generator = get_generator(seed, self.device)

            images = self.pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                **kwargs,
            ).images

            return images


class IPAdapter_sd3_Instruct (IPAdapter_sd3):
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.tokenizer_instruct = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        self.text_encoder_Instruct = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(self.device, dtype=torch.float16)   
        #self.dino_image_encoder = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device, dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()
        #self.dino_image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

        # image proj model
        self.image_proj_model = self.init_proj()
        if self.ip_ckpt is not None:
            self.load_ip_adapter()





    def init_proj(self):
        image_proj_model = ResamplerSD3_Instruct(
        dim=1280, #cross attention dim
        depth=4, # a bit bigger
        dim_head=64,
        heads=20,
        num_queries=self.num_tokens,#num tokens
        embedding_dim_image_embeds=self.image_encoder.config.hidden_size,
        embedding_dim_instruct_embeds=self.text_encoder_Instruct.config.hidden_size,
        output_dim=4096,
        ff_mult=4,
    ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self, n=2):
        attn_procs = {}
        transformer_sd = self.pipe.transformer.state_dict()

        old_state_dict = self.pipe.transformer.state_dict()
        self.pipe.transformer = replace_transformer_blocks(self.pipe.transformer)
        self.pipe.transformer.load_state_dict(old_state_dict, strict=True)

        for i, name in enumerate(self.pipe.transformer.attn_processors.keys()):
            layer_name = name.split(".processor")[0]
            hidden_size = self.pipe.transformer.config.joint_attention_dim
            
            weights = {
                "to_k_ip.weight": transformer_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": transformer_sd[layer_name + ".to_v.weight"],
                "add_k_proj_ip.weight": transformer_sd[layer_name + ".add_k_proj.weight"],
                "add_v_proj_ip.weight": transformer_sd[layer_name + ".add_v_proj.weight"],
            }


            if i % n == 0 or i == 0:  # Apply IPJointAttnProcessor2_0 to every nth block
                attn_procs[name] = IPJointAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=self.pipe.transformer.config.caption_projection_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                    num_heads=12,
                ).to(self.device, dtype=torch.float16)
                attn_procs[name].load_state_dict(weights,strict=False)


            else:  # Use JointAttnProcessor2_0 for other blocks
                attn_procs[name] = JointAttnProcessor2_0(
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)

        self.pipe.transformer.set_attn_processor(attn_procs)
        self.pipe.transformer = self.pipe.transformer.to(self.device, dtype=torch.float16)




    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        print(self.image_proj_model.load_state_dict(state_dict["image_proj"],strict=False))
        ip_layers = torch.nn.ModuleList(self.pipe.transformer.attn_processors.values())
        print(ip_layers.load_state_dict(state_dict["ip_adapter"],strict=False)) #for testing



        

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None,instruct_embeds=None,negative_instruct_embeds=None,prompt_embeds_small=None,negative_prompt_embeds_small=None,instruct_embeds_everything=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)

        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        

        image_prompt_embeds = self.image_proj_model(clip_image_embeds,instruct_embeds,prompt_embeds_small)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
   
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds,negative_instruct_embeds,negative_prompt_embeds_small)
        img_only_prompt_embeds = self.image_proj_model(clip_image_embeds,negative_prompt_embeds_small,negative_prompt_embeds_small)

        img_prompt_everything_cond = self.image_proj_model(clip_image_embeds,instruct_embeds_everything,prompt_embeds_small)
        
        return image_prompt_embeds, uncond_image_prompt_embeds,img_only_prompt_embeds,img_prompt_everything_cond

    def set_scale(self, scale):
        for attn_processor in self.pipe.transformer.attn_processors.values():
            if isinstance(attn_processor, IPJointAttnProcessor2_0):
                attn_processor.scale = scale

    def generate(
            self,
            pil_image=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=6.0,
            image_guidance_scale=1.0,
            instruct_guidance_scale=6.0, 
            num_inference_steps=30,
            simple_cfg_mode=False,
            query="",
            **kwargs,
        ):
            self.set_scale(scale)

            if pil_image is not None:
                num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
            else:
                num_prompts = clip_image_embeds.size(0)

            if prompt is None:
                prompt = "best quality, high quality"
            if negative_prompt is None:
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            if not isinstance(prompt, List):
                prompt = [prompt] * num_prompts
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_prompts




            #print("imge prompt embeds = ",image_prompt_embeds.shape)
            with torch.inference_mode():
                prompt_embeds_, negative_prompt_embeds_,pooled,negative_pooled = self.pipe.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt,
                    prompt_3=prompt,
                    device=self.device,
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                
                prompt_ids_instruction_everything_temp = self.tokenizer_instruct(
                    "everything", max_length=self.tokenizer_instruct.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(self.device)
                instruct_embeds_everything= self.text_encoder_Instruct(prompt_ids_instruction_everything_temp, return_dict=True)[0]

                prompt_ids_instruction = self.tokenizer_instruct(
                    query, max_length=self.tokenizer_instruct.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(self.device)
                instruct_embeds = self.text_encoder_Instruct(prompt_ids_instruction, return_dict=True)[0]
                prompt_ids_instruction_negative = self.tokenizer_instruct(
                    "", max_length=self.tokenizer_instruct.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(self.device)
                negative_instruct_embeds = self.text_encoder_Instruct(prompt_ids_instruction_negative, return_dict=True)[0]
                prompt_ids_prompt_small = self.tokenizer_instruct(
                    prompt, max_length=self.tokenizer_instruct.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(self.device)
                prompt_embeds_small = self.text_encoder_Instruct(prompt_ids_prompt_small, return_dict=True)[0]
                negative_prompt_ids_prompt_small = self.tokenizer_instruct(
                    negative_prompt, max_length=self.tokenizer_instruct.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(self.device)
                negative_prompt_embeds_small = self.text_encoder_Instruct(negative_prompt_ids_prompt_small, return_dict=True)[0]
            print(instruct_embeds_everything.shape)
            image_prompt_embeds, uncond_image_prompt_embeds,img_only_prompt_embeds,img_prompt_everything_cond = self.get_image_embeds(
                pil_image=pil_image, clip_image_embeds=clip_image_embeds,instruct_embeds=instruct_embeds,negative_instruct_embeds=negative_instruct_embeds,prompt_embeds_small=prompt_embeds_small,negative_prompt_embeds_small=negative_prompt_embeds_small,instruct_embeds_everything=instruct_embeds_everything,
            )
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)


            img_only_prompt_embeds = img_only_prompt_embeds.repeat(1, num_samples, 1)
            img_only_prompt_embeds = img_only_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            img_prompt_everything_cond = img_prompt_everything_cond.repeat(1, num_samples, 1)
            img_prompt_everything_cond = img_prompt_everything_cond.view(bs_embed * num_samples, seq_len, -1)

            generator = get_generator(seed, self.device)

            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1).to(self.device, dtype=torch.float16) 
            img_only_prompt_embeds = torch.cat([negative_prompt_embeds_, img_only_prompt_embeds], dim=1)
            img_prompt_everything_cond = torch.cat([negative_prompt_embeds_, img_prompt_everything_cond], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1).to(self.device, dtype=torch.float16) 

            if "style" in query or "colour" in query or "everything" in query or "color" in query:
                simple_cfg_mode = True

            #replace colour with color
            if "colour" in query:
                query = query.replace("colour","color")

            images = self.pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=negative_prompt_embeds,
                img_only_prompt_embeds=img_only_prompt_embeds,
                img_prompt_everything_cond=img_prompt_everything_cond,
                negative_pooled_prompt_embeds=negative_pooled,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                instruct_guidance_scale=instruct_guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                simple_cfg_mode=simple_cfg_mode,
                **kwargs,
            ).images

            return images