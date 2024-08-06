# IP Adapter Instruct: Resolving Ambiguity in Image-based Conditioning using Instruct Prompts

<a href='https://unity-research.github.io/IP-Adapter-Instruct.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href=''><img src='https://img.shields.io/badge/Technique-Report-red'></a> 
<a href='https://huggingface.co/CiaraRowles/IP-Adapter-Instruct'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>



We present: IP Adapter Instruct: By conditioning the transformer model used in IP-Adapter-Plus on additional text embeddings, one model can be used to effectively perform a wide range of image generation tasks with minimal setup.

Diffusion models continuously push the boundary of state-of-the-art image generation, but the process is hard to control with any nuance: practice proves that textual prompts are inadequate for accurately describing image style or fine structural details (such as faces).

ControlNet and IPAdapter address this shortcoming by conditioning the generative process on imagery instead, but each individual instance is limited to modeling a single conditional posterior: for practical use-cases, where multiple different posteriors are desired within the same workflow, training and using multiple adapters is cumbersome.

We propose IPAdapter-Instruct, which combines natural-image conditioning with "Instruct" prompts to swap between interpretations for the same conditioning image: style transfer, object extraction, both, or something else still? IPAdapterInstruct efficiently learns multiple tasks with minimal loss in quality compared to dedicated per-task models..





![header_image_small](https://github.com/user-attachments/assets/f37e5d54-8c2a-4278-a59b-66546f97e590)



How to use

```
pip install -r requirements.txt

download the models from here: https://huggingface.co/CiaraRowles/IP-Adapter-Instruct

place them in the "models" folder

run either demo.py, demo_sdxl.py or demo_sd3_instrct.py 

```

The possible current queries for each model are (it doesn't have to be exactly these queries, just this is the list of trained tasks)

```
sd15
"use everything from the image"
"use the style"
"use the colour"
"use the pose"
"use the composition"
"use the face"
"use the foreground" (beta)
"use the background" (beta)
"use the object"
```

```
sdxl and sd3
"use everything from the image"
"use the style"
"use the colour"
"use the pose"
"use the composition"
"use the face"
```

The demo scripts currently sort out the specifics of the cfg and scale settings, for your own implementation you may want to allow more control over these, as a general rule of thumb, everything, style, colour and face work better with simple cfg, use three way cfg for the other tasks, you may need to lower the scale to get good results with these later instruction types.

## Citation

WIP
