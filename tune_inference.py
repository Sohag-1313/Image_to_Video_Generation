import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
repo_id = "1"
pipe = CogVideoXImageToVideoPipeline.from_pretrained("/workspace/pretrained_model", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("/workspace/pretrained_model",
                        weight_name="/workspace/output/cogvideox-lora__optimizer_adam__steps_10__lr-schedule_cosine_with_restarts__learning-rate_1e-4/pytorch_lora_weights.safetensors", 
                        adapter_name="cogvideox-lora")

# The LoRA adapter weights are determined by what was used for training.
# In this case, we assume `--lora_alpha` is 32 and `--rank` is 64.
# It can be made lower or higher from what was used in training to decrease or amplify the effect
# of the LoRA upto a tolerance, beyond which one might notice no effect at all or overflows.
pipe.set_adapters(["cogvideox-lora"], [128 / 128])

image = load_image("/workspace/images/hugg__.jpg")
validation_prompt = "One man and one woman standing close to each other. Then they hugging each other. They appear to be happy and enjoying their time together."
video = pipe(image=image, prompt="{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video(video, "toutput.mp4", fps=8)