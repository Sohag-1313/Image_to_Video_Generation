import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# Load the base pre-trained model
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "/workspace/pretrained_model", torch_dtype=torch.float16  # Use float16 for compatibility
).to("cuda")

# Load LoRA fine-tuned weights
pipe.load_lora_weights(
    "/workspace/fine_tuned_hug/",
    weight_name="hug_150.safetensors", 
    adapter_name="cogvideox-lora"
)

# Set LoRA adapters and scaling
#LoRA scaling (0.5–1.0).
pipe.set_adapters(["cogvideox-lora"], [0.5])  # Adjust scaling factor if needed

# Load the input image
# image = load_image("/workspace/images/hugg__.jpg")
image = load_image("/workspace/images/hugg__.jpg")


# Define the validation prompt
validation_prompt = "One man and one woman standing close to each other. then slowly move closer. The man wraps their arms around the woman's waist, and the womanman gently hugs the man back. They embrace warmly, smiling and showing happiness."
# validation_prompt_7 = "One baby and one woman standing close to each other. then slowly move closer. The woman wraps their arms around the baby's waist, and the baby gently hugs the womman back. They embrace warmly, smiling and showing happiness."
# validation_prompt = "One young boy and one young girl standing close to each other. then slowly move closer. The young boy wraps their arms around the young girl's waist, and the young girl gently hugs the young boy back. They embrace warmly, smiling and showing happiness."
# validation_prompt = "One man and one woman standing close to each other. Then they hug each other. They appear to be happy and enjoying their time together."
# validation_prompt = "A young child wearing a red shirt and a Santa hat stands next to an older man wearing a Santa hat. They smile at each other, then slowly move closer. The child wraps their arms around the older man's waist, and the older man gently hugs the child back. They embrace warmly, smiling and showing happiness."
# hug_1_prompt = "One man and one woman standing close to each other. then slowly move closer. The man wraps their arms around the woman's waist, and the womanman gently hugs the man back. They embrace warmly, smiling and showing happiness."
# validation_prompt = "A young child wearing a red shirt and a Santa hat and older man who is also wearing a Santa hat standing close to each other. A young child wearing a red shirt and a Santa hat, hugging an older man who is also wearing a Santa hat. The background is decorated with festive lights and sparkles, creating a cheerful and festive atmosphere. The child appears to be happy and excited, while the older man looks content and joyful. The overall scene conveys the spirit of Christmas and the joy of spending time with loved ones during the holiday season."
# validation_prompt = "One man and one woman standing close to each other. They move slowly toward each other and hug with each other. The background remains calm and consistent, emphasizing their happy expressions and body movement as they hug."
# validation_prompt = "Make a gentle hug video. The background remains calm and consistent, emphasizing their hug to each other"

# Run inference
video = pipe(
    image=image, 
    prompt=validation_prompt, 
    # Adjust guidance scale (3.5–5.5).
    guidance_scale=4,  # Try reducing guidance scale
    use_dynamic_cfg=False  # Disable dynamic CFG if not used during training
).frames[0]

# Export the generated video
export_to_video(video, "/workspace/output_hug_1/hugg__.mp4", fps=8)
