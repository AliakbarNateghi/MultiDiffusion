# 1. Install necessary libraries
# Uncomment and run if you haven't installed them yet
# !pip install torch diffusers transformers accelerate gradio Pillow numpy tqdm opencv-python

# Ensure you have the `panorama.py` file in the same directory as this notebook
# or accessible in your Python path.

# 2. Import Libraries and Load Base Stable Diffusion Model
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

# Import our custom MultiDiffusion Panorama Pipeline
from panorama import MultiDiffusionPanoramaPipeline

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the base Stable Diffusion pipeline
# We recommend using a v1.5 model for general compatibility with MultiDiffusion
print("Loading Stable Diffusion v1.5 pipeline...")
model_id = "runwayml/stable-diffusion-v1-5"
base_sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
base_sd_pipeline.to(device)
print("Stable Diffusion pipeline loaded.")

# Initialize the MultiDiffusion Panorama Pipeline
panorama_pipeline = MultiDiffusionPanoramaPipeline(base_sd_pipeline)
print("MultiDiffusion Panorama Pipeline initialized.")

# Set your parameters here
prompt = "A breathtaking wide shot of an ancient, enchanted forest with towering trees, glowing magical fungi, and a misty, ethereal atmosphere, fantasy art, volumetric lighting"
negative_prompt = "blurry, low quality, distorted, ugly, dark, gloomy, monochrome, bad anatomy"
width = 1536  # Example: 3x standard 512 width
height = 512 # Standard height
guidance_scale = 8.0 # Experiment with values like 7.0, 9.0, 10.0
num_inference_steps = 50 # Common values are 20, 30, 50
seed = -1 # Set to an integer (e.g., 42) for reproducible results, -1 for random

# 4. Run Panorama Generation
if seed == -1:
    generated_seed = torch.seed() # Get a random seed from torch
else:
    generated_seed = seed

generator = torch.Generator(device=device).manual_seed(generated_seed)

print(f"Starting panorama generation with seed: {generated_seed}...")

output = panorama_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator
)

generated_image = output.images[0]

print("Panorama generation complete!")
print(f"Generated image size: {generated_image.size}")
print(f"Seed used for generation: {generated_seed}")

# 5. Display the Generated Panorama
display(generated_image)
