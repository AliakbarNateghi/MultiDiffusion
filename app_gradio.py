import gradio as gr
import torch
from PIL import Image
import numpy as np
import uuid # For unique session IDs

# Import the MultiDiffusion pipelines
from panorama import MultiDiffusionPanoramaPipeline
from region_based import MultiDiffusionRegionBasedPipeline
from illusion import MultiDiffusionIllusionDiffusionPipeline

# --- Model Loading ---
# Using a common function to load the model
# Consider caching this for better performance if running locally
@gr.cache
def load_sd_pipeline():
    """Loads the Stable Diffusion pipeline."""
    # This might take a while, so it's good to cache it or load once globally
    # For a Gradio app, it's often best to load once at the top level
    from diffusers import StableDiffusionPipeline
    model_id = "runwayml/stable-diffusion-v1-5" # Or "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")
    return pipe

# Load the base pipeline once globally to avoid reloading on each request
# This requires `gr.load` or manual handling if not using `gr.Interface` directly with caching
# For a direct script, it's common to load here:
print("Loading Stable Diffusion pipeline...")
base_sd_pipeline = load_sd_pipeline()
print("Stable Diffusion pipeline loaded.")

# --- MultiDiffusion Pipeline Instances ---
panorama_pipeline = MultiDiffusionPanoramaPipeline(base_sd_pipeline)
region_based_pipeline = MultiDiffusionRegionBasedPipeline(base_sd_pipeline)
illusion_pipeline = MultiDiffusionIllusionDiffusionPipeline(base_sd_pipeline)

# --- Inference Functions ---

def run_panorama_inference(prompt: str, negative_prompt: str, width: int, height: int,
                           guidance_scale: float, num_inference_steps: int, seed: int):
    """Runs the panorama generation with MultiDiffusion."""
    if not prompt:
        return None, "Please enter a positive prompt."
    if seed == -1:
        seed = torch.seed() # Use torch's current random seed if -1 is provided
    generator = torch.Generator("cuda").manual_seed(seed)

    print(f"Generating panorama for prompt: '{prompt}', neg_prompt: '{negative_prompt}', "
          f"width: {width}, height: {height}, guidance: {guidance_scale}, "
          f"steps: {num_inference_steps}, seed: {seed}")
    
    # Pass all new parameters to the pipeline
    result = panorama_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]
    return result, f"Panorama generated successfully! Seed: {seed}"

def run_region_based_inference(prompt_regions: str, negative_prompt: str,
                               guidance_scale: float, num_inference_steps: int, seed: int):
    """Runs region-based generation with MultiDiffusion."""
    if not prompt_regions:
        return None, "Please enter prompt regions."
    if seed == -1:
        seed = torch.seed()
    generator = torch.Generator("cuda").manual_seed(seed)

    # Example: Parse prompt_regions string into a list of (mask_image, prompt) tuples
    # For a more robust app, this would involve UI elements for masks.
    # Here, we'll assume prompt_regions is a JSON string or a simple comma-separated list
    # For simplicity, let's assume it's just a main prompt for now and the actual mask
    # input would be handled by Gradio's image component for mask drawing.
    
    # This example needs actual mask input. For a live demo, Gradio's image editor is great.
    # For now, we'll simulate a simple case or assume masks are pre-defined/uploaded
    # For a real Gradio app, `gr.Image(type="filepath", tool="sketch")` would be used
    # and the mask data would be passed.
    
    # Placeholder for mask input. In a real Gradio app, you'd pass a gr.Image component.
    # For this example, let's simplify and make the prompt_regions just the main prompt,
    # and the mask handling (e.g., random or a fixed one) happens inside the pipeline.
    
    # The original region_based.py expects a list of (mask, text).
    # This requires a more complex UI. For now, let's provide a basic example input.
    # In a real scenario, Gradio's image editing tools would provide the masks.

    print(f"Generating region-based image for prompt_regions: '{prompt_regions}', "
          f"neg_prompt: '{negative_prompt}', guidance: {guidance_scale}, "
          f"steps: {num_inference_steps}, seed: {seed}")
    
    # Simulating a basic input for demonstration.
    # The `prompt_regions` would need to be parsed into (mask, text) pairs.
    # Let's assume `prompt_regions` is a single string for now and you'll adapt it
    # with real mask inputs in your Gradio setup.
    # Example: masks_and_prompts = [(mask_image1, "a red car"), (mask_image2, "a blue sky")]
    
    # To make this runnable without complex UI for masks, let's make it accept a main prompt
    # and use a dummy mask for now or assume region_based_pipeline handles parsing.
    # A more complete Gradio app would have a gr.Image with "sketch" tool for mask input.

    # Example: Assume prompt_regions is a JSON string like
    # '[{"mask_path": "path/to/mask1.png", "prompt": "a dog"}, {"mask_path": "path/to/mask2.png", "prompt": "a cat"}]'
    # Or, for simplicity, let's just use the main text prompt and imply a single mask or global generation
    # if the UI doesn't provide explicit masks.
    
    # The `MultiDiffusionRegionBasedPipeline` expects `regions` as `List[Tuple[Image.Image, str]]`.
    # This Gradio interface can't easily get multiple image inputs per region.
    # Let's simplify the Gradio input for `region_based_inference` to just a main prompt and
    # assume the masks or regions are defined internally or by a pre-loaded setup for the demo.
    
    # For a more creative UI, you'd need `gr.Image(tool="sketch")` for *each* region.
    # Or a dynamic UI where users add regions.
    
    # For the purpose of making the code runnable and demonstrating parameter passing,
    # let's assume `prompt_regions` acts as the main prompt for a single region or global generation.
    # The actual implementation of multi-region input needs a more complex Gradio structure.

    # TEMPORARY SIMPLIFICATION:
    # Assuming a single, global prompt for demonstration
    # In a real app, you'd design the UI to input multiple masks and prompts.
    # This might require a custom Gradio component or a more involved state management.
    
    # As the original region_based.py uses `prompts` (list of strings) and `masks` (list of PIL Images),
    # this Gradio interface should provide that.
    # For a practical Gradio demo, a single prompt and a single mask drawing tool:
    # Let's use `gr.Image(type="numpy", tool="sketch")` in the UI to get a mask.
    
    # REVISED PLAN for run_region_based_inference:
    # The Gradio UI will provide a main prompt and an image input where user can draw a mask.
    # The mask will be assumed to define the "region" for the main prompt.
    # This simplifies the UI while still showing region-based generation.
    
    # The original `region_based.py` pipeline expects `regions` as `List[Tuple[Image.Image, str]]`
    # Let's adapt this by providing a single mask and a single prompt as input.
    # This implies the UI will have `main_prompt` (text) and `mask_image` (image with sketch).
    
    # To correctly handle the `regions` input for `region_based_pipeline`, we need:
    # `image_with_mask` from Gradio's `gr.Image(tool="sketch")`
    # `region_prompt` from Gradio's `gr.Textbox`
    
    # Let's modify the `run_region_based_inference` signature to accept `mask_image` directly.
    # The current `prompt_regions` is too generic.
    pass # This function will be updated below with `mask_image` input
    # It requires a change in the Gradio interface definition later.

def run_illusion_inference(prompt: str, control_image: Image.Image, negative_prompt: str,
                           guidance_scale: float, num_inference_steps: int, seed: int,
                           ddim_steps: int = 50, strength: float = 0.5):
    """Runs illusion diffusion with MultiDiffusion."""
    if not prompt:
        return None, "Please enter a positive prompt."
    if control_image is None:
        return None, "Please upload a control image for illusion diffusion."
    if seed == -1:
        seed = torch.seed()
    generator = torch.Generator("cuda").manual_seed(seed)

    print(f"Generating illusion for prompt: '{prompt}', neg_prompt: '{negative_prompt}', "
          f"control_image shape: {control_image.size}, guidance: {guidance_scale}, "
          f"steps: {num_inference_steps}, seed: {seed}, strength: {strength}")
    
    result = illusion_pipeline(
        prompt=prompt,
        control_image=control_image,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        ddim_steps=ddim_steps, # Ensure this matches the pipeline's expected argument
        strength=strength # Ensure this matches the pipeline's expected argument
    ).images[0]
    return result, f"Illusion generated successfully! Seed: {seed}"

# --- Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # MultiDiffusion: Creative Image Generation
        A unified framework for versatile and controllable image generation using a pre-trained text-to-image diffusion model.
        Explore Panorama, Region-Based, and Illusion Diffusion modes!
        """
    )
    
    with gr.Tabs() as mode_tabs:
        # --- Panorama Tab ---
        with gr.TabItem("Panorama Generation"):
            with gr.Row():
                with gr.Column():
                    panorama_prompt = gr.Textbox(label="Positive Prompt", placeholder="e.g., a beautiful snowy mountain range")
                    panorama_negative_prompt = gr.Textbox(label="Negative Prompt (Optional)", placeholder="e.g., blurry, bad quality, low resolution")
                    with gr.Row():
                        panorama_width = gr.Slider(minimum=512, maximum=4096, step=64, value=1024, label="Width")
                        panorama_height = gr.Slider(minimum=512, maximum=2048, step=64, value=512, label="Height")
                    with gr.Row():
                        panorama_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=7.5, label="Guidance Scale", info="Higher values make the image more faithful to the prompt.")
                        panorama_num_inference_steps = gr.Slider(minimum=10, maximum=100, step=5, value=50, label="Inference Steps", info="More steps generally lead to better quality.")
                    panorama_seed = gr.Number(label="Seed (for reproducibility, -1 for random)", value=-1, precision=0)
                    panorama_generate_btn = gr.Button("Generate Panorama", variant="primary")
                with gr.Column():
                    panorama_output = gr.Image(label="Generated Panorama", type="pil")
                    panorama_status = gr.Textbox(label="Status", interactive=False)
            
            panorama_generate_btn.click(
                fn=run_panorama_inference,
                inputs=[
                    panorama_prompt,
                    panorama_negative_prompt,
                    panorama_width,
                    panorama_height,
                    panorama_guidance_scale,
                    panorama_num_inference_steps,
                    panorama_seed
                ],
                outputs=[panorama_output, panorama_status]
            )

        # --- Region-Based Tab ---
        with gr.TabItem("Region-Based Generation"):
            # REVISED INTERFACE FOR REGION-BASED:
            # Simplifies to one main prompt and one mask drawable by the user.
            # For multiple regions, a more complex UI (e.g., dynamic adding of image/text pairs)
            # would be needed, which is beyond a quick enhancement.
            
            gr.Markdown(
                """
                ### Region-Based Generation (Simplified Demo)
                Draw a mask on the black canvas below to define a region, and enter a prompt for that region.
                For multi-region control, you'd need a more advanced UI (e.g., multiple mask/prompt inputs).
                """
            )
            
            with gr.Row():
                with gr.Column():
                    region_main_prompt = gr.Textbox(label="Main Prompt for Masked Region", placeholder="e.g., a red car")
                    region_mask_image = gr.Image(label="Draw Mask Here (Black for background, White for region)", 
                                                 source="upload", tool="sketch", type="pil", height=256, width=256,
                                                 image_mode="L", invert_colors=False, brush_radius=15, brush_color=(255, 255, 255))
                    region_negative_prompt = gr.Textbox(label="Negative Prompt (Optional)", placeholder="e.g., blurry, bad quality")
                    region_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=7.5, label="Guidance Scale")
                    region_num_inference_steps = gr.Slider(minimum=10, maximum=100, step=5, value=50, label="Inference Steps")
                    region_seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    region_generate_btn = gr.Button("Generate Region-Based Image", variant="primary")
                with gr.Column():
                    region_output = gr.Image(label="Generated Image", type="pil")
                    region_status = gr.Textbox(label="Status", interactive=False)

            # Updated function to handle single mask input
            def run_single_region_inference(main_prompt: str, mask_image: np.ndarray, negative_prompt: str,
                                            guidance_scale: float, num_inference_steps: int, seed: int):
                if not main_prompt:
                    return None, "Please enter a main prompt."
                if mask_image is None:
                    return None, "Please draw or upload a mask."
                if seed == -1:
                    seed = torch.seed()
                generator = torch.Generator("cuda").manual_seed(seed)

                # Convert mask_image from numpy array (from Gradio) to PIL Image (expected by pipeline)
                # Gradio's sketch tool returns an RGBA array. We need a single channel mask.
                mask_pil = Image.fromarray(mask_image[:,:,0], mode='L') # Take red channel as mask intensity

                # The pipeline expects List[Tuple[Image.Image, str]]
                regions_input = [(mask_pil, main_prompt)]

                print(f"Generating single region for prompt: '{main_prompt}', neg_prompt: '{negative_prompt}', "
                      f"mask size: {mask_pil.size}, guidance: {guidance_scale}, "
                      f"steps: {num_inference_steps}, seed: {seed}")

                result = region_based_pipeline(
                    regions=regions_input, # Pass the list of (mask, prompt)
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                ).images[0]
                return result, f"Region-based image generated successfully! Seed: {seed}"


            region_generate_btn.click(
                fn=run_single_region_inference,
                inputs=[
                    region_main_prompt,
                    region_mask_image,
                    region_negative_prompt,
                    region_guidance_scale,
                    region_num_inference_steps,
                    region_seed
                ],
                outputs=[region_output, region_status]
            )

        # --- Illusion Diffusion Tab ---
        with gr.TabItem("Illusion Diffusion"):
            with gr.Row():
                with gr.Column():
                    illusion_prompt = gr.Textbox(label="Positive Prompt", placeholder="e.g., a futuristic city")
                    illusion_control_image = gr.Image(label="Upload Control Image (e.g., a pattern or outline)", type="pil")
                    illusion_negative_prompt = gr.Textbox(label="Negative Prompt (Optional)", placeholder="e.g., noise, distorted")
                    with gr.Row():
                        illusion_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=7.5, label="Guidance Scale")
                        illusion_num_inference_steps = gr.Slider(minimum=10, maximum=100, step=5, value=50, label="Inference Steps")
                    with gr.Row():
                        illusion_ddim_steps = gr.Slider(minimum=10, maximum=100, step=5, value=50, label="DDIM Steps (for DDIM scheduler)", info="Often same as Inference Steps")
                        illusion_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Strength of Control Image", info="How much the control image influences the generation.")
                    illusion_seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    illusion_generate_btn = gr.Button("Generate Illusion", variant="primary")
                with gr.Column():
                    illusion_output = gr.Image(label="Generated Illusion", type="pil")
                    illusion_status = gr.Textbox(label="Status", interactive=False)
            
            illusion_generate_btn.click(
                fn=run_illusion_inference,
                inputs=[
                    illusion_prompt,
                    illusion_control_image,
                    illusion_negative_prompt,
                    illusion_guidance_scale,
                    illusion_num_inference_steps,
                    illusion_seed,
                    illusion_ddim_steps,
                    illusion_strength
                ],
                outputs=[illusion_output, illusion_status]
            )

demo.launch(share=False) # Set share=True to get a public link
