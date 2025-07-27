import torch
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import cv2


class MultiDiffusionIllusionDiffusionPipeline:
    """
    A MultiDiffusion pipeline for Illusion Diffusion.
    It takes a base Stable Diffusion pipeline and combines it with a control image
    to generate images that incorporate the control image's structure/pattern.
    """

    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe = pipe
        self.scheduler = pipe.scheduler  # Use the scheduler from the base pipeline

    @torch.no_grad()
    def __call__(
            self,
            prompt: str,
            control_image: Image.Image,
            negative_prompt: str = None,  # Added negative prompt
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,  # Added guidance scale
            generator: torch.Generator = None,  # Added generator for seed control
            ddim_steps: int = 50,  # Specific to DDIM if used for inversion/forward process
            strength: float = 0.5  # Controls how much the control image influences
    ):
        device = self.pipe.device
        dtype = self.pipe.unet.dtype

        # 1. Prepare control image
        # Resize control image to match target generation size (e.g., 512x512)
        # Assuming target size is 512x512 for standard SD v1.5
        image_size = (512, 512)
        if control_image.size != image_size:
            control_image = control_image.resize(image_size, Image.LANCZOS)

        # Convert control image to a tensor and normalize
        # Assuming grayscale or average for simple control, or more complex pre-processing
        control_tensor = torch.from_numpy(np.array(control_image).astype(np.float32) / 255.0).to(device).unsqueeze(0)
        if control_tensor.shape[3] == 3:  # If RGB, convert to grayscale or take one channel
            control_tensor = control_tensor.mean(dim=-1, keepdim=True)  # Simple grayscale

        # In a real illusion diffusion, `control_image` would be used to guide the noise prediction.
        # This often involves adding a "control" signal to the UNet's input or features.
        # The original `illusion.py` uses `self.encode_img` and `self.ddim_inversion`.
        # This implies it's using the control image to get an initial latent representation,
        # and then modifying the diffusion process.

        # The paper's `illusion.py` is quite specific, using DDIM inversion.
        # Let's align with its original intent for `control_image` usage.
        # `strength` here is crucial for how much the inverted control image impacts.

        # The original illusion.py has `_encode_img` and `ddim_inversion`
        # and then modifies latents directly.
        # Let's ensure the `strength` parameter is used to blend the inverted latent.

        # --- Re-using the structure from the original illusion.py with added parameters ---
        # The original code encodes the image and then adds noise to it based on the scheduler steps
        # This is where strength usually applies - by starting at a "noisier" version of the image.

        # 1. Encode prompts
        text_input = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(device))[0]

        if negative_prompt is None:
            uncond_embeddings = torch.zeros_like(text_embeddings)
        else:
            uncond_input = self.pipe.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(device))[0]

        text_embeddings_cond_uncond = torch.cat([uncond_embeddings, text_embeddings])

        # 2. Prepare latents based on control_image and strength
        # This is the core "illusion" part. The control_image is first encoded,
        # then possibly noised to a certain degree based on `strength`.

        # Encode the control image
        # control_image needs to be 3-channel for VAE encoder
        if control_image.mode != 'RGB':
            control_image = control_image.convert('RGB')

        control_image_tensor = self.pipe.image_processor.preprocess(control_image).to(device=device, dtype=dtype)
        # Using pipe's VAE to encode the control image
        control_latent = self.pipe.vae.encode(control_image_tensor).latent_dist.sample()
        control_latent = control_latent * self.pipe.vae.config.scaling_factor

        # Determine the starting noise level based on strength
        # The number of steps the control image is noised to
        init_timestep = int(num_inference_steps * strength)
        init_timestep = min(init_timestep, num_inference_steps - 1)  # Ensure valid timestep

        # Get initial latents by adding noise to the control latent
        noise = torch.randn(control_latent.shape, generator=generator, device=device, dtype=dtype)
        # Scheduler provides a `sample` at the specified timestep `t`
        # This is the `latents` that diffusion starts from.

        # Using the scheduler's `add_noise` to get `z_t` from `z_0`
        timesteps_to_noise = self.scheduler.timesteps[init_timestep]  # Get the specific t for init_timestep
        latents = self.scheduler.add_noise(control_latent, noise, timesteps_to_noise)

        # Set timesteps for the denoising loop
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps[init_timestep:]  # Start from init_timestep

        # 3. Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Generating illusion")):
            # Standard classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings_cond_uncond,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 4. Decode latents to image
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        image = self.pipe.vae.decode(latents.cpu()).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = self.pipe.numpy_to_pil(image)
        return {"images": image}

