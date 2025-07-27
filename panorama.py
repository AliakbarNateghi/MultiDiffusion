import torch
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np


class MultiDiffusionPanoramaPipeline:
    """
    A MultiDiffusion pipeline for generating panoramic images.
    It takes a base Stable Diffusion pipeline and applies the MultiDiffusion
    principle to create seamless panoramic outputs.
    """

    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe = pipe
        self.scheduler = pipe.scheduler  # Use the scheduler from the base pipeline

    @torch.no_grad()
    def __call__(
            self,
            prompt: str,
            negative_prompt: str = None,  # Added negative prompt
            width: int = 1024,
            height: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,  # Added guidance scale
            generator: torch.Generator = None,  # Added generator for seed control
            overlap: int = 64,  # Overlap in pixels, 64 is a common default
            views_per_step: int = 4  # Number of views processed in parallel per step
    ):
        device = self.pipe.device
        dtype = self.pipe.unet.dtype

        # 1. Encode prompts
        text_input = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(device))[0]

        # Handle negative prompt embeddings
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

        # Concatenate positive and negative embeddings for classifier-free guidance
        text_embeddings_cond_uncond = torch.cat([uncond_embeddings, text_embeddings])

        # 2. Prepare latents
        # Latent dimensions are usually 1/8th of image dimensions
        latent_width = width // self.pipe.vae_scale_factor
        latent_height = height // self.pipe.vae_scale_factor
        latents_shape = (1, self.pipe.unet.config.in_channels, latent_height, latent_width)

        # Initialize latents with noise
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Calculate crop dimensions and positions for the MultiDiffusion approach
        # Assume square crops for simplicity, as per the paper's base model assumption
        crop_size = 512 // self.pipe.vae_scale_factor  # Latent crop size for Stable Diffusion v1.5

        # Generate overlapping views (crops)
        # Define a list of (start_x, start_y, end_x, end_y) for each crop in latent space
        # This will cover the entire panoramic latent space
        views = []
        # Calculate horizontal stride to achieve desired overlap for a 512x512 crop
        # Stride = Crop_size - Overlap
        stride_x = crop_size - (overlap // self.pipe.vae_scale_factor)

        for y in range(0, latent_height,
                       stride_x):  # Use stride_x for y too for square patches, or define a new stride_y
            for x in range(0, latent_width, stride_x):
                # Ensure the crop doesn't go out of bounds
                end_x = min(x + crop_size, latent_width)
                end_y = min(y + crop_size, latent_height)
                start_x = end_x - crop_size
                start_y = end_y - crop_size
                if start_x < 0: start_x = 0
                if start_y < 0: start_y = 0
                views.append((start_x, start_y, end_x, end_y))

        # Remove duplicate views and sort (optional, but good for consistency)
        views = sorted(list(set(views)))

        # Define weight map for fusing patches (e.g., using a Gaussian or constant overlap)
        # This is the W_i in the paper's equation 4
        # For simplicity, we can use a constant overlap weight or a blended mask.
        # A common technique is to use cosine weighting for seamless blending.
        # For now, let's assume simple averaging in overlapping regions after individual denoising.
        # The paper's Equation 5 handles this implicitly for "direct pixel samples" which implies averaging.

        # Create a blank array for the combined denoised latents
        combined_denoised_latents = torch.zeros_like(latents)
        # Create a blank array for weights (to handle overlaps)
        weights_map = torch.zeros_like(latents)

        # Initialize weights map based on overlaps (for proper weighted average)
        # For each view, create a local weight mask and sum it up
        # This acts as the denominator in Eq. 5
        local_weight_mask = torch.ones((1, self.pipe.unet.config.in_channels, crop_size, crop_size), device=device,
                                       dtype=dtype)
        for vx1, vy1, vx2, vy2 in views:
            weights_map[:, :, vy1:vy2, vx1:vx2] += local_weight_mask

        # Ensure no division by zero in non-covered regions
        weights_map[weights_map == 0] = 1e-6  # Small epsilon to prevent div by zero

        # 3. Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Generating panorama")):
            latents_input = latents

            # This is where the MultiDiffusion magic happens:
            # We collect denoised predictions from multiple overlapping views.
            # Then we combine them.

            # Store predictions from each view
            denoised_views_sum = torch.zeros_like(latents)

            for j in range(0, len(views), views_per_step):
                batch_views = views[j: j + views_per_step]

                batch_latents_crops = []
                for vx1, vy1, vx2, vy2 in batch_views:
                    # F_i(J_t) operation: Take a crop from current latents
                    crop = latents_input[:, :, vy1:vy2, vx1:vx2]
                    # Pad if crop size is smaller than expected (e.g., at edges)
                    pad_x = crop_size - crop.shape[3]
                    pad_y = crop_size - crop.shape[2]
                    if pad_x > 0 or pad_y > 0:
                        crop = torch.nn.functional.pad(crop, (0, pad_x, 0, pad_y), "constant", 0)
                    batch_latents_crops.append(crop)

                if not batch_latents_crops:
                    continue

                batch_latents_crops = torch.cat(batch_latents_crops, dim=0)  # Batch for UNet

                # Predict noise for each crop (Phi(I_t^i | y_i) operation conceptually)
                # Apply classifier-free guidance
                latent_model_input = torch.cat([batch_latents_crops] * 2) if guidance_scale > 1 else batch_latents_crops
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Uncond and cond embeddings for the batch
                batch_text_embeddings_cond_uncond = text_embeddings_cond_uncond.repeat(len(batch_views), 1, 1)

                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=batch_text_embeddings_cond_uncond,
                    return_dict=False,
                )[0]

                # Perform guidance
                if guidance_scale > 1:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Denoise each crop individually (part of Phi(I_t^i | y_i) -> I_t-1^i)
                # The scheduler step implicitly gives the denoised latents
                denoised_crops = self.scheduler.step(noise_pred, t, batch_latents_crops).pred_original_sample

                # F_i^-1 operation: Place denoised crops back into the full panorama latent map
                for k, (vx1, vy1, vx2, vy2) in enumerate(batch_views):
                    denoised_crop_k = denoised_crops[k]
                    # Remove padding if applied
                    denoised_crop_k = denoised_crop_k[:, :vy2 - vy1, :vx2 - vx1]
                    denoised_views_sum[:, :, vy1:vy2, vx1:vx2] += denoised_crop_k

            # Average overlapping regions by dividing by the sum of weights (Equation 5 numerator)
            fused_latents = denoised_views_sum / weights_map

            # Now, we need to calculate the *actual* noise prediction for the full latent
            # that would result in `fused_latents`
            # This is a bit tricky, as the paper suggests solving for J_{t-1} directly.
            # The simplified closed-form in Eq. 5 is for `Psi(J_t|z)` returning the *denoised* J_{t-1}.
            # We need to adapt the standard diffusion step to this fused J_{t-1}.

            # The standard diffusion step (x_t -> x_{t-1}) involves pred_original_sample (x_0_pred)
            # and then applying scheduler equations.
            # So, `fused_latents` here *is* our `pred_original_sample` for the entire image.

            # Compute the residual noise based on this fused prediction
            noise_pred_for_full_latent = (latents - fused_latents) / self.scheduler.sigmas[i]

            # Update latents for the next step using the scheduler
            latents = self.scheduler.step(noise_pred_for_full_latent, t, latents).prev_sample

        # 4. Decode latents to image
        # scale and decode the image latents with vae
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        image = self.pipe.vae.decode(latents.cpu()).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = self.pipe.numpy_to_pil(image)
        return {"images": image}

