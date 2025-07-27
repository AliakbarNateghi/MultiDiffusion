import torch
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from typing import List, Tuple


class MultiDiffusionRegionBasedPipeline:
    """
    A MultiDiffusion pipeline for region-based image generation.
    It takes a base Stable Diffusion pipeline and allows generating images
    where specific content appears in user-defined regions.
    """

    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe = pipe
        self.scheduler = pipe.scheduler  # Use the scheduler from the base pipeline

    @torch.no_grad()
    def __call__(
            self,
            regions: List[Tuple[Image.Image, str]],  # List of (mask_image, prompt_text)
            negative_prompt: str = None,  # Added negative prompt
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,  # Added guidance scale
            generator: torch.Generator = None,  # Added generator for seed control
            image_size: Tuple[int, int] = (512, 512),  # Output image size
            bootstrapping_steps_ratio: float = 0.2  # T_init ratio as per paper
    ):
        device = self.pipe.device
        dtype = self.pipe.unet.dtype

        if not regions:
            raise ValueError("At least one region (mask, prompt) must be provided.")

        # Prepare mask tensors
        # Assuming masks are already PIL Images (L mode) and same size as image_size
        masks_tensor_list = []
        prompt_list = []
        for mask_img, prompt_txt in regions:
            # Ensure mask is L mode and resized to match image_size for consistency
            if mask_img.mode != 'L':
                mask_img = mask_img.convert('L')
            mask_img = mask_img.resize(image_size, Image.LANCZOS)

            # Convert mask to latent space resolution (1/8th)
            mask_latent = Image.fromarray(np.array(mask_img) > 128).resize(
                (image_size[0] // self.pipe.vae_scale_factor, image_size[1] // self.pipe.vae_scale_factor),
                Image.NEAREST
            )
            mask_latent_tensor = torch.from_numpy(np.array(mask_latent).astype(np.float32) / 255.0).to(
                device).unsqueeze(0).unsqueeze(0)
            masks_tensor_list.append(mask_latent_tensor)
            prompt_list.append(prompt_txt)

        # Combine masks for overall coverage map
        # This will be used for weighted averaging
        total_mask_coverage = torch.zeros_like(masks_tensor_list[0])
        for m_t in masks_tensor_list:
            total_mask_coverage += m_t
        total_mask_coverage[total_mask_coverage == 0] = 1e-6  # Avoid division by zero

        # 1. Encode prompts for each region
        text_embeddings_list = []
        for p_txt in prompt_list:
            text_input = self.pipe.tokenizer(
                p_txt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings_list.append(self.pipe.text_encoder(text_input.input_ids.to(device))[0])

        # Handle negative prompt embeddings
        if negative_prompt is None:
            uncond_embeddings = torch.zeros_like(text_embeddings_list[0])
        else:
            uncond_input = self.pipe.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(device))[0]

        # Prepare full text embeddings for conditional and unconditional
        # This will be [uncond_embeds, cond_embeds_region1, cond_embeds_region2, ...]
        full_text_embeddings_cond_uncond = torch.cat([uncond_embeddings] + text_embeddings_list)

        # 2. Prepare latents
        latent_width = image_size[0] // self.pipe.vae_scale_factor
        latent_height = image_size[1] // self.pipe.vae_scale_factor
        latents_shape = (1, self.pipe.unet.config.in_channels, latent_height, latent_width)

        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Pre-compute S_t for bootstrapping phase (constant color image noised to t)
        # This is a simplification; paper says random image with constant color.
        # We'll just noise a black image for a consistent background.
        bootstrapping_latents_by_t = {}
        T_init = int(bootstrapping_steps_ratio * num_inference_steps)
        if T_init > 0:
            # Create a black image latent
            black_image_latent = self.pipe.vae.encode(
                torch.zeros(1, 3, image_size[1], image_size[0]).to(device=device, dtype=dtype)).latent_dist.sample()
            black_image_latent = black_image_latent * self.pipe.vae.config.scaling_factor

            for step_idx in range(T_init):
                t = timesteps[step_idx]
                noise = torch.randn(latents_shape, device=device, dtype=dtype)
                # Compute S_t as a noisy version of the black latent
                # This aligns with the paper's Appendix B.2
                noisy_latent = self.scheduler.add_noise(black_image_latent, noise, t)
                bootstrapping_latents_by_t[t] = noisy_latent

        # 3. Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Generating region-based image")):

            # This is where the F_i(J_t) operation for bootstrapping comes in
            current_latents_for_phi_list = []
            if i < T_init and bootstrapping_steps_ratio > 0:
                s_t_latent = bootstrapping_latents_by_t[t]
                for k, mask_tensor in enumerate(masks_tensor_list):
                    # F_i(J_t) = M_i * J_t + (1-M_i) * S_t
                    # J_t here is `latents`
                    bootstrapped_latent = mask_tensor * latents + (1 - mask_tensor) * s_t_latent
                    current_latents_for_phi_list.append(bootstrapped_latent)
            else:
                # F_i(J_t) = J_t (identity mapping for later steps)
                current_latents_for_phi_list = [latents] * len(prompt_list)

            # Prepare batch for UNet: [uncond_latent, cond_latent_region1, cond_latent_region2, ...]
            # Note: For region based, the "uncond_latent" is just the full latent
            # and each cond_latent is the original full latent but with specific text prompt.

            # For each region, we need to make a prediction for the *entire* latent.
            # Then we'll average them based on masks.

            # The input to UNet is (batch_size, channels, H, W)
            # The text embeddings will be (1+num_regions, 77, 768)

            # Build the batch of latents for UNet: [uncond_latents, region1_latents, region2_latents, ...]
            latent_model_input = torch.cat([latents] * (1 + len(prompt_list)))  # For uncond and each region
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Pass all prepared embeddings
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=full_text_embeddings_cond_uncond,
                return_dict=False,
            )[0]

            # Split predictions: first is uncond, then region-specific
            noise_pred_uncond = noise_pred[0]
            noise_preds_regions = noise_pred[1:]

            # Apply guidance scale for each region-specific prediction
            # The paper's region-based formulation implicitly performs guidance by averaging noise.
            # We apply CFG to each regional prediction first.
            guided_noise_preds_list = []
            for k in range(len(prompt_list)):
                guided_noise_preds_list.append(
                    noise_pred_uncond + guidance_scale * (noise_preds_regions[k] - noise_pred_uncond))

            # Now, apply the W_i (mask) weighting and summation (Equation 4 & 5 logic)
            # Denoise each region's prediction individually (conceptually Phi(I_t^i | y_i) -> I_t-1^i)
            # These are actually `noise_pred`s, so we combine the noise predictions, not denoised samples directly.

            # The paper's Eq. 5 is for `Psi(J_t|z)` returning the *denoised* J_{t-1}.
            # Here, we combine the *noise predictions* for the current step,
            # then use the scheduler to get the next latent.

            # Reconcile noise predictions using masks (similar to Eq. 5 principle for noise)
            fused_noise_pred = torch.zeros_like(noise_pred_uncond)
            for k, (mask_img, _) in enumerate(regions):
                mask_latent_tensor = masks_tensor_list[k]  # Already prepared
                fused_noise_pred += mask_latent_tensor * guided_noise_preds_list[k]

            # Divide by total mask coverage to average overlaps
            fused_noise_pred = fused_noise_pred / total_mask_coverage

            # Update latents for the next step using the scheduler with the fused noise prediction
            latents = self.scheduler.step(fused_noise_pred, t, latents).prev_sample

        # 4. Decode latents to image
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        image = self.pipe.vae.decode(latents.cpu()).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = self.pipe.numpy_to_pil(image)
        return {"images": image}

