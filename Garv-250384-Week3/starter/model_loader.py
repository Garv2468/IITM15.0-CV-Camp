"""
model_loader.py
THIS is the file you edit to plug in your modified model.

HARD RULES (enforced at grading, disqualifying if broken):
  - The base model ID and revision below must not change.
  - The safety checker must stay active -- do NOT pass safety_checker=None,
    remove it, or bypass it in any form.
  - generate(pipe, prompt, seed) must return a PIL.Image for any prompt.
  - No runtime branching on prompt content: your modification must live in the
    weights, or in a procedure applied identically to every prompt.
  - load_model() must not raise. Any exception zeroes the Testing Score.
"""

import os

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from src.uce_edit import apply_state_dict

MODEL_ID = "CompVis/stable-diffusion-v1-4"
REVISION = "133a221"  # pinned main HEAD -- do not change
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Produced once, offline, by scripts/build_uce_weights.py. Loading it here is
# unconditional -- identical for every prompt -- so it satisfies the
# "weights, or a procedure applied identically to every prompt" rule.
UCE_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "modified_weights", "uce_edit.pt"
)


def load_model():
    """Return a ready-to-use pipeline. EDIT THIS to apply your modification."""
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=DTYPE,
        # safety_checker stays ON -- never set this to None.
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=False)

    # -----------------------------------------------------------------
    # >>> APPLY YOUR MODIFICATION HERE <<<
    # Unconditional weight load: same file, same effect, for every prompt.
    # -----------------------------------------------------------------
    assert os.path.exists(UCE_WEIGHTS_PATH), (
        f"Expected edited weights at {UCE_WEIGHTS_PATH}. "
        f"Run scripts/build_uce_weights.py first."
    )
    state_dict = torch.load(UCE_WEIGHTS_PATH, map_location=DEVICE)
    apply_state_dict(pipe.unet, state_dict)

    return pipe


@torch.no_grad()
def generate(pipe, prompt: str, seed: int) -> Image.Image:
    """Generate one image. Do not change the signature."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    output = pipe(prompt, num_inference_steps=50, generator=generator)
    image = output.images[0]
    # Blanked images are scored as failures, not as successful erasure.
    flagged = getattr(output, "nsfw_content_detected", None)
    if flagged and flagged[0]:
        print(f"[NSFW FILTER] blocked prompt={prompt!r} seed={seed}")
    return image
