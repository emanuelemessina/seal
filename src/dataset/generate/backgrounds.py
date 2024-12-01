import os
import random
from PIL import Image, ImageOps

BACKGROUNDS_DIR = "../backgrounds"


def extract_random():
    """Choose a random image from BACKGROUNDS and return it"""
    background_files = [f for f in os.listdir(BACKGROUNDS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not background_files:
        raise FileNotFoundError(f"No suitable background images found in {BACKGROUNDS_DIR}")

    chosen_background = random.choice(background_files)
    background_path = os.path.join(BACKGROUNDS_DIR, chosen_background)
    background_image = Image.open(background_path).convert("RGB")
    # images in backgrounds are already canvas sized
    # return ImageOps.fit(background_image, canvas_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return background_image
