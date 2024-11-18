import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

OUTPUT_DIR = "output"
FONT_DIR = "fonts"
CANVAS_SIZE = (1024, 1024)
MIN_FONT_SIZE = 30
MAX_FONT_SIZE = 100
NUM_IMAGES = 2  # Number of images to generate
MAX_CHARACTERS = 20  # Maximum characters per image
CHARACTERS = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"  # Example Seal Script characters

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load available fonts
fonts = [os.path.join(FONT_DIR, font) for font in os.listdir(FONT_DIR) if font.endswith(".ttf") or font.endswith(".otf")]
if not fonts:
    raise ValueError("No fonts found in the FONT_DIR folder.")

# Function to apply random data augmentation
def augment_image(image):
    # Add tonality variation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))
    # Add Gaussian blur
    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))
    # Add random noise
    if random.random() > 0.5:
        np_image = np.array(image)
        noise = np.random.randint(0, 50, (np_image.shape[0], np_image.shape[1], 1), dtype='uint8')
        np_image = np.clip(np_image + noise, 0, 255)
        image = Image.fromarray(np_image.astype('uint8'))
    return image

# Function to check overlap of bounding boxes
def check_overlap(bbox, bboxes):
    for existing_bbox in bboxes:
        if (bbox[0] < existing_bbox[2] and bbox[2] > existing_bbox[0] and
                bbox[1] < existing_bbox[3] and bbox[3] > existing_bbox[1]):
            return True
    return False

# Function to generate random image with characters
def generate_image(index):
    canvas = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    num_chars = random.randint(1, MAX_CHARACTERS)
    bboxes = []
    annotations = []

    for _ in range(num_chars):
        char = random.choice(CHARACTERS)
        font_path = random.choice(fonts)
        font_size = random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)
        font = ImageFont.truetype(font_path, font_size)

        # Render character
        char_image = Image.new("RGBA", (font_size * 2, font_size * 2), color=0)
        char_draw = ImageDraw.Draw(char_image)
        char_draw.text((font_size // 2, font_size // 2), char, font=font, fill="black")
        char_image = char_image.crop(char_image.getbbox())

        # Apply augmentation
        #char_image = augment_image(char_image)

        # Determine position
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            x = random.randint(0, CANVAS_SIZE[0] - char_image.width)
            y = random.randint(0, CANVAS_SIZE[1] - char_image.height)
            bbox = (x, y, x + char_image.width, y + char_image.height)
            if not check_overlap(bbox, bboxes):
                placed = True
                bboxes.append(bbox)
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                annotations.append(f"{char} {center_x} {center_y} {char_image.width} {char_image.height}")
                canvas.paste(char_image, (x, y), char_image)
            attempts += 1

    # Save image
    image_path = os.path.join(OUTPUT_DIR, f"image_{index}.png")
    canvas.save(image_path)

    # Save annotations
    annotation_path = os.path.join(OUTPUT_DIR, f"{index}.txt")
    with open(annotation_path, "w", encoding="utf-8") as f:
        f.write("\n".join(annotations))

# Generate dataset
for i in range(NUM_IMAGES):
    generate_image(i)
    print(f"Generated image {i + 1}/{NUM_IMAGES}")
