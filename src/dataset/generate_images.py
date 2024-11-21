import os
import random
import sqlite3
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

# generation params

NUM_IMAGES = 2  # Number of images to generate
CANVAS_SIZE = (1024, 1024)
MIN_FONT_SIZE = 30
MAX_FONT_SIZE = 100
MAX_CHARACTERS = 20  # Maximum characters per image

# files

FONTS_DIR = "fonts"
OUTPUT_DIR = "output"
CHARACTER_SUPPORT_DB = "chardb/chardb.sqlite"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# load font support

conn = sqlite3.connect(CHARACTER_SUPPORT_DB)
cursor = conn.cursor()


# character extraction

def extract_character():
    while True:
        # Generate a random character from the CJK Unified Ideographs range
        char = chr(random.randint(0x4E00, 0x9FFF))

        # Get the list of fonts that support this character
        supported_fonts = font_support_map.get(char, [])
        if supported_fonts:
            radical = get_character_radical(char)
            return char, radical, random.choice(supported_fonts)


def augment_image(image):
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))

    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))

    if random.random() > 0.5:
        np_image = np.array(image)
        noise = np.random.randint(0, 10, (np_image.shape[0], np_image.shape[1], 1), dtype='uint8')
        np_image = np.clip(np_image + noise, 0, 255)
        image = Image.fromarray(np_image.astype('uint8'))
    return image

# character arrangment

def check_overlap(bbox, bboxes):
    for existing_bbox in bboxes:
        if (bbox[0] < existing_bbox[2] and bbox[2] > existing_bbox[0] and
                bbox[1] < existing_bbox[3] and bbox[3] > existing_bbox[1]):
            return True
    return False


def place_characters(canvas, draw, num_chars, char_images, bboxes, annotations):
    """
    Places characters on the canvas in priority:
    1. Vertical arrangements
    2. Square arrangements
    3. Random single placements
    """
    def check_group_overlap(group_bboxes):
        """Check if a group of bounding boxes overlaps with any existing boxes."""
        for bbox in group_bboxes:
            if check_overlap(bbox, bboxes):
                return True
        return False

    def place_vertical_arrangement():
        """Places a vertical arrangement of characters."""
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            num_vertical = random.randint(2, min(6, len(char_images)))  # Random vertical group size
            group_bboxes = []
            group_annotations = []
            total_height = sum(char_images[i].height for i in range(num_vertical))
            max_width = max(char_images[i].width for i in range(num_vertical))
            start_x = random.randint(0, CANVAS_SIZE[0] - max_width)
            start_y = random.randint(0, CANVAS_SIZE[1] - total_height)
            y = start_y

            for i in range(num_vertical):
                char_image = char_images[i]
                bbox = (start_x, y, start_x + char_image.width, y + char_image.height)
                group_bboxes.append(bbox)
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                group_annotations.append(
                    f"{char_image.char} {center_x} {center_y} {char_image.width} {char_image.height}"
                )
                y += char_image.height

            if not check_group_overlap(group_bboxes):
                for bbox, annotation, char_image in zip(group_bboxes, group_annotations, char_images[:num_vertical]):
                    bboxes.append(bbox)
                    annotations.append(annotation)
                    canvas.paste(char_image.image, (bbox[0], bbox[1]), char_image.image)
                del char_images[:num_vertical]
                placed = True
            attempts += 1

    def place_square_arrangement():
        """Places characters in a square arrangement."""
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            if len(char_images) < 4:
                return False  # Need at least 4 characters for a square
            square_size = char_images[0].width  # Assume equal sizes
            group_bboxes = []
            group_annotations = []

            # Choose top-left corner for the square
            start_x = random.randint(0, CANVAS_SIZE[0] - 2 * square_size)
            start_y = random.randint(0, CANVAS_SIZE[1] - 2 * square_size)

            # Calculate bboxes for the square
            for dx, dy in [(0, 0), (square_size, 0), (0, square_size), (square_size, square_size)]:
                x = start_x + dx
                y = start_y + dy
                bbox = (x, y, x + square_size, y + square_size)
                group_bboxes.append(bbox)
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                char_image = char_images.pop(0)
                group_annotations.append(
                    f"{char_image.char} {center_x} {center_y} {square_size} {square_size}"
                )

            if not check_group_overlap(group_bboxes):
                for bbox, annotation, char_image in zip(group_bboxes, group_annotations, char_images[:4]):
                    bboxes.append(bbox)
                    annotations.append(annotation)
                    canvas.paste(char_image.image, (bbox[0], bbox[1]), char_image.image)
                placed = True
            attempts += 1

    def place_random():
        """Places a single character randomly."""
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            char_image = char_images.pop(0)
            x = random.randint(0, CANVAS_SIZE[0] - char_image.width)
            y = random.randint(0, CANVAS_SIZE[1] - char_image.height)
            bbox = (x, y, x + char_image.width, y + char_image.height)

            if not check_overlap(bbox, bboxes):
                bboxes.append(bbox)
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                annotations.append(
                    f"{char_image.char} {center_x} {center_y} {char_image.width} {char_image.height}"
                )
                canvas.paste(char_image.image, (x, y), char_image.image)
                placed = True
            attempts += 1

    # Prioritize placements
    char_images.sort(key=lambda c: c.height, reverse=True)  # Place larger characters first
    while char_images:
        if random.random() < 0.4 and len(char_images) >= 2:  # 40% chance for vertical arrangement
            place_vertical_arrangement()
        elif random.random() < 0.3 and len(char_images) >= 4:  # 30% chance for square arrangement
            place_square_arrangement()
        else:  # Fallback to random placement
            place_random()

def generate_image(index):
    canvas = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    num_chars = random.randint(1, MAX_CHARACTERS)
    bboxes = []
    annotations = []

    for _ in range(num_chars):
        char, radical, font_filename = extract_character()
        font_path = os.path.join(FONTS_DIR, font_filename)
        font_size = random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)
        font = ImageFont.truetype(font_path, font_size)

        char_image = Image.new("RGBA", (font_size * 2, font_size * 2), color=0)
        char_draw = ImageDraw.Draw(char_image)
        char_draw.text((0, 0), char, font=font, fill="black")
        char_image = char_image.crop(char_image.getbbox())

        #char_image = augment_image(char_image)

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
                annotations.append(
                    f"{char} {center_x} {center_y} {char_image.width} {char_image.height} {radical} {font_filename}")
                canvas.paste(char_image, (x, y), char_image)
            attempts += 1

    image_path = os.path.join(OUTPUT_DIR, f"{index}.png")
    canvas.save(image_path)

    annotation_path = os.path.join(OUTPUT_DIR, f"{index}.txt")
    with open(annotation_path, "w", encoding="utf-8") as f:
        f.write("\n".join(annotations))


for i in range(NUM_IMAGES):
    generate_image(i)
    print(f"Generated image {i + 1}/{NUM_IMAGES}")

conn.close()