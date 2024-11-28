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

def extract_random_font():
    # Check if the fonts table is empty
    cursor.execute("SELECT COUNT(*) FROM fonts")
    fonts_count = cursor.fetchone()[0]
    if fonts_count == 0:
        conn.close()
        raise AssertionError("Empty fonts table")
    random_offset = random.randint(0, fonts_count - 1)
    query = """
        SELECT id, filename 
        FROM fonts
        LIMIT 1 OFFSET ?
    """
    cursor.execute(query, (random_offset,))
    row = cursor.fetchone()
    return {"id": row[0], "filename": row[1]}


def extract_character(font=None):
    """
       Extract a random character. Optionally filter by a specific font.

       Args:
           font (dict, optional): A dictionary containing the font's 'id' and 'filename'.
                                  If provided, only characters associated with this font
                                  will be considered.

       Returns:
           tuple: (character, radical, font, query_character)
       """
    if font:
        cursor.execute(
            "SELECT COUNT(*) FROM font_support WHERE font_id = ?",
            (font["id"],)
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM font_support")

    count = cursor.fetchone()[0]
    if count == 0:
        conn.close()
        raise AssertionError("No entries found in font_support table for the given criteria")

    # Get a random offset
    random_offset = random.randint(0, count - 1)

    query = """
            SELECT 
                fs.id, 
                c.character, 
                f.filename AS font, 
                fs.query_character, 
                c.radical
            FROM font_support fs
            INNER JOIN characters c ON fs.character_id = c.id
            INNER JOIN fonts f ON fs.font_id = f.id
        """

    parameters = []

    if font:
        query += """
        WHERE fs.font_id = ?
        """
        parameters = [font["id"]]

    query += """
        LIMIT 1 OFFSET ?
    """

    parameters.append(random_offset)

    cursor.execute(query, parameters)

    row = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    row_dict = dict(zip(columns, row))
    return row_dict["character"], row_dict["radical"], row_dict["font"], row_dict["query_character"]


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


# generate char image

def generate_random_char_image(font_size, font=None):
    char, radical, font_filename, query = extract_character(font)
    font_path = os.path.join(FONTS_DIR, font_filename)
    font = ImageFont.truetype(font_path, font_size)

    canvas_size = font_size * 3  # initial blank canvas larger than necessary
    char_image = Image.new("RGBA", (canvas_size, canvas_size), color=0)
    char_draw = ImageDraw.Draw(char_image)

    # Calculate text dimensions
    text_bbox = char_draw.textbbox((0, 0), query, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Center the text within the canvas
    x_offset = (canvas_size - text_width) // 2 - text_bbox[0]
    y_offset = (canvas_size - text_height) // 2 - text_bbox[1]

    char_draw.text((x_offset, y_offset), query, font=font, fill="black")

    char_image = char_image.crop(char_image.getbbox())
    return char_image, char, radical, font_filename


# character arrangment

def check_overlap(bbox, bboxes):
    for existing_bbox in bboxes:
        if (bbox[0] < existing_bbox[2] and bbox[2] > existing_bbox[0] and
                bbox[1] < existing_bbox[3] and bbox[3] > existing_bbox[1]):
            return True
    return False


def check_group_overlap(group_bboxes, bboxes):
    """Check if a group of bounding boxes overlaps with any existing boxes."""
    for bbox in group_bboxes:
        if check_overlap(bbox, bboxes):
            return True
    return False


# TODO: INFLATE BB A LITTLE BIT, UNIFY SPACING AND MARGIN FOR ALL PLACEMENTS, TIDY UP CODE, ADD BACKGROUND IMAGES, VISUALIZE AND STABILIZE CHAR PROBABILITY DISTRIBUTION


def place_vertical_arrangement(canvas, bboxes, annotations, spacing=0.2, margin=10):
    """Places a vertical arrangement of a randum number of characters. Returns the placed number"""
    # generate a random num of chars of equal font and size
    font_size = random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)
    font = extract_random_font()
    generated = [generate_random_char_image(font_size, font)]

    char_height = generated[0][0].height
    char_width = generated[0][0].width

    spacing = int(np.ceil(spacing * char_width))

    max_vertical_num = int(np.floor((CANVAS_SIZE[1]-2*margin)/(char_height+spacing)))
    vertical_num = random.randint(2, max_vertical_num)

    generated.extend([generate_random_char_image(font_size, font) for _ in range(vertical_num-1)])

    attempts = 0
    while attempts < 100:

        group_bboxes = []
        group_annotations = []
        char_images = []

        # arrangement anchor
        start_x = random.randint(margin, CANVAS_SIZE[0] - char_width - margin)
        start_y = random.randint(margin, CANVAS_SIZE[1] - vertical_num * (char_height+spacing) - margin)

        # character relative anchors
        for dy in [(char_height + spacing)*n for n in range(vertical_num)]:
            # absolute anchors
            x = start_x
            y = start_y + dy

            bbox = (x, y, x + char_width, y + char_height)
            group_bboxes.append(bbox)

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            char_image, char, radical, font_filename = generated.pop(0)

            char_images.append(char_image)

            group_annotations.append(
                f"{char} {center_x} {center_y} {char_width} {char_height} {radical} {font_filename}"
            )

        if not check_group_overlap(group_bboxes, bboxes):
            for bbox, annotation, char_image in zip(group_bboxes, group_annotations, char_images):
                bboxes.append(bbox)
                annotations.append(annotation)
                canvas.paste(char_image, (bbox[0], bbox[1]), char_image)
            return vertical_num
        attempts += 1

    return 0


def place_square_arrangement(canvas, bboxes, annotations, spacing=0.2, margin=10):
    """Places characters in a square arrangement."""
    # generate 4 chars of equal font and size
    font_size = random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)
    font = extract_random_font()
    generated = [generate_random_char_image(font_size, font) for _ in range(4)]

    char_width = generated[0][0].width  # equal sizes
    char_height = generated[0][0].height

    spacing = int(np.ceil(spacing * char_width))

    placed = False
    attempts = 0
    while not placed and attempts < 100:

        group_bboxes = []
        group_annotations = []
        char_images = []

        # arrangement anchor
        start_x = random.randint(margin, CANVAS_SIZE[0] - 2 * char_width - margin)
        start_y = random.randint(margin, CANVAS_SIZE[1] - 2 * char_height - margin)

        # character relative anchors
        for dx, dy in [(0, 0), (char_width + spacing, 0), (0, char_height + spacing),
                       (char_width + spacing, char_height + spacing)]:
            # absolute anchors
            x = start_x + dx
            y = start_y + dy

            bbox = (x, y, x + char_width, y + char_height)
            group_bboxes.append(bbox)

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            char_image, char, radical, font_filename = generated.pop(0)

            char_images.append(char_image)

            group_annotations.append(
                f"{char} {center_x} {center_y} {char_width} {char_height} {radical} {font_filename}"
            )

        if not check_group_overlap(group_bboxes, bboxes):
            for bbox, annotation, char_image in zip(group_bboxes, group_annotations, char_images):
                bboxes.append(bbox)
                annotations.append(annotation)
                canvas.paste(char_image, (bbox[0], bbox[1]), char_image)
            placed = True
        attempts += 1


def place_random(canvas, bboxes, annotations):
    """Places a single character randomly."""
    placed = False
    attempts = 0
    while not placed and attempts < 100:
        font_size = random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)
        char_image, char, radical, font_filename = generate_random_char_image(font_size)

        x = random.randint(0, CANVAS_SIZE[0] - char_image.width)
        y = random.randint(0, CANVAS_SIZE[1] - char_image.height)
        bbox = (x, y, x + char_image.width, y + char_image.height)

        if not check_overlap(bbox, bboxes):
            bboxes.append(bbox)
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            annotations.append(
                f"{char} {center_x} {center_y} {char_image.width} {char_image.height} {radical} {font_filename}"
            )
            canvas.paste(char_image, (x, y), char_image)
            placed = True
        attempts += 1


def generate_image(index):
    # create canvas
    canvas = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    # choose total num of characters to place
    num_chars = random.randint(1, MAX_CHARACTERS)
    bboxes = []
    annotations = []
    # place characters
    while num_chars > 0:
        # Prioritize placements
        if random.random() < 0.4 and num_chars >= 2:  # 40% chance for vertical arrangement
            num_chars -= place_vertical_arrangement(canvas, bboxes, annotations)
        if random.random() < 0.3 and num_chars >= 4:  # 30% chance for square arrangement
            place_square_arrangement(canvas, bboxes, annotations)
            num_chars -= 4
        else:  # Fallback to random placement
            place_random(canvas, bboxes, annotations)
            num_chars -= 1

    image_path = os.path.join(OUTPUT_DIR, f"{index}.png")
    canvas.save(image_path)

    annotation_path = os.path.join(OUTPUT_DIR, f"{index}.txt")
    with open(annotation_path, "w", encoding="utf-8") as f:
        f.write("\n".join(annotations))


for i in range(NUM_IMAGES):
    generate_image(i)
    print(f"Generated image {i + 1}/{NUM_IMAGES}")

conn.close()
