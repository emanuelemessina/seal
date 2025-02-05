import random, os

import cv2
from PIL import Image, ImageDraw

import chardb
import chars
from bbox import *
import overlap
from arrangements import Arrangement
import arrangements
import backgrounds
import bbox as bb

from config import MIN_FONT_SIZE, MAX_FONT_SIZE, MAX_CHARACTERS, NUM_IMAGES, CANVAS_SIZE
from chars import char_to_label

# files

OUTPUT_DIR = "../output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "labels")
YOLO_ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "labels_yolo")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(YOLO_ANNOTATIONS_DIR, exist_ok=True)

def rotate_image(image, angle):
    """
    Rotates an RGBA PIL image by a given angle in degrees.

    Args:
        image (PIL.Image): The input RGBA image.
        angle (float): The angle in degrees to rotate the image.

    Returns:
        PIL.Image: The rotated RGBA image.
    """
    # Expand the canvas to fit the rotated image
    rotated_image = image.rotate(angle, expand=True, resample=Image.BILINEAR)

    # If the image has transparency, ensure it is preserved
    if image.mode == 'RGBA':
        # Create a new transparent image with the same size as the rotated image
        new_image = Image.new('RGBA', rotated_image.size, (0, 0, 0, 0))
        # Paste the rotated image onto the new image, preserving transparency
        new_image.paste(rotated_image, (0, 0), rotated_image)
        return new_image
    else:
        return rotated_image


def perspective_distort(image, fixed_side='bottom', factor=0.2):
    width, height = image.size

    # Convert PIL image to OpenCV format
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2BGRA)

    src_points = np.float32([
        [0, height],          # Bottom-left corner
        [width, height],      # Bottom-right corner
        [0, 0],               # Top-left corner
        [width, 0]            # Top-right corner
    ])

    dw = width*factor
    dh = height*factor

    # Define the destination points (transformed image) based on the fixed side
    if fixed_side == "bottom":
        dst_points = np.float32([
            [0, height],          # Bottom-left corner (fixed)
            [width, height],      # Bottom-right corner (fixed)
            [dw, 0],              # Top-left corner (moved)
            [width - dw, 0]       # Top-right corner (moved)
        ])
    elif fixed_side == "top":
        dst_points = np.float32([
            [dw, height],         # Bottom-left corner (moved)
            [width - dw, height], # Bottom-right corner (moved)
            [0, 0],               # Top-left corner (fixed)
            [width, 0]            # Top-right corner (fixed)
        ])
    elif fixed_side == "left":
        dst_points = np.float32([
            [0, height],          # Bottom-left corner (fixed)
            [width, height - dh], # Bottom-right corner (moved)
            [0, 0],               # Top-left corner (fixed)
            [width, dh]           # Top-right corner (moved)
        ])
    elif fixed_side == "right":
        dst_points = np.float32([
            [0, height - dh],     # Bottom-left corner (moved)
            [width, height],      # Bottom-right corner (fixed)
            [0, dh],              # Top-left corner (moved)
            [width, 0]            # Top-right corner (fixed)
        ])
    else:
        raise ValueError("Invalid fixed_side value. Choose from 'top', 'bottom', 'left', 'right'.")

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(opencv_image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Convert the result back to PIL format
    warped_pil_image = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGRA2RGBA))

    return warped_pil_image


def generate_random_font_size():
    return random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)


def arrange_random(arrangement_type, canvas, image_chars, bboxes, radicals, fonts):
    # choose font and size
    font_size = generate_random_font_size()
    font = chardb.extract_random_font()
    # generate first char
    generated = [chars.generate_random(font_size, font)]
    # get char size (estimate)
    est_char_width, est_char_height = generated[0][0].size
    # create arrangement instance
    factory = arrangement_type.value
    arrangement = factory(est_char_width, est_char_height)
    # number of characters to place
    n = arrangement.get_n()
    # generate remaining n-1 chars
    generated.extend([chars.generate_random(font_size, font) for _ in range(n - 1)])
    # fix for different widths (get the max width among the characters)
    # height should be equal for all
    arrangement.char_width = max(generated, key=lambda item: item[0].width)[0].width
    # attempt to place
    attempts = 0
    while attempts < 100:
        group_chars = []
        group_radicals = []
        group_fonts = []
        group_bboxes = []
        char_images = []
        char_anchors = []
        # arrangement anchor
        start_x, start_y = arrangements.generate_random_anchor(arrangement)
        # relative anchors
        relative_ancors = arrangement.get_chars_relative_anchors()
        # calculate absolute anchors, bboxes, annotations
        idx = 0
        for dx, dy in relative_ancors:
            # absolute anchors
            x = start_x + dx
            y = start_y + dy
            char_anchors.append((x, y))
            # char image
            char_image, char, radical, font_filename = generated[idx]
            char_image = perspective_distort(char_image, random.choice(['top', 'bottom', 'left', 'right']), random.uniform(0,0.2))
            inflate = 'positive'
            if random.random() > 0.5:
                char_image = rotate_image(char_image, random.uniform(-30, 30))
                inflate = 'negative'
            char_images.append(char_image)
            char_width, char_height = char_image.size  # don't trust first character, get actual

            group_chars.append(char)
            group_radicals.append(radical)
            group_fonts.append(font_filename)

            # bbox
            bbox = bb.calculate(x, y, char_width, char_height, inflate)
            group_bboxes.append(bbox)

            idx += 1
        # check overlap
        if not overlap.check_group(group_bboxes, bboxes):
            # append single chars to global lists
            for char, radical, font, bbox, char_image, char_anchor in zip(group_chars, group_radicals, group_fonts, group_bboxes, char_images,
                                                                 char_anchors):
                image_chars.append(char)
                bboxes.append(bbox)
                radicals.append(radical)
                fonts.append(font)
                canvas.paste(char_image, char_anchor, char_image)

            return n
        # retry
        attempts += 1
    return 0


def generate_image(index):
    # create canvas
    canvas = backgrounds.extract_random()
    draw = ImageDraw.Draw(canvas)
    # choose total num of characters to place
    num_chars = random.randint(1, MAX_CHARACTERS)
    bboxes = []
    chars = []
    radicals = []
    fonts = []
    annotations = []
    annotations_yolo = []
    # place characters
    while num_chars > 0:
        if random.random() > 0.5 and num_chars >= 2:  # 50% chance for vertical arrangement
            arrangement_type = Arrangement.VERTICAL
        elif random.random() <= 0.5 and num_chars >= 4:  # 50% chance for square arrangement
            arrangement_type = Arrangement.SQUARE
        else:  # Fallback to random placement
            arrangement_type = Arrangement.SINGLE

        num_chars -= arrange_random(arrangement_type, canvas, chars, bboxes, radicals, fonts)

    image_width = CANVAS_SIZE[0]
    image_height = CANVAS_SIZE[1]

    # annotations
    for char, bbox, radical, font_filename in zip(chars, bboxes, radicals, fonts):
        x1, y1, x2, y2 = bbox
        # yolo version
        center_x, center_y = bb.get_center(bbox)
        center_x /= image_width
        center_y /= image_height
        bbox_width, bbox_height = bb.get_w_h(bbox)
        bbox_width /= image_width
        bbox_height /= image_height
        annotations.append(f"{char} {x1} {y1} {x2} {y2} {radical} {font_filename}")
        annotations_yolo.append(f"{char_to_label(char)} {center_x} {center_y} {bbox_width} {bbox_height}")

    image_path = os.path.join(IMAGES_DIR, f"{index}.png")
    canvas.save(image_path)

    annotation_path = os.path.join(ANNOTATIONS_DIR, f"{index}.txt")
    with open(annotation_path, "w", encoding="utf-8") as f:
        f.write("\n".join(annotations))

    annotation_path = os.path.join(YOLO_ANNOTATIONS_DIR, f"{index}.txt")
    with open(annotation_path, "w", encoding="utf-8") as f:
        f.write("\n".join(annotations_yolo))


for i in range(NUM_IMAGES):
    generate_image(i)
    print(f"Generated image {i + 1}/{NUM_IMAGES}")

chardb.close_conn()
