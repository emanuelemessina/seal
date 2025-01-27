import random, os
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import chardb
import chars
from bbox import *
import overlap
from arrangements import Arrangement
import arrangements
import backgrounds
import bbox as bb

from config import MIN_FONT_SIZE, MAX_FONT_SIZE, MAX_CHARACTERS, NUM_IMAGES


# files

OUTPUT_DIR = "../output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)


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


def generate_random_font_size():
    return random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)


def arrange_random(arrangement_type, canvas, bboxes, annotations):
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
        group_bboxes = []
        group_annotations = []
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
            char_images.append(char_image)
            char_width, char_height = char_image.size  # don't trust first character, get actual
            # bbox
            bbox = bb.calculate(x, y, char_width, char_height)
            group_bboxes.append(bbox)
            center_x, center_y = bb.get_center(bbox)
            bbox_width, bbox_height = bb.get_w_h(bbox)
            # annotation
            group_annotations.append(
                f"{char} {center_x} {center_y} {bbox_width} {bbox_height} {radical} {font_filename}"
            )
            idx += 1
        # check overlap
        if not overlap.check_group(group_bboxes, bboxes):
            # append single chars to global lists
            for bbox, annotation, char_image, char_anchor in zip(group_bboxes, group_annotations, char_images,
                                                                 char_anchors):
                bboxes.append(bbox)
                annotations.append(annotation)
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
    annotations = []
    # place characters
    while num_chars > 0:
        if random.random() > 0.5 and num_chars >= 2:  # 50% chance for vertical arrangement
            arrangement_type = Arrangement.VERTICAL
        elif random.random() <= 0.5 and num_chars >= 4:  # 50% chance for square arrangement
            arrangement_type = Arrangement.SQUARE
        else:  # Fallback to random placement
            arrangement_type = Arrangement.SINGLE

        num_chars -= arrange_random(arrangement_type, canvas, bboxes, annotations)

    image_path = os.path.join(IMAGES_DIR, f"{index}.png")
    canvas.save(image_path)

    annotation_path = os.path.join(ANNOTATIONS_DIR, f"{index}.txt")
    with open(annotation_path, "w", encoding="utf-8") as f:
        f.write("\n".join(annotations))


for i in range(NUM_IMAGES):
    generate_image(i)
    print(f"Generated image {i + 1}/{NUM_IMAGES}")

chardb.close_conn()
