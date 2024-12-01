import chardb
import os
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = "../fonts"


# generate char image

def generate_random(font_size, font=None):
    """:returns char_image, char, radical, font_filename"""
    char, radical, font_filename, query = chardb.extract_random_character(font)
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
