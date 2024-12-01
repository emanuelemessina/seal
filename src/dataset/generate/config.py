import json

CONFIG_FILE = "config.json"


def load_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


config = load_config()

# vars

NUM_IMAGES = config["NUM_IMAGES"]  # Number of images to generate
MIN_FONT_SIZE = config["MIN_FONT_SIZE"]
MAX_FONT_SIZE = config["MAX_FONT_SIZE"]
MAX_CHARACTERS = config["MAX_CHARACTERS"]  # Maximum characters per image

CANVAS_SIZE = tuple(config["CANVAS_SIZE"])
MARGIN = config["MARGIN"]  # padding from canvas borders for characters placement
SPACING = config["SPACING"]  # spacing between characters in an arrangement