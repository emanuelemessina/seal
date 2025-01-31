import math

import numpy as np
from config import config

BBOX_INFLATE = config["BBOX_INFLATE"]  # percentage of char width to inflate the bbox


def calculate(x, y, width, height, inflate='positive'):
    if inflate is None:
        inflate = 0
    else:
        inflate = int(BBOX_INFLATE * width) * (-1 if inflate == 'negative' else 1)

    bbox = (x - inflate, y - inflate, x + width + inflate, y + height + inflate)
    return bbox


def get_center(bbox):
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return center_x, center_y


def get_w_h(bbox):
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    return bbox_width, bbox_height