from enum import (Enum)
import numpy as np
import random
from config import MARGIN, CANVAS_SIZE, SPACING


# arrangement helpers

def generate_random_anchor(arrangement):
    group_width, group_height = arrangement.get_group_w_and_h()
    start_x = random.randint(MARGIN, CANVAS_SIZE[0] - group_width - MARGIN)
    start_y = random.randint(MARGIN, CANVAS_SIZE[1] - group_height - MARGIN)
    return start_x, start_y


# character arrangment

class ArrangementFactory:
    def __init__(self, char_width, char_height):
        """
        char_width and char_height are used to estimate the group size and maximum number of chars that can be placed
        """
        self.char_width = char_width
        self.char_height = char_height
        self.spacing = int(np.ceil(SPACING * char_width))

    def get_n(self):
        """get number of character to place"""
        return 0

    def get_group_w_and_h(self):
        return 0, 0

    def get_chars_relative_anchors(self):
        return []


class SingleCharacter(ArrangementFactory):
    def __init__(self, char_width, char_height):
        super().__init__(char_width, char_height)

    def get_n(self):
        return 1

    def get_group_w_and_h(self):
        return self.char_width, self.char_height

    def get_chars_relative_anchors(self):
        return [(0, 0)]


class SquareSeal(ArrangementFactory):
    def __init__(self, char_width, char_height):
        super().__init__(char_width, char_height)

    def get_n(self):
        return 4

    def get_group_w_and_h(self):
        return 2 * self.char_width + self.spacing, 2 * self.char_height + self.spacing

    def get_chars_relative_anchors(self):
        return [(0, 0),
                (self.char_width + self.spacing, 0),
                (0, self.char_height + self.spacing),
                (self.char_width + self.spacing, self.char_height + self.spacing)]


class Column(ArrangementFactory):
    def __init__(self, char_width, char_height):
        super().__init__(char_width, char_height)
        self.max_vertical_num = int(np.floor((CANVAS_SIZE[1] - 2 * MARGIN) / (char_height + self.spacing)))
        self.vertical_num = random.randint(2, self.max_vertical_num)

    def get_n(self):
        return self.vertical_num

    def get_group_w_and_h(self):
        return self.char_width, self.vertical_num * (self.char_height + self.spacing)

    def get_chars_relative_anchors(self):
        return [(0, (self.char_height + self.spacing) * n) for n in range(self.vertical_num)]


class Arrangement(Enum):
    SINGLE = SingleCharacter
    SQUARE = SquareSeal
    VERTICAL = Column
