import os
import sqlite3
from enum import Enum
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

LOCAL_PATH = os.path.dirname(__file__)

CHARACTER_DB_FILENAME = "chardb.sqlite"
CHARACTER_DB_PATH = os.path.join(LOCAL_PATH, "chardb", CHARACTER_DB_FILENAME)

conn = sqlite3.connect(CHARACTER_DB_PATH)
cursor = conn.cursor()


def get_all_characters():
    cursor.execute("SELECT character FROM characters")
    rows = cursor.fetchall()
    return [row[0] for row in rows]


def get_radical_character_counts():
    # Fetch the radical and its associated character count
    cursor.execute(
        """
        SELECT radical, COUNT(character)
        FROM characters
        GROUP BY radical
        """
    )
    radical_counts = cursor.fetchall()
    return sorted(radical_counts, key=lambda x: x[1], reverse=True)


def get_radicals():
    radical_counts = get_radical_character_counts()
    radicals = [radical for radical, count in radical_counts]
    return radicals


def group_radical_counts(radical_counts, threshold):
    grouped_counts = []
    current_group = []
    current_sum = 0

    for radical, count in radical_counts:
        if current_sum + count > threshold:
            # group would be over threshold
            if current_group:  # current group is non empty
                # insert current group
                grouped_counts.append((current_group, current_sum))
            # create new group with current cursor
            current_group = [radical]
            current_sum = count
        else:  # can group because sum is under threshold
            current_group.append(radical)
            current_sum += count

    if current_group:  # insert last group
        grouped_counts.append((current_group, current_sum))

    return grouped_counts


def load_weights(data_folder, what):
    if what == 'radicals':
        what = "radical_stats.csv"
    elif what == 'chars':
        what = "character_stats.csv"
    else:
        raise ValueError
    csv_path = os.path.join(data_folder, what)
    df = pd.read_csv(csv_path)
    weights = torch.tensor(df["Weight"].values, dtype=torch.float32)
    return weights


def load_image_mean_std(data_folder):
    with open(os.path.join(data_folder, "image_stats.txt"), 'r') as f:
        lines = f.readlines()
        mean = torch.tensor(eval(lines[0].split('mean ')[1].strip()))
        std = torch.tensor(eval(lines[1].split('std ')[1].strip()))

        return mean, std


class CharacterDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        self.image_files = [f for f in os.listdir(os.path.join(data_folder, "images")) if f.endswith('.png')]

        self.classes = get_all_characters()
        self.classes = ['background'] + self.classes

        self.labels = {self.classes[idx]: idx for idx in range(len(self.classes))}

        self.radical_counts = get_radical_character_counts()
        self.radical_labels = {self.radical_counts[i][0]: i for i in range(len(self.radical_counts))}
        self.radical_groups = group_radical_counts(self.radical_counts, self.radical_counts[0][1])

        self.mean, self.std = load_image_mean_std(data_folder)
        self.class_weights = load_weights(data_folder, 'chars')
        self.superclass_weights = load_weights(data_folder, 'radicals')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image file
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_folder, "images", img_name)

        # Open image
        image = Image.open(img_path).convert("RGB")

        # Get label file
        label_name = img_name.replace('.png', '.txt')
        label_path = os.path.join(self.data_folder, "labels", label_name)

        # Parse labels
        boxes = []
        labels = []
        superlabels = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                char = parts[0]
                center_x, center_y, width, height = map(float, parts[1:5])
                radical = parts[5]
                font_file = parts[6]
                # Convert center_x, center_y, width, height to x_min, y_min, x_max, y_max
                x_min = center_x - width / 2
                y_min = center_y - height / 2
                x_max = center_x + width / 2
                y_max = center_y + height / 2
                boxes.append([x_min, y_min, x_max, y_max])

                labels.append(self.labels[char])
                superlabels.append(self.radical_labels[radical])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        superlabels = torch.as_tensor(superlabels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "superlabels": superlabels}

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, target
