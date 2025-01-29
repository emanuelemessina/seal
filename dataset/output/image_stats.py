import os
import random
from PIL import Image
import torch
import numpy as np

print("sampling images...")

all_files = os.listdir('images')
png_files = [f for f in all_files if f.endswith('.png')]
num_files = len(png_files)

sample_size = max(1, num_files // 5)
sampled_files = random.sample(png_files, sample_size)
images = [Image.open(os.path.join('images', f)) for f in sampled_files]

print("creating tensors...")

images = [torch.tensor(np.array(img)).float() for img in images]
images = torch.stack(images)
images = images.permute(0, 3, 1, 2)

print("calculating...")

mean = images.mean(dim=(0, 2, 3))/255
std = images.std(dim=(0, 2, 3))/255

with open("image_stats.txt", "w") as f:
    print(f'mean: {mean}')
    print(f'std: {std}')
    f.write(f"mean {str(mean.numpy().tolist())}\nstd {str(std.numpy().tolist())}")
