import argparse
import glob
import os
import random
import re
import sys
import threading
import time
from typing import Dict, List, Tuple
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from model.infer import infer
from model.model import make_model, multiscale_roi_align
from model.train import train

# Add the parent directory of the dataset module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.dataset import CharacterDataset
from model.load_checkpoint import load_checkpoint

parser = argparse.ArgumentParser(description='SEAL')
parser.add_argument('--checkpoint_path', type=str, default='',
                    help='Path to the checkpoint file ("ignore" to force no checkpoint, otherwise latest checkpoint found will be used)')
parser.add_argument('--eval', type=str, default='',
                    help='Which dataset to evaluate (train|dev|test), if empty the model trains on the train dataset')
parser.add_argument('--force_cpu', type=bool, default=False, help='Force to use the CPU instead of CUDA')
parser.add_argument('--discard_optim', type=bool, default=False, help='Discard optim state dict')
args = parser.parse_args()

checkpoint_path = args.checkpoint_path
eval = args.eval
force_cpu = args.force_cpu
discard_optim = args.discard_optim

device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
print(f'Using device: {device}')

# Define the dataset and dataloader
data_folder = '../dataset/output'
batch_size = 2
dataset = CharacterDataset(data_folder, split=('train' if not eval else eval), transform=T.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
num_classes = len(dataset.classes)
num_superclasses = len(dataset.radical_counts)
superclasses_groups = dataset.radical_groups

model, multiscale_roi_align = make_model(device, dataset.mean, dataset.std, num_superclasses, superclasses_groups)

print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

if not eval:
    train(device, model, multiscale_roi_align, dataset, dataloader, batch_size, checkpoint_path, discard_optim)
    quit()
else:
    load_checkpoint(checkpoint_path, discard_optim, model)
    model.eval()

from torchvision.ops import box_iou

import matplotlib.font_manager as fm

fprop = fm.FontProperties(fname='C:\Windows\Fonts\msjhl.ttc')


def compute_iou(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return torch.tensor(0.0)
    ious = box_iou(pred_boxes, gt_boxes)
    max_ious = ious.max(dim=1)[0]
    return max_ious.mean().item()


def evaluate_detection(dataloader):
    iou_scores = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            print(f'Evaluating batch {batch_idx + 1}/{len(dataloader)}...')
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].detach().cpu()
                gt_boxes = target["boxes"].cpu()

                iou = compute_iou(pred_boxes, gt_boxes)
                iou_scores.append(iou)

            if batch_idx == 10:
                # early stop
                break

    avg_iou = sum(iou_scores) / len(iou_scores)
    print(f"Average IoU: {avg_iou:.4f}")


def visualize_predictions(images, boxes, scores, super_labels, sub_labels):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, image, im_boxes, im_scores, im_superlabels, im_sublabels in zip(axes, images, boxes, scores, super_labels,
                                                                            sub_labels):
        ax.imshow(T.ToPILImage()(image))
        im_boxes = im_boxes.detach().cpu()
        for box, score, superlabel, sublabel in zip(im_boxes, im_scores, im_superlabels, im_sublabels):
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, color='red', linewidth=2))
            ax.text(x1, y1 - 5, f"[{score:.2f}] {dataset.classes[sublabel]} ({dataset.radical_counts[superlabel][0]})",
                    color='red', fontproperties=fprop)
    plt.show()


print(f"Evaluating on {eval} set")

while True:

    images_targets = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(3)]
    images, targets = zip(*images_targets)

    pred_boxes, pred_scores, super_labels, sub_labels = infer(model, multiscale_roi_align, device, images, targets)

    visualize_predictions(images, pred_boxes, pred_scores, super_labels, sub_labels)

    answer = input("Do you want to see other images? (y/n, default is y): ").strip().lower()

    if answer not in ['y', '']:
        break

evaluate_detection(dataloader)
