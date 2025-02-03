import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from infer import infer
from model import make_model
from train import train

# Add the parent directory of the dataset module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics import compute_map
from dataset.dataset import CharacterDataset
from load_checkpoint import load_checkpoint

parser = argparse.ArgumentParser(description='SEAL')
parser.add_argument('--checkpoint_path', type=str, default='',
                    help='Path to the checkpoint file ("ignore" to force no checkpoint, otherwise latest checkpoint found will be used)')
parser.add_argument('--eval', type=str, default='',
                    help='Which dataset to evaluate (train|dev|test), if empty the model trains on the train dataset')
parser.add_argument('--force_cpu', type=bool, default=False, help='Force to use the CPU instead of CUDA')
parser.add_argument('--discard_optim', type=bool, default=False, help='Discard optim state dict')
parser.add_argument('--disable_hc', type=bool, default=False, help='Disable hierachical classification')

args = parser.parse_args()

checkpoint_path = args.checkpoint_path
eval = args.eval
force_cpu = args.force_cpu
discard_optim = args.discard_optim
disable_hc = args.disable_hc

device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
print(f'Using device: {device}')

# Define the dataset and dataloader
data_folder = '../dataset/output'
batch_size = 2
dataset = CharacterDataset(data_folder, split=('train' if not eval else eval), transform=T.ToTensor())
num_classes = len(dataset.classes)
num_superclasses = len(dataset.radical_counts)
superclasses_groups = dataset.radical_groups

model, multiscale_roi_align = make_model(device, dataset.mean, dataset.std, disable_hc, num_superclasses, superclasses_groups)

print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

if not eval:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    train(device, model, multiscale_roi_align, dataset, dataloader, batch_size, checkpoint_path, discard_optim)
    quit()

load_checkpoint(checkpoint_path, discard_optim, model)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
model.eval()

from torchvision.ops import box_iou

import matplotlib.font_manager as fm

fprop = fm.FontProperties(fname='C:\Windows\Fonts\msjhl.ttc')


def evaluate_map(dataloader):
    tot_gt_boxes = []
    tot_pred_boxes = []
    with torch.no_grad():
        for idx, (image_b, targets_b) in enumerate(dataloader):
            print(f'Evaluating image {idx + 1}/{len(dataloader)}...')
            image_b = [image_b[0].to(device)]
            pred_boxes, pred_scores, super_labels, sub_labels = infer(model, multiscale_roi_align, device, image_b, targets_b, suppressionmaxxing_thresh=0)
            tot_pred_boxes.append({'object': [[*tuple(pred_boxes[0][i].numpy()), pred_scores[0][i]] for i in range(len(pred_boxes[0]))]})
            targets = targets_b[0]
            tot_gt_boxes.append({'object': [targets['boxes'][i].numpy() for i in range(len(targets['boxes']))]})

    print("Calculating mAP...")
    mean_ap, per_class_ap, precisions, recalls = compute_map(tot_pred_boxes, tot_gt_boxes, method='interp')
    print('Mean Average Precision : {:.4f}'.format(mean_ap))
    plt.figure()
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()


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

evaluate_map(dataloader)
