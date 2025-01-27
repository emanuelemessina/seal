import argparse
import glob
import os
import re
import sys
import threading
import time
from typing import Dict, List, Tuple
import torch
import torchvision.transforms as T
from torch import nn
from torch.fft import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.patches as patches
import numpy as np
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN, TwoMLPHead
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops import MultiScaleRoIAlign
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Add the parent directory of the dataset module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.dataset import CharacterDataset

parser = argparse.ArgumentParser(description='SEAL')
parser.add_argument('--model_type', type=str, default='custom', help='Type of the model to use (custom or standard)')
parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint file')
args = parser.parse_args()

model_type = args.model_type
checkpoint_path = args.checkpoint_path

# Check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# Define the dataset and dataloader
data_folder = '../dataset/output'
dataset = CharacterDataset(data_folder, transform=T.ToTensor())
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
num_classes = len(dataset.classes)
num_superclasses = len(dataset.radical_counts)
superclasses_groups = dataset.radical_groups

min_size = max_size = 256

backbone = resnet_fpn_backbone(backbone_name="resnet50", trainable_layers=5, weights=ResNet50_Weights.IMAGENET1K_V1)

anchor_generator = AnchorGenerator(
    sizes=((16,), (32,), (64,), (128,), (256,),),
    aspect_ratios=((1.0, 2.0,),) * 5
)
rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0])


class CustomRoIAlign(MultiScaleRoIAlign):
    def __init__(self, featmap_names, output_size=7, sampling_ratio=2):
        super().__init__(featmap_names, output_size, sampling_ratio)
        self.features = torch.empty(0)
        self.image_shapes = torch.empty(0)

    def forward(
            self,
            x: Dict[str, Tensor],
            boxes: List[Tensor],
            image_shapes: List[Tuple[int, int]],
    ) -> Tensor:
        self.features = x
        self.image_shapes = image_shapes

        return super().forward(x, boxes, image_shapes)


roi_output_size = 8
featmap_names = ["0", "1", "2", "3"]
roi_sampling_ratio = 2

multiscale_roi_align = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=roi_output_size,
                                          sampling_ratio=roi_sampling_ratio)
box_roi_align = CustomRoIAlign(featmap_names=featmap_names, output_size=roi_output_size,
                               sampling_ratio=roi_sampling_ratio)

representation_size = 1024
input_features_size = backbone.out_channels * roi_output_size ** 2


class SubClassifier(nn.Module):
    def __init__(self, in_channels, representation_size, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomBoxHead(nn.Module):
    def __init__(self, in_channels, representation_size, num_superclasses, superclasses_groups):
        super().__init__()

        self.two_mlp_head = TwoMLPHead(in_channels, representation_size)

        self.super_logits = torch.empty(0)
        self.sub_logits = torch.empty(0)

        self.super_classifier = nn.Sequential(TwoMLPHead(in_channels, representation_size),
                                              nn.Linear(representation_size, num_superclasses))

        self.sub_classifiers = nn.ModuleList(
            [SubClassifier(num_superclasses + in_channels, representation_size, superclasses_groups[i][1]) for i in
             range(len(superclasses_groups))])

    def forward(self, x):
        x = x.flatten(start_dim=1)

        if not self.training:
            self.custom_forward(x)

        #  bypass of input feature to box predictor
        return self.two_mlp_head(x)

    def custom_forward(self, x):
        x = x.flatten(start_dim=1)
        self.sub_logits = torch.empty(0).to(device)

        # predict superclasses
        self.super_logits = self.super_classifier(x)

        for i, sub_cl in enumerate(self.sub_classifiers):
            # predict outputs of this subclass group with this group's head, input is cat superlogits | shared features
            sub_head_output = sub_cl(torch.cat((self.super_logits, x), 1))
            # append it to the total output
            self.sub_logits = torch.cat((self.sub_logits, sub_head_output), 1)

        return self.super_logits, self.sub_logits


if model_type == "custom":
    model = FasterRCNN(image_mean=dataset.mean, image_std=dataset.std, min_size=min_size, max_size=max_size,
                       backbone=backbone, rpn_anchor_generator=anchor_generator,
                       rpn_head=rpn_head,
                       box_roi_pool=box_roi_align,
                       box_head=CustomBoxHead(input_features_size, representation_size, num_superclasses,
                                              superclasses_groups),
                       box_predictor=FastRCNNPredictor(representation_size, 2))

    # using custom classification, don't care about the fake one
    for param in model.roi_heads.box_predictor.cls_score.parameters():
        param.requires_grad = False
else:
    model = FasterRCNN(image_mean=dataset.mean, image_std=dataset.std, min_size=min_size, max_size=max_size,
                       backbone=backbone, rpn_anchor_generator=anchor_generator,
                       rpn_head=rpn_head,
                       box_roi_pool=box_roi_align, box_head=TwoMLPHead(input_features_size, representation_size),
                       box_predictor=FastRCNNPredictor(representation_size, num_classes))

print(model)

model.to(device)

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
loss_fn_superclass = CrossEntropyLoss(weight=dataset.superclass_weights.to(device))
loss_fn_class = CrossEntropyLoss(weight=dataset.class_weights.to(device))
lambda_superclasses = 1
lambda_classes = 0.9

try:
    if not checkpoint_path:
        checkpoint_files = glob.glob(f'checkpoint_{model_type}_*.pth')
        if len(checkpoint_files) == 0:
            print("No checkpoint, training from scratch.")
            raise Exception  # to break out
        checkpoint_files.sort(key=lambda x: re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', x).group(), reverse=True)
        checkpoint_path = checkpoint_files[0]
    print(f'Using checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
except Exception as e:
    if e is FileNotFoundError:
        print(f"No checkpoint found at '{checkpoint_path}'.")
        quit()

# Training loop

model.train()

log_file = open('loss_output.txt', 'a')


def log(msg):
    print(msg)
    log_file.write(msg+'\n')


log(f'Started training {time.strftime("%Y-%m-%d_%H-%M-%S")}')


def plot_losses_async(plot_data, model_type):
    plt.ion()
    fig, ax = plt.subplots()
    rpn_classification_line, = ax.plot([], [], label='RPN Classification Loss')
    rpn_localization_line, = ax.plot([], [], label='RPN Localization Loss')
    frcnn_localization_line, = ax.plot([], [], label='Head Localization Loss')

    if model_type == "standard":
        frcnn_classification_line, = ax.plot([], [], label='Head Classification Loss')
    else:
        custom_classification_line, = ax.plot([], [], label='Custom Classification Loss')

    ax.set_xlabel('#batch/10')
    ax.set_ylabel('Loss')
    ax.legend()

    current_max = 1.0

    while True:
        if plot_data['stop_flag']:
            break

        # Update the data for the plot
        rpn_classification_line.set_data(range(len(plot_data['rpn_classification'])), plot_data['rpn_classification'])
        rpn_localization_line.set_data(range(len(plot_data['rpn_localization'])), plot_data['rpn_localization'])
        frcnn_localization_line.set_data(range(len(plot_data['frcnn_localization'])), plot_data['frcnn_localization'])

        if model_type == "standard":
            frcnn_classification_line.set_data(range(len(plot_data['frcnn_classification'])),
                                               plot_data['frcnn_classification'])
        else:
            custom_classification_line.set_data(range(len(plot_data['custom_classification'])),
                                                plot_data['custom_classification'])

        # Determine the maximum value among the last items
        last_values = [
            plot_data['rpn_classification'][-1] if plot_data['rpn_classification'] else 0,
            plot_data['rpn_localization'][-1] if plot_data['rpn_localization'] else 0,
            plot_data['frcnn_localization'][-1] if plot_data['frcnn_localization'] else 0,
        ]
        if model_type == "standard":
            last_values.append(plot_data['frcnn_classification'][-1] if plot_data['frcnn_classification'] else 0)
        else:
            last_values.append(plot_data['custom_classification'][-1] if plot_data['custom_classification'] else 0)

        max_last_value = max(last_values)

        if max_last_value < 0.2 * current_max:
            current_max = 0.2 * current_max
        elif max_last_value > current_max:
            current_max = 2 * max_last_value

        ax.autoscale_view()
        ax.relim()
        ax.set_ylim(0, current_max)
        fig.canvas.flush_events()
        time.sleep(0.1)  # Update interval


# Shared data for plotting
plot_data = {
    'rpn_classification': [],
    'rpn_localization': [],
    'frcnn_localization': [],
    'frcnn_classification': [],
    'custom_classification': [],
    'stop_flag': False
}

# Start the plotting thread
plot_thread = threading.Thread(target=plot_losses_async, args=(plot_data, model_type))
plot_thread.start()

num_epochs = 1
for epoch in range(num_epochs):

    log(f'Epoch {epoch} ...')

    rpn_classification_losses = []
    rpn_localization_losses = []
    frcnn_classification_losses = []
    frcnn_localization_losses = []
    custom_classification_losses = []

    for batch_idx, (images, targets) in enumerate(dataloader):

        log(f'Batch {batch_idx + 1}/{len(dataloader)}...')

        optimizer.zero_grad()

        gt_boxes = []
        gt_sublabels = []
        gt_superlabels = []
        for idx, target in enumerate(targets):
            target['boxes'] = target['boxes'].float().to(device)
            gt_boxes.append(target['boxes'])
            if model_type == "custom":
                target['superlabels'] = target['superlabels'].long().to(device)
                target['sublabels'] = target['labels'].long().to(
                    device) - 1  # subtract the background class because in the custom classifier we don't have it
                target['labels'] = torch.ones_like(target['labels']).long().to(
                    device)  # dummy labels for the fake classifier
                gt_sublabels.append(target['sublabels'])
                gt_superlabels.append(target['superlabels'])
            else:
                target['labels'] = target['labels'].long().to(device)

        images = [image.float().to(device) for image in images]

        batch_losses = model(images, targets)

        custom_loss = 0
        if model_type == "custom":
            features = model.roi_heads.box_roi_pool.features
            image_shapes = model.roi_heads.box_roi_pool.image_shapes
            box_features = multiscale_roi_align(features, gt_boxes, image_shapes)
            custom_classifier = model.roi_heads.box_head.custom_forward
            super_logits, sub_logits = custom_classifier(box_features)
            gt_sublabels = torch.cat(gt_sublabels, dim=0)
            gt_superlabels = torch.cat(gt_superlabels, dim=0)
            custom_loss = lambda_superclasses * loss_fn_superclass(super_logits,
                                                                   gt_superlabels) + lambda_classes * loss_fn_class(
                sub_logits, gt_sublabels)

        loss = batch_losses['loss_box_reg']
        loss += batch_losses['loss_rpn_box_reg']
        loss += batch_losses['loss_objectness']
        if model_type == "standard":
            loss += batch_losses['loss_classifier']
        else:
            loss += custom_loss

        loss.backward()

        rpn_classification_losses.append(batch_losses['loss_objectness'].item())
        rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
        frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())
        if model_type == "standard":
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
        else:
            custom_classification_losses.append(custom_loss.item())

        optimizer.step()

        if batch_idx % 10 == 0:
            rpn_classification_mean = np.mean(rpn_classification_losses)
            rpn_localization_mean = np.mean(rpn_localization_losses)
            frcnn_localization_mean = np.mean(frcnn_localization_losses)

            loss_output = ''
            loss_output += f'{"RPN Classification Loss":<26}: {rpn_classification_mean:.4f}\n'
            loss_output += f'{"RPN Localization Loss":<26}: {rpn_localization_mean:.4f}\n'
            loss_output += f'{"Head Localization Loss":<26}: {frcnn_localization_mean:.4f}\n'
            if model_type == "standard":
                frcnn_classification_mean = np.mean(frcnn_classification_losses)
                loss_output += f'{"Head Classification Loss":<26}: {frcnn_classification_mean:.4f}\n'
            else:
                custom_classification_mean = np.mean(custom_classification_losses)
                loss_output += f'{"Custom Classification Loss":<26}: {custom_classification_mean:.4f}'

            log(loss_output)

            # Update the plot data
            plot_data['rpn_classification'].append(rpn_classification_mean / np.max(rpn_classification_losses))
            plot_data['rpn_localization'].append(rpn_localization_mean / np.max(rpn_localization_losses))
            plot_data['frcnn_localization'].append(frcnn_localization_mean / np.max(frcnn_localization_losses))

            if model_type == "standard":
                plot_data['frcnn_classification'].append(frcnn_classification_mean / np.max(frcnn_classification_losses))
            else:
                plot_data['custom_classification'].append(custom_classification_mean / np.max(custom_classification_losses))

        if batch_idx != 0 and batch_idx % 1000 == 0:
            # save state
            checkpoint_name = f'checkpoint_{model_type}_e{epoch}_b{batch_idx}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth'
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       checkpoint_name)
            log(f"Saved checkpoint {checkpoint_name}")

log("Training done.")

log_file.close()
