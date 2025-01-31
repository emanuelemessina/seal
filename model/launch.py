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
from sympy import false
from torch import nn
from torch.fft import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
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

from model.load_checkpoint import load_checkpoint

# Add the parent directory of the dataset module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.dataset import CharacterDataset

parser = argparse.ArgumentParser(description='SEAL')
parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint file ("ignore" to force no checkpoint, otherwise latest checkpoint found will be used)')
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

min_size = max_size = 256  # so that the images are not resized! (and so the boxes aren't either)

backbone = resnet_fpn_backbone(backbone_name="resnet50", trainable_layers=5, returned_layers=[2, 3], weights=None)

'''
ResNet Block (Stage) | Feature Map | Output Size (for 256x256 image) | Layer Index
Conv2_x	C1	64x64	1
Conv3_x	C2	32x32	2
Conv4_x	C3	16x16	3
Conv5_x	C4	8x8	    4

Feature Map Level | Image Size Downsampling	| Typical Target Object Size
P2	8x (32x32)	Small objects (32-64)
P3	16x (16x16)	Medium objects (64–128)
P4	32x (8x8)   Large objects (128+)
P5	64x (4x4)	
'''

anchor_generator = AnchorGenerator(
    sizes=((16,), (50, ), (70, ),),
    aspect_ratios=((1.0, 2.0,),) * 3
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


roi_output_size = 7
featmap_names = ["0", "1"]
roi_sampling_ratio = 2

multiscale_roi_align = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=roi_output_size,
                                          sampling_ratio=roi_sampling_ratio)
box_roi_align = CustomRoIAlign(featmap_names=featmap_names, output_size=roi_output_size,
                               sampling_ratio=roi_sampling_ratio)


input_features_size = backbone.out_channels * roi_output_size ** 2


class BypassHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return x


class CustomPredictor(nn.Module):
    def __init__(self, num_superclasses, superclasses_groups, in_features, high_dim=4096, mid_dim=1024, funneled_dim=256):
        super().__init__()

        self.box_distancer = nn.Sequential(nn.Linear(in_features, mid_dim), nn.ReLU(), nn.Linear(mid_dim, mid_dim))
        self.cls_score_dummy = nn.Linear(mid_dim, 2)
        self.bbox_pred = nn.Linear(mid_dim, 2 * 4)  # hardcoded 2 classes (obj/bgd)

        self.super_logits = torch.empty(0)
        self.sub_logits = torch.empty(0)

        self.classifier_distancer = nn.Sequential(nn.Linear(in_features, high_dim), nn.ReLU(), nn.Linear(high_dim, high_dim))
        self.superclassifier_funnel = nn.Linear(high_dim, mid_dim)
        self.subclassifier_funnel = nn.Linear(high_dim, mid_dim)

        self.super_classifier = nn.Linear(mid_dim, num_superclasses)  # superclasses 150
        # in_ch 12k, subclasses per group 50
        self.sub_classifiers = nn.ModuleList([nn.Linear(num_superclasses + mid_dim, superclasses_groups[i][1]) for i in range(len(superclasses_groups))])

    def forward(self, x):
        x = x.flatten(start_dim=1)

        box_features = F.relu(self.box_distancer(x))

        scores = self.cls_score_dummy(box_features)
        bbox_deltas = self.bbox_pred(box_features)

        return scores, bbox_deltas

    def custom_forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.classifier_distancer(x))

        x_super = F.relu(self.superclassifier_funnel(x))
        x_sub = F.relu(self.subclassifier_funnel(x))

        self.sub_logits = torch.empty(0).to(device)

        # predict superclasses
        self.super_logits = self.super_classifier(x_super)

        for i, sub_cl in enumerate(self.sub_classifiers):
            # predict outputs of this subclass group with this group's head, input is cat superlogits | shared features
            sub_head_output = sub_cl(torch.cat((self.super_logits, x_sub), 1))
            # append it to the total output
            self.sub_logits = torch.cat((self.sub_logits, sub_head_output), 1)

        return self.super_logits, self.sub_logits

model = FasterRCNN(image_mean=dataset.mean, image_std=dataset.std, min_size=min_size, max_size=max_size,
                   backbone=backbone, rpn_anchor_generator=anchor_generator,
                   rpn_head=rpn_head,
                   box_roi_pool=box_roi_align,
                   box_head=BypassHead(input_features_size),
                   box_predictor=CustomPredictor(num_superclasses,
                                                 superclasses_groups, input_features_size))

print(model)

model.to(device)

# split params for different learning rates (optional)

box_regression_params = []
classification_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "bbox_pred" in name:
        box_regression_params.append(param)
    else:
        classification_params.append(param)

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
iterations_per_epoch = (len(dataset) + batch_size - 1) // batch_size
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=iterations_per_epoch, T_mult=2, eta_min=0.0001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
loss_fn_superclass = CrossEntropyLoss(weight=dataset.superclass_weights.to(device))
loss_fn_class = CrossEntropyLoss(weight=dataset.class_weights.to(device))
lambda_superclasses = 0.99
lambda_classes = 0.95

load_checkpoint(checkpoint_path, discard_optim, model, optimizer, scheduler)

if not eval:

    # Training loop

    model.train()

    date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = open(f'log_{date_time}.txt', 'a')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')

    loss_file = open(f'loss_{date_time}.csv', 'a')

    loss_file.write('epoch,batch,rpn_localization_loss,rpn_classification_loss,frcnn_localization_loss,frcnn_classification_loss')
    loss_file.write(',custom_classification_super_loss,custom_classification_sub_loss')
    loss_file.write('\n')

    log(f'Started training {date_time}')

    num_epochs = 10
    for epoch in range(num_epochs):

        log(f'Epoch {epoch} ...')

        for batch_idx, (images, targets) in enumerate(dataloader):

            log(f'Batch {batch_idx + 1}/{len(dataloader)}...')

            rpn_classification_losses = []
            rpn_localization_losses = []
            frcnn_classification_losses = []
            frcnn_localization_losses = []
            custom_classification_superlosses = []
            custom_classification_sublosses = []

            optimizer.zero_grad()

            gt_boxes = []
            gt_sublabels = []
            gt_superlabels = []
            for idx, target in enumerate(targets):
                target['boxes'] = target['boxes'].float().to(device)
                gt_boxes.append(target['boxes'])
                target['superlabels'] = target['superlabels'].long().to(device)
                target['sublabels'] = target['labels'].long().to(device)
                target['labels'] = torch.ones_like(target['labels']).long().to(device)  # dummy labels800 for the binary classifier
                gt_sublabels.append(target['sublabels'])
                gt_superlabels.append(target['superlabels'])

            images = [image.float().to(device) for image in images]

            batch_losses = model(images, targets)

            superloss = subloss = 0

            features = model.roi_heads.box_roi_pool.features
            image_shapes = model.roi_heads.box_roi_pool.image_shapes
            box_features = multiscale_roi_align(features, gt_boxes, image_shapes)
            box_head = model.roi_heads.box_head
            x1 = box_head(box_features)
            custom_classifier = model.roi_heads.box_predictor.custom_forward
            super_logits, sub_logits = custom_classifier(x1)
            gt_sublabels = torch.cat(gt_sublabels, dim=0)
            gt_superlabels = torch.cat(gt_superlabels, dim=0)
            superloss = lambda_superclasses * loss_fn_superclass(super_logits, gt_superlabels)
            subloss = lambda_classes * loss_fn_class(sub_logits, gt_sublabels)

            loss = batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']
            loss += batch_losses['loss_classifier']

            loss += superloss + subloss

            loss.backward()

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())

            custom_classification_superlosses.append(superloss.item())
            custom_classification_sublosses.append(subloss.item())

            optimizer.step()

            if batch_idx % 10 == 0:
                rpn_classification_mean = np.mean(rpn_classification_losses)
                rpn_localization_mean = np.mean(rpn_localization_losses)
                frcnn_localization_mean = np.mean(frcnn_localization_losses)

                loss_output = ''
                loss_output += f'{"RPN Localization Loss":<26}: {rpn_localization_mean:.20f}\n'
                loss_output += f'{"RPN Classification Loss":<26}: {rpn_classification_mean:.20f}\n'
                loss_output += f'{"Head Localization Loss":<26}: {frcnn_localization_mean:.20f}\n'
                frcnn_classification_mean = np.mean(frcnn_classification_losses)
                loss_output += f'{"Head Classification Loss":<26}: {frcnn_classification_mean:.20f}\n'

                custom_classification_supermean = np.mean(custom_classification_superlosses)
                custom_classification_submean = np.mean(custom_classification_sublosses)
                loss_output += f'{"Super Classification Loss":<26}: {custom_classification_supermean:.20f}\n'
                loss_output += f'{"Sub Classification Loss":<26}: {custom_classification_submean:.20f}\n'

                log(loss_output)

                loss_file.write(f'{epoch},{batch_idx},{rpn_localization_mean:.20f},{rpn_classification_mean:.20f},{frcnn_localization_mean:.20f},{frcnn_classification_mean:.20f}')

                loss_file.write(f',{custom_classification_supermean:.20f},{custom_classification_submean:.20f}')
                loss_file.write('\n')

            if batch_idx != 0 and batch_idx % (len(dataset)//(batch_size*2) - 1) == 0:
                # save state
                checkpoint_name = f'checkpoint_e{epoch}_b{batch_idx}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth'
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),  'scheduler_state_dict': scheduler.state_dict()},
                           checkpoint_name)
                log(f"Saved checkpoint {checkpoint_name}")

            #scheduler.step(epoch + batch_idx / iterations_per_epoch)
        scheduler.step()

    log("Training done.")
    log_file.close()
    loss_file.close()
    quit()
else:
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
    for ax, image, im_boxes, im_scores, im_superlabels, im_sublabels in zip(axes, images, boxes, scores, super_labels, sub_labels):
        ax.imshow(T.ToPILImage()(image))
        im_boxes = im_boxes.detach().cpu()
        for box, score, superlabel, sublabel in zip(im_boxes, im_scores, im_superlabels, im_sublabels):
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, color='red', linewidth=2))
            ax.text(x1, y1 - 5, f"[{score:.2f}] {dataset.classes[sublabel]} ({dataset.radical_counts[superlabel][0]})", color='red', fontproperties=fprop)
    plt.show()

if eval == 'train':
    print("Evaluating on Train set")

    while True:

        images_targets = [dataset[random.randint(0, len(dataset)-1)] for _ in range(3)]
        images, targets = zip(*images_targets)

        outputs = model([img.to(device) for img in images])

        pred_boxes = []
        pred_labels = []
        scores = []
        boxes_per_image = []
        for output, target in zip(outputs, targets):
            pred_boxes.append(output["boxes"])
            pred_labels.append(output["labels"].detach().cpu())
            scores.append(output["scores"].detach().cpu())
            boxes_per_image.append(output['boxes'].shape[0])

        features = model.roi_heads.box_roi_pool.features
        image_shapes = model.roi_heads.box_roi_pool.image_shapes
        box_features = multiscale_roi_align(features, pred_boxes, image_shapes)
        custom_classifier = model.roi_heads.box_predictor.custom_forward
        super_logits, sub_logits = custom_classifier(box_features)
        super_scores = F.softmax(super_logits, -1)
        sub_scores = F.softmax(sub_logits, -1)
        super_labels = torch.argmax(super_scores, dim=1).detach().cpu().split(boxes_per_image, 0)
        sub_labels = torch.argmax(sub_scores, dim=1).detach().cpu().split(boxes_per_image, 0)

        visualize_predictions(images, pred_boxes, scores, super_labels, sub_labels)

        answer = input("Do you want to see other images? (y/n, default is y): ").strip().lower()

        if answer not in ['y', '']:
            break

    evaluate_detection(dataloader)


