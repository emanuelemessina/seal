import random

import matplotlib.font_manager as fm
import torch
from matplotlib import pyplot as plt

from infer import infer
from load_checkpoint import load_checkpoint
from metrics import calculate_metrics
import torchvision.transforms as T

fprop = fm.FontProperties(fname='C:\Windows\Fonts\msjhl.ttc')


def visualize_predictions(images, boxes, scores, super_labels, sub_labels, dataset):
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

epsilon = 1e-8  # Small value to prevent division by zero

from collections import defaultdict
import numpy as np

from sklearn.metrics import auc

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection + epsilon
    return intersection / union


def evaluate_image(pred_boxes, pred_scores, sub_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    matched_gt = set()
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for pred_idx, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, sub_labels)):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_label == label and gt_idx not in matched_gt:
                iou = compute_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[pred_idx] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[pred_idx] = 1

    # Return true positives, false positives, and sorted scores
    return tp, fp, pred_scores


def compute_class_metrics(gt_boxes, predictions, iou_threshold=0.5):
    # Sort predictions by descending confidence
    predictions = sorted(predictions, key=lambda x: -x[1])
    tp = []
    fp = []
    matched = set()

    for pred_box, _ in predictions:
        found_match = False

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched:
                continue

            if compute_iou(pred_box, gt_box) >= iou_threshold:
                found_match = True
                matched.add(gt_idx)
                break

        if found_match:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    # Cumulative sums for precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + epsilon)
    recall = tp_cumsum / (len(gt_boxes) + epsilon)

    # Compute AP using trapezoidal rule (AUC)
    if len(recall) > 1:
        ap = auc(recall, precision)
    else:
        ap = 0

    return precision, recall, ap


def evaluate(device, model, multiscale_roi_align, dataset, dataloader, checkpoint_path, discard_optim):

    load_checkpoint(checkpoint_path, discard_optim, model)

    model.eval()

    # Metrics to compute
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    average_precisions = {}

    all_gt_boxes = defaultdict(list)  # Stores ground truth boxes per class
    all_gt_labels = defaultdict(list)  # Stores labels per class
    all_predictions = defaultdict(list)  # Stores predictions (boxes, scores) per class

    for idx, (image_b, targets_b) in enumerate(dataloader):
        print(f'Evaluating image {idx + 1}/{len(dataloader)}...')

        image_b = [image_b[0].to(device)]
        targets = targets_b[0]

        # Get predictions
        pred_boxes, pred_scores, super_labels, sub_labels = infer(
            model, multiscale_roi_align, device, image_b, targets_b, suppressionmaxxing_thresh=0
        )
        pred_boxes, pred_scores, sub_labels = pred_boxes[0], pred_scores[0], sub_labels[0].tolist()

        # Ground truth
        gt_boxes = targets["boxes"].tolist()
        gt_labels = targets["labels"].tolist()

        # Store per-class predictions and ground truths
        for box, label in zip(gt_boxes, gt_labels):
            all_gt_boxes[label].append(box)
            all_gt_labels[label].append(1)  # Mark this as unmatched initially

        for box, score, label in zip(pred_boxes, pred_scores, sub_labels):
            all_predictions[label].append((box, score))

        if idx == 2:
            break

    average_precisions = {}
    for label in all_gt_boxes.keys():
        precision, recall, ap = compute_class_metrics(all_gt_boxes[label], all_predictions[label])
        average_precisions[label] = ap
        print(f"Class {label}: AP = {ap:.4f}")

    mAP = np.mean(list(average_precisions.values())) if average_precisions else 0
    print(f"Total mAP: {mAP:.4f}")

    while True:

        images_targets = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(3)]
        images, targets = zip(*images_targets)

        pred_boxes, pred_scores, super_labels, sub_labels = infer(model, multiscale_roi_align, device, images, targets)

        visualize_predictions(images, pred_boxes, pred_scores, super_labels, sub_labels, dataset)

        answer = input("Do you want to see other images? (y/n, default is y): ").strip().lower()

        if answer not in ['y', '']:
            break
