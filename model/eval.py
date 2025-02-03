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


def compute_class_metrics(tp, fp, scores):
    # Sort by score descending
    sorted_indices = np.argsort(-np.array(scores))
    tp = tp[sorted_indices]
    fp = fp[sorted_indices]

    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Precision and Recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + epsilon)
    recall = tp_cumsum / (len(tp_cumsum) if len(tp_cumsum) > 0 else epsilon)

    # Average precision
    ap = auc(recall, precision) if len(recall) > 0 else 0

    return precision, recall, ap


def evaluate(device, model, multiscale_roi_align, dataset, dataloader, checkpoint_path, discard_optim):

    load_checkpoint(checkpoint_path, discard_optim, model)

    model.eval()

    # Metrics to compute
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    average_precisions = {}

    for idx, (image_b, targets_b) in enumerate(dataloader):
        print(f'Evaluating image {idx + 1}/{len(dataloader)}...')

        image_b = [image_b[0].to(device)]
        targets = targets_b[0]

        # Get predictions
        pred_boxes, pred_scores, super_labels, sub_labels = infer(
            model, multiscale_roi_align, device, image_b, targets_b, suppressionmaxxing_thresh=0
        )
        pred_boxes, pred_scores, sub_labels = pred_boxes[0], pred_scores[0], sub_labels[0]

        # Ground truth
        gt_boxes = targets["boxes"].tolist()
        gt_labels = targets["labels"].tolist()

        # Evaluate for each class
        tp, fp, scores = evaluate_image(pred_boxes, pred_scores, sub_labels, gt_boxes, gt_labels)

        for label in set(sub_labels + gt_labels):
            class_tp = np.array([tp[i] for i in range(len(tp)) if sub_labels[i] == label])
            class_fp = np.array([fp[i] for i in range(len(fp)) if sub_labels[i] == label])
            class_scores = [scores[i] for i in range(len(scores)) if sub_labels[i] == label]

            precision, recall, ap = compute_class_metrics(class_tp, class_fp, class_scores)

            precisions[label].append(precision)
            recalls[label].append(recall)
            average_precisions[label] = ap

    map_score = np.mean(list(average_precisions.values())) if average_precisions else 0
    print(f"mAP: {map_score:.4f}")

    while True:

        images_targets = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(3)]
        images, targets = zip(*images_targets)

        pred_boxes, pred_scores, super_labels, sub_labels = infer(model, multiscale_roi_align, device, images, targets)

        visualize_predictions(images, pred_boxes, pred_scores, super_labels, sub_labels, dataset)

        answer = input("Do you want to see other images? (y/n, default is y): ").strip().lower()

        if answer not in ['y', '']:
            break
