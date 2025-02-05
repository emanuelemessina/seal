import numpy as np
import torch
from collections import defaultdict

from matplotlib import pyplot as plt

from infer import infer


def calculate_iou(box1, box2):
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection and area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def match_predictions(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5):
    matched_indices = set()
    tp_labels = []
    fp_labels = []
    unmatched_gt_labels = gt_labels.tolist()

    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_indices:
                continue

            iou = calculate_iou(pred_box.tolist(), gt_box.tolist())
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx != -1:
            # this was the best matching bb, check label
            if pred_labels[pred_idx] == gt_labels[best_gt_idx]:
                # matching label -> tp
                tp_labels.append(pred_labels[pred_idx].item())
                unmatched_gt_labels.remove(gt_labels[best_gt_idx].item())
                matched_indices.add(best_gt_idx)
            else:  # wrong label -> fp
                fp_labels.append(pred_labels[pred_idx].item())
        else:
            # No match or low IoU; count as FP
            fp_labels.append(pred_labels[pred_idx].item())

    # Remaining unmatched ground truths are FNs
    fn_labels = unmatched_gt_labels

    return tp_labels, fp_labels, fn_labels


def map_update(map_metric, gt_boxes, gt_labels, pred_boxes, pred_scores, sub_labels):
    # Format predictions and targets
    predictions = [{
        "boxes": pred_boxes,
        "scores": pred_scores,
        "labels": sub_labels
    }]

    targets_formatted = [{
        "boxes": gt_boxes,
        "labels": gt_labels
    }]

    # Update the mAP metric
    map_metric.update(predictions, targets_formatted)


def confmat_update(confmat, gt_boxes, gt_labels, pred_boxes, sub_labels):

    BACKGROUND_CLASS = confmat.num_classes - 1  # background class label

    # Perform IoU-based matching
    tp_labels, fp_labels, fn_labels = match_predictions(
        pred_boxes, sub_labels, gt_boxes, gt_labels
    )

    # Add background class for unmatched predictions
    pred_classes = tp_labels + fp_labels
    target_classes = tp_labels + fn_labels

    pred_mismatch = len(sub_labels) - len(gt_labels)
    background_pad = [BACKGROUND_CLASS] * abs(len(fp_labels) - len(fn_labels))
    if pred_mismatch > 0:  # more false positives (pred is longer)
        target_classes += background_pad
    elif pred_mismatch < 0:  # mroe false negatives (target is longer)
        pred_classes += background_pad

    # Update confusion matrix
    confmat.update(
        torch.tensor(pred_classes),
        torch.tensor(target_classes)
    )


def plot_per_class_metrics(map_results):
    results = map_results
    mar_100_per_class = [max(0, val) for val in results['mar_100_per_class'].tolist()]
    map_per_class = [max(0, val) for val in results['map_per_class'].tolist()]
    classes = results['classes'].tolist()
    x = np.arange(len(classes))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, mar_100_per_class, width, label='mAR')
    rects2 = ax.bar(x + width / 2, map_per_class, width, label='mAP@0.75')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Classes')
    ax.set_ylabel('Value')
    ax.set_title('mAR and mAP per class')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.legend()

    fig.tight_layout()

    plt.show()


def draw_confmat(confmat, filename):
    # Get the confusion matrix
    conf_matrix = confmat.compute().numpy()

    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix = np.divide(
        conf_matrix, row_sums, out=np.zeros_like(conf_matrix, dtype=float), where=row_sums != 0
    )

    print("Writing confusion matrix...")

    plt.figure(figsize=(12, 10))  # Set larger figure size for better resolution
    plt.imshow(conf_matrix, cmap="Blues")
    plt.colorbar()

    # Add axis labels and ticks
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Save without displaying
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print("Confusion matrix saved.")