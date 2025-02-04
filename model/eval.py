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


from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import ConfusionMatrix
from collections import defaultdict

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

def evaluate(device, model, multiscale_roi_align, dataset, dataloader, checkpoint_path, discard_optim):
    load_checkpoint(checkpoint_path, discard_optim, model)

    model.eval()

    # Initialize the metric
    map_metric = MeanAveragePrecision()

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    all_precision = []
    all_recall = []

    confmat = ConfusionMatrix(task="multiclass", num_classes=len(dataset.classes))

    for idx, (image_b, targets_b) in enumerate(dataloader):
        print(f'Evaluating image {idx + 1}/{len(dataloader)}...')

        image_b = [image_b[0].to(device)]
        targets = targets_b[0]

        gt_boxes = targets["boxes"].detach().cpu()
        gt_labels = targets["labels"].detach().cpu()

        # Get model predictions
        pred_boxes, pred_scores, _, sub_labels = infer(
            model, multiscale_roi_align, device, image_b, targets_b
        )
        pred_boxes, pred_scores, sub_labels = pred_boxes[0].detach().cpu(), torch.tensor(pred_scores[0]), sub_labels[0].detach().cpu()

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

        # Perform IoU-based matching
        tp_labels, fp_labels, fn_labels = match_predictions(
            pred_boxes, sub_labels, gt_boxes, gt_labels
        )

        # Update confusion matrix
        confmat.update(torch.tensor(tp_labels + fp_labels),  # Predictions
                       torch.tensor(tp_labels + fn_labels))  # Targets

        if idx == 200:
            break

    # Compute mAP and extract precision-recall stats
    results = map_metric.compute()
    print("Final mAP:", results["map"].item())

    # Get the confusion matrix
    conf_matrix = confmat.compute()
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot the confusion matrix as a heatmap
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix.int().cpu(), annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Extract precision-recall for plotting
    for class_id in results["classes"]:
        precision = results["precision"][class_id]
        recall = results["recall"][class_id]
        all_precision.append(precision.tolist())
        all_recall.append(recall.tolist())

    # Plot PR curve for each class
    for idx, (precision, recall) in enumerate(zip(all_precision, all_recall)):
        plt.plot(recall, precision, label=f'Class {idx}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    while True:

        images_targets = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(3)]
        images, targets = zip(*images_targets)

        pred_boxes, pred_scores, super_labels, sub_labels = infer(model, multiscale_roi_align, device, images, targets)

        visualize_predictions(images, pred_boxes, pred_scores, super_labels, sub_labels, dataset)

        answer = input("Do you want to see other images? (y/n, default is y): ").strip().lower()

        if answer not in ['y', '']:
            break
