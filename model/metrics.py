import torch
from collections import defaultdict
from sklearn.metrics import auc

from infer import infer


def calculate_metrics(dataloader, model, multiscale_roi_align, device, iou_threshold=0.5):
    # Initialize dictionaries to store TP, FP, and FN for each class
    class_tp = defaultdict(list)
    class_fp = defaultdict(list)
    class_fn = defaultdict(list)

    # Process each image in the dataloader
    for idx, (image_b, targets_b) in enumerate(dataloader):
        print(f'Evaluating image {idx + 1}/{len(dataloader)}...')
        image_b = [image_b[0].to(device)]
        targets = targets_b[0]

        # Get predictions
        pred_boxes, pred_scores, super_labels, sub_labels = infer(model, multiscale_roi_align, device, image_b, targets_b, suppressionmaxxing_thresh=0)
        pred_boxes = pred_boxes[0]
        pred_scores = pred_scores[0]
        super_labels = super_labels[0]
        sub_labels = sub_labels[0]

        # Ground truth
        gt_boxes = targets['boxes']
        gt_labels = targets['labels']

        # Match predictions with ground truth
        for label in torch.unique(gt_labels):
            label = label.item()

            # Get predictions and ground truth for this class
            pred_mask = (sub_labels == label)
            gt_mask = (gt_labels == label)

            pred_boxes_class = pred_boxes[pred_mask]
            pred_scores_class = [pred_scores[i] for i, mask in enumerate(pred_mask) if mask]
            gt_boxes_class = gt_boxes[gt_mask]

            # Sort predictions by confidence score
            sorted_indices = torch.argsort(torch.tensor(pred_scores_class), descending=True)
            pred_boxes_class = pred_boxes_class[sorted_indices]
            pred_scores_class = [pred_scores_class[i] for i in sorted_indices]

            # Initialize TP and FP for this class
            tp = torch.zeros(len(pred_boxes_class), dtype=torch.bool)
            fp = torch.zeros(len(pred_boxes_class), dtype=torch.bool)

            # Match predictions to ground truth
            for i, pred_box in enumerate(pred_boxes_class):
                if len(gt_boxes_class) == 0:
                    fp[i] = True
                    continue

                # Calculate IoU with all ground truth boxes
                ious = box_iou(pred_box.unsqueeze(0), gt_boxes_class)
                max_iou, gt_idx = torch.max(ious, dim=1)

                if max_iou >= iou_threshold:
                    tp[i] = True
                    gt_boxes_class = torch.cat([gt_boxes_class[:gt_idx], gt_boxes_class[gt_idx + 1:]])
                else:
                    fp[i] = True

            # Store TP and FP for this class
            class_tp[label].append(tp)
            class_fp[label].append(fp)
            class_fn[label].append(len(gt_boxes_class))

    # Calculate precision, recall, AP, and mAP
    class_metrics = {}
    for label in class_tp.keys():
        tp = torch.cat(class_tp[label])
        fp = torch.cat(class_fp[label])
        fn = sum(class_fn[label])

        # Precision and Recall
        precision = torch.sum(tp) / (torch.sum(tp) + torch.sum(fp) + 1e-10)
        recall = torch.sum(tp) / (torch.sum(tp) + fn + 1e-10)

        # Average Precision (AP) using the area method
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        recalls = tp_cumsum / (torch.sum(tp) + fn + 1e-10)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # Append zero to make the curve start at (0, 0)
        recalls = torch.cat([torch.tensor([0.0]), recalls])
        precisions = torch.cat([torch.tensor([1.0]), precisions])

        # Calculate AP using the area under the precision-recall curve
        ap = auc(recalls.numpy(), precisions.numpy())

        # Store metrics for this class
        class_metrics[label] = {
            'precision': precision.item(),
            'recall': recall.item(),
            'ap': ap
        }

    # Calculate mean AP (mAP)
    mean_ap = sum([metrics['ap'] for metrics in class_metrics.values()]) / len(class_metrics)

    return class_metrics, mean_ap


def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes.
    Args:
        boxes1 (Tensor): Shape (N, 4)
        boxes2 (Tensor): Shape (M, 4)
    Returns:
        iou (Tensor): Shape (N, M)
    """
    # Calculate intersection areas
    inter = intersection(boxes1, boxes2)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    iou = inter / union
    return iou


def intersection(boxes1, boxes2):
    """
    Calculate intersection areas between two sets of boxes.
    Args:
        boxes1 (Tensor): Shape (N, 4)
        boxes2 (Tensor): Shape (M, 4)
    Returns:
        inter (Tensor): Shape (N, M)
    """
    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    return inter


def box_area(boxes):
    """
    Calculate area of boxes.
    Args:
        boxes (Tensor): Shape (N, 4)
    Returns:
        area (Tensor): Shape (N,)
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])