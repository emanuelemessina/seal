import random

import matplotlib.font_manager as fm
import numpy as np
import torch
from matplotlib import pyplot as plt

from infer import infer
from load_checkpoint import load_checkpoint
import torchvision.transforms as T

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import ConfusionMatrix

from metrics import confmat_update, plot_per_class_metrics, draw_confmat
from metrics import map_update

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


def evaluate(what, device, model, multiscale_roi_align, dataset, dataloader, checkpoint_path, discard_optim, max_images):
    load_checkpoint(checkpoint_path, discard_optim, model)

    model.eval()

    if 'visual' in what:
        visual_inspection(device, model, multiscale_roi_align, dataset)

    if 'metrics' in what:
        calc_metrics(device, model, multiscale_roi_align, dataset, dataloader, max_images)


def calc_metrics(device, model, multiscale_roi_align, dataset, dataloader,  max_images=4000):
    # Initialize the metrics
    map_metric = MeanAveragePrecision(class_metrics=True, extended_summary=True, iou_thresholds=[0.75])

    confmat = ConfusionMatrix(task="multiclass", num_classes=len(dataset.classes) + 1)  # add background

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
        pred_boxes, pred_scores, sub_labels = pred_boxes[0].detach().cpu(), torch.tensor(pred_scores[0]), sub_labels[
            0].detach().cpu()

        map_update(map_metric, gt_boxes, gt_labels, pred_boxes, pred_scores, sub_labels)

        confmat_update(confmat, gt_boxes, gt_labels, pred_boxes, sub_labels)

        if max_images is not None and idx == max_images:
            break

    print("Computing metrics...")

    # Compute mAP and extract precision-recall stats
    results = map_metric.compute()

    print("mAP@0.75:", results["map"].item())
    print("mAR:", results["mar_100"].item())

    print("Plotting per-class metrics...")

    plot_per_class_metrics(results)

    print("Computing confusion matrix...")

    draw_confmat(confmat)


def visual_inspection(device, model, multiscale_roi_align, dataset):
    while True:

        images_targets = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(3)]
        images, targets = zip(*images_targets)

        pred_boxes, pred_scores, super_labels, sub_labels = infer(model, multiscale_roi_align, device, images, targets)

        visualize_predictions(images, pred_boxes, pred_scores, super_labels, sub_labels, dataset)

        answer = input("Do you want to see other images? (y/n, default is y): ").strip().lower()

        if answer not in ['y', '']:
            break
