import random

import matplotlib.font_manager as fm
import torch
from matplotlib import pyplot as plt

from infer import infer
from load_checkpoint import load_checkpoint
from metrics import compute_map
import torchvision.transforms as T

fprop = fm.FontProperties(fname='C:\Windows\Fonts\msjhl.ttc')


def evaluate_map(device, model, multiscale_roi_align, dataloader):
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


def evaluate(device, model, multiscale_roi_align, dataset, dataloader, checkpoint_path, discard_optim):

    load_checkpoint(checkpoint_path, discard_optim, model)

    model.eval()

    print(f"Evaluating on {eval} set")

    while True:

        images_targets = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(3)]
        images, targets = zip(*images_targets)

        pred_boxes, pred_scores, super_labels, sub_labels = infer(model, multiscale_roi_align, device, images, targets)

        visualize_predictions(images, pred_boxes, pred_scores, super_labels, sub_labels, dataset)

        answer = input("Do you want to see other images? (y/n, default is y): ").strip().lower()

        if answer not in ['y', '']:
            break

    evaluate_map(device, model, multiscale_roi_align, dataloader)

