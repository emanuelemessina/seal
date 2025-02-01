import torch
import torch.nn.functional as F

SUPPRESSIONMAXXING_THRESH = 0.9


def infer(model, multiscale_roi_align, device, images, targets):
    outputs = model([img.to(device) for img in images])

    pred_boxes = []
    pred_scores = []
    boxes_per_image = []
    for output, target in zip(outputs, targets):
        boxes = output["boxes"]
        scores = output["scores"].detach().cpu()

        # low thresh suppresh
        filtered_boxes = torch.empty(size=(0, 4)).to(device)
        filtered_scores = []
        for i, score in enumerate(scores):
            if score >= SUPPRESSIONMAXXING_THRESH:
                filtered_boxes = torch.cat((filtered_boxes, boxes[i].reshape(1, -1)))
                filtered_scores.append(float(score))

        pred_boxes.append(filtered_boxes)
        pred_scores.append(filtered_scores)
        boxes_per_image.append(len(filtered_boxes))

    features = model.roi_heads.box_roi_pool.features
    image_shapes = model.roi_heads.box_roi_pool.image_shapes
    box_features = multiscale_roi_align(features, pred_boxes, image_shapes)
    custom_classifier = model.roi_heads.box_predictor.custom_forward
    super_logits, sub_logits = custom_classifier(box_features)
    super_scores = F.softmax(super_logits, -1)
    sub_scores = F.softmax(sub_logits, -1)
    super_labels = torch.argmax(super_scores, dim=1).detach().cpu().split(boxes_per_image, 0)
    sub_labels = torch.argmax(sub_scores, dim=1).detach().cpu().split(boxes_per_image, 0)

    return pred_boxes, pred_scores, super_labels, sub_labels
