from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import MultiScaleRoIAlign
import torch.nn.functional as F


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
    sizes=((16,), (50,), (70,),),
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
    def __init__(self, device, disable_hc, num_superclasses, superclasses_groups, in_features, high_dim=4096,
                 mid_dim=1024,
                 funneled_dim=256):
        super().__init__()

        self.device = device
        self.disable_hc = disable_hc

        self.box_distancer = nn.Sequential(nn.Linear(in_features, mid_dim), nn.ReLU(), nn.Linear(mid_dim, mid_dim))
        self.cls_score_dummy = nn.Linear(mid_dim, 2)
        self.bbox_pred = nn.Linear(mid_dim, 2 * 4)  # hardcoded 2 classes (obj/bgd)

        self.super_logits = torch.empty(0)
        self.sub_logits = torch.empty(0)

        self.classifier_distancer = nn.Sequential(nn.Linear(in_features, high_dim), nn.ReLU(), nn.Linear(high_dim, high_dim))

        self.superclassifier_funnel = nn.Linear(high_dim, mid_dim)
        self.subclassifier_funnel = nn.Linear(high_dim, mid_dim)

        self.super_classifier = nn.Linear(mid_dim, num_superclasses)  # superclasses 200

        # in_ch 12k, subclasses per group 300
        self.sub_classifiers = nn.ModuleList(
            [nn.Linear((num_superclasses if not disable_hc else 0) + mid_dim, superclasses_groups[i][1]) for i in
             range(len(superclasses_groups))])

    def forward(self, x):
        x = x.flatten(start_dim=1)

        box_features = F.relu(self.box_distancer(x))

        scores = self.cls_score_dummy(box_features)
        bbox_deltas = self.bbox_pred(box_features)

        return scores, bbox_deltas

    def custom_forward(self, x):
        x = x.flatten(start_dim=1)

        x1 = F.relu(self.classifier_distancer(x))

        # predict superclasses
        x_super = F.relu(self.superclassifier_funnel(x1))
        self.super_logits = self.super_classifier(x_super)

        x_sub = F.relu(self.subclassifier_funnel(x1))
        self.sub_logits = torch.empty(0).to(self.device)

        for i, sub_cl in enumerate(self.sub_classifiers):
            if self.disable_hc:
                sub_head_output = sub_cl(x_sub)
            else:
                # predict outputs of this subclass group with this group's head, input is cat superlogits | shared features
                sub_head_output = sub_cl(torch.cat((self.super_logits, x_sub), 1))
            # append it to the total output
            self.sub_logits = torch.cat((self.sub_logits, sub_head_output), 1)

        return self.super_logits, self.sub_logits


def make_model(device, mean, std, disable_hc, num_superclasses, superclasses_groups):
    model = FasterRCNN(image_mean=mean, image_std=std, min_size=min_size, max_size=max_size,
                       backbone=backbone, rpn_anchor_generator=anchor_generator,
                       rpn_head=rpn_head,
                       box_roi_pool=box_roi_align,
                       box_head=BypassHead(input_features_size),
                       box_predictor=CustomPredictor(device, disable_hc, num_superclasses,
                                                     superclasses_groups, input_features_size))
    model.to(device)

    return model, multiscale_roi_align
