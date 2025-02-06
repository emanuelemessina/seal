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


roi_output_size = 8
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
        return x


class CustomPredictor(nn.Module):
    def __init__(self, device, disable_hc, num_superclasses, superclasses_groups, in_lin_features, funneled_dim=1024, in_channels=256):
        super().__init__()

        self.device = device
        self.disable_hc = disable_hc

        self.box_distancer = nn.Sequential(nn.Linear(in_lin_features, 4096), nn.ReLU(), nn.Linear(4096, funneled_dim))
        self.cls_score_dummy = nn.Linear(funneled_dim, 2)
        self.bbox_pred = nn.Linear(funneled_dim, 2 * 4)  # hardcoded 2 classes (obj/bgd)

        self.super_logits = torch.empty(0)
        self.sub_logits = torch.empty(0)

        self.superclassifier_funnel = nn.Sequential(  # in_channels > num_superclasses
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # same 8x8
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # same 8x8
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # same 8x8
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),  # shrink 6x6
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=2, padding=1),  # same 6x6
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=2, padding=1),  # same 6x6
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),  # pool 3x3
        )

        self.superclassifier_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels*3*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_superclasses)  # super logits
        )

        self.subclassifier_funnel = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),  # same 8x8, expand channels
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # same 8x8
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # same 8x8
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3),  # shrink 6x6 expand channels
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=2, padding=1),  # same 6x6
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=2, padding=1),  # same 6x6
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),  # pool 3x3
        )

        subcl_junc_chans = (in_channels if not disable_hc else 0) + 1024

        self.subclassifier_junction = nn.Sequential(
            nn.Conv2d(subcl_junc_chans, subcl_junc_chans, kernel_size=1),  # dense like 1x1 + super
            nn.ReLU(),
            nn.Conv2d(subcl_junc_chans, subcl_junc_chans, kernel_size=1),  # again
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(subcl_junc_chans*3*3, 2048),
            nn.ReLU()
        )

        # subclasses per group 300
        self.sub_classifiers = nn.ModuleList([nn.Linear(2048, superclasses_groups[i][1]) for i in range(len(superclasses_groups))])

    def forward(self, x):
        x_flat = x.flatten(start_dim=1)
        box_features = F.relu(self.box_distancer(x_flat))

        scores = self.cls_score_dummy(box_features)
        bbox_deltas = self.bbox_pred(box_features)

        return scores, bbox_deltas

    def custom_forward(self, x):

        x_super = self.superclassifier_funnel(x)
        self.super_logits = self.superclassifier_head(x_super)

        x_sub = self.subclassifier_funnel(x)

        if not self.disable_hc:
            x_sub = torch.cat((x_super, x_sub), 1)
        x_sub = self.subclassifier_junction(x_sub)

        self.sub_logits = torch.empty(0).to(self.device)

        for i, sub_cl in enumerate(self.sub_classifiers):
            # predict outputs of this subclass group with this group's head
            sub_head_output = sub_cl(x_sub)
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
