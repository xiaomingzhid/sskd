# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from fastreid.layers import *
from fastreid.modeling.heads import EmbeddingHead
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier


@REID_HEADS_REGISTRY.register()
class CamAwareHead(EmbeddingHead):
    def __init__(self, cfg):
        super().__init__(cfg)

        # fmt: off
        in_feat     = cfg.MODEL.BACKBONE.FEAT_DIM
        cam_classes = cfg.MODEL.HEADS.CAM_CLASSES
        cam_feat    = cfg.MODEL.HEADS.CAM_FEAT
        # fmt: on

        self.cam_bottleneck = nn.Sequential(
            nn.Conv2d(in_feat, cam_feat, 1, 1),
            get_norm(cfg.MODEL.HEADS.NORM, cam_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(p=0.5),
        )
        self.cam_bottleneck.apply(weights_init_kaiming)

        self.cam_classifier = nn.Linear(cam_feat, cam_classes, bias=False)
        self.cam_classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        outputs = super().forward(features, targets)

        # Camera branch
        global_feat = self.pool_layer(features)
        cam_feat = self.cam_bottleneck(global_feat)
        cam_feat = cam_feat[..., 0, 0]

        # Evaluation
        # fmt: off
        if not self.training: return outputs, cam_feat
        # fmt: on

        # Training
        # Camera branch
        cam_cls_outputs = self.cam_classifier(cam_feat)
        outputs["cam_cls_outputs"] = cam_cls_outputs

        return outputs
