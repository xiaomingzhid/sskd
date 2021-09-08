# encoding: utf-8
"""
@author:  lingxiao he
@contact: helingxiao3@jd.com
"""

from torch import nn
import torch
import torch.nn.functional as F
from fastreid.layers import *
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY
from fastreid.utils.weight_init import weights_init_classifier, weights_init_kaiming


class OcclusionUnit(nn.Module):
    def __init__(self, in_planes=2048):
        super(OcclusionUnit, self).__init__()
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)
        self.mask_layer = nn.Linear(in_planes, 1, bias=False)

    def forward(self, x):
        SpaFeat1 = self.MaxPool1(x)  # shape: [n, c, h, w]
        SpaFeat2 = self.MaxPool2(x)
        SpaFeat3 = self.MaxPool3(x)
        SpaFeat4 = self.MaxPool4(x)

        Feat1 = SpaFeat1.view(SpaFeat1.size(0), SpaFeat1.size(1), SpaFeat1.size(2) * SpaFeat1.size(3))
        Feat2 = SpaFeat2.view(SpaFeat2.size(0), SpaFeat2.size(1), SpaFeat2.size(2) * SpaFeat2.size(3))
        Feat3 = SpaFeat3.view(SpaFeat3.size(0), SpaFeat3.size(1), SpaFeat3.size(2) * SpaFeat3.size(3))
        Feat4 = SpaFeat4.view(SpaFeat4.size(0), SpaFeat4.size(1), SpaFeat4.size(2) * SpaFeat4.size(3))
        SpatialFeatAll = torch.cat((Feat1, Feat2, Feat3, Feat4), 2)
        SpatialFeatAll = SpatialFeatAll.transpose(1, 2)  # shape: [n, c, m]
        y = self.mask_layer(SpatialFeatAll)
        mask_weight = torch.sigmoid(y[:, :, 0])

        feat_dim = SpaFeat1.size(2) * SpaFeat1.size(3)
        mask_score = F.normalize(mask_weight[:, :feat_dim], p=1, dim=1)
        mask_weight_norm = F.normalize(mask_weight, p=1, dim=1)
        mask_score = mask_score.unsqueeze(1)

        SpaFeat1 = SpaFeat1.transpose(1, 2)
        SpaFeat1 = SpaFeat1.transpose(2, 3)  # shape: [n, h, w, c]
        SpaFeat1 = SpaFeat1.view((SpaFeat1.size(0), SpaFeat1.size(1) * SpaFeat1.size(2), -1))  # shape: [n, h*w, c]

        global_feats = mask_score.matmul(SpaFeat1).view(SpaFeat1.shape[0], -1, 1, 1)
        return global_feats, mask_weight, mask_weight_norm


@REID_HEADS_REGISTRY.register()
class DSRHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        self.pool_layer = pool_layer

        self.mask_layer = nn.Conv2d(in_feat, 1, 1, bias=False)

        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)

        self.bnneck_occ = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck_occ.apply(weights_init_kaiming)

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':
            self.classifier = nn.Linear(in_feat, num_classes, bias=False)
            self.classifier_occ = nn.Linear(in_feat, num_classes, bias=False)
        elif cls_type == 'arcface':
            self.classifier = Arcface(cfg, in_feat, num_classes)
            self.classifier_occ = Arcface(cfg, in_feat, num_classes)
        elif cls_type == 'circle':
            self.classifier = Circle(cfg, in_feat, num_classes)
            self.classifier_occ = Circle(cfg, in_feat, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcface' and 'circle'.")

        self.classifier.apply(weights_init_classifier)
        self.classifier_occ.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """

        mask_feat = self.mask_layer(features)  # (b, 1, h, w)
        mask_score = torch.sigmoid(mask_feat)

        mask_size = mask_score.size()
        mask_score = mask_score.view(mask_size[0], 1, -1)  # (b, 1, h*w)
        mask_score = F.normalize(mask_score, p=1, dim=-1)

        feat_size = features.size()
        spatial_feat = features.view(feat_size[0], feat_size[1], -1).transpose(1, 2)  # (b, h*w, c)
        foreground_feat = mask_score.matmul(spatial_feat).view(feat_size[0], -1, 1, 1)  # (b, c, 1, 1)

        bn_foreground_feat = self.bnneck_occ(foreground_feat)
        bn_foreground_feat = bn_foreground_feat[..., 0, 0]

        # Evaluation
        if not self.training: return bn_foreground_feat

        # Training

        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        try:
            cls_outputs = self.classifier(bn_feat)
            fore_cls_outputs = self.classifier_occ(bn_foreground_feat)
        except TypeError:
            cls_outputs = self.classifier(bn_feat, targets)
            fore_cls_outputs = self.classifier_occ(bn_foreground_feat, targets)

        pred_class_logits = F.linear(bn_foreground_feat, self.classifier.weight)

        if self.neck_feat == "before":  fore_feat = foreground_feat[..., 0, 0]
        elif self.neck_feat == "after": fore_feat = bn_foreground_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")

        return cls_outputs, fore_cls_outputs, pred_class_logits, fore_feat
