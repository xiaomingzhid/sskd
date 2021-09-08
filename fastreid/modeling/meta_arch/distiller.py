# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging

from torch import nn
import torch
import torch.nn.functional as F
import copy
from fastreid.layers import *
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import META_ARCH_REGISTRY, build_model, Baseline
from fastreid.utils.checkpoint import Checkpointer
import pdb
from fastreid.modeling.losses.utils import euclidean_dist
import numpy as np
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from sklearn.linear_model import LinearRegression
logger = logging.getLogger(__name__)

def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m

@META_ARCH_REGISTRY.register()
class Distiller(Baseline):
    def __init__(self, cfg):
        super(Distiller, self).__init__(cfg)
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        norm_type     = cfg.MODEL.HEADS.NORM
        # Get teacher model config
        cfg_t = get_cfg()
        
        cfg_t.merge_from_file(cfg.KD.MODEL_CONFIG)
        model_t = build_model(cfg_t)
        logger.info("Teacher model:\n{}".format(model_t))
        # No gradients for teacher model
        for param in model_t.parameters():
            param.requires_grad_(False)

        logger.info("Loading teacher model weights ...")
        Checkpointer(model_t).load(cfg.KD.MODEL_WEIGHTS)
        self.model_t = [model_t.backbone, model_t.heads]
        self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        self.classifier.apply(weights_init_classifier) 
        self.loss = nn.MSELoss()

    def forward(self, batched_inputs, unbatched_inputs):
            # teacher model forward
        if self.training:
            #camids  = batched_inputs["camids"].to(self.device)
            targets = batched_inputs["targets"].to(self.device)
            untargets = unbatched_inputs["targets"].to(self.device)
            
            images = self.preprocess_image(batched_inputs)
            unimages = self.preprocess_image(unbatched_inputs)
            # student model forward
            s_feat = self.backbone(images)
            s_outputs = self.heads(s_feat, targets)

            unimages = self.preprocess_image(unbatched_inputs)
            uns_feat = self.backbone(unimages)
            uns_feat = self.heads.pool_layer(uns_feat)
            uns_feat = self.heads.bottleneck(uns_feat)
            uns_feat = uns_feat[..., 0, 0]
            pred_class_logits = self.classifier.s * F.linear(F.normalize(uns_feat), F.normalize(self.classifier.weight))
            uns_outputs = {"pred_class_logits": pred_class_logits, "features": uns_feat}
            with torch.no_grad():
                #self.updata_parameter()
                t_feat = self.model_t[0](images)
                t_outputs = self.model_t[1](t_feat, targets)
                unt_feat = self.model_t[0](unimages)
                unt_outputs = self.model_t[1](unt_feat, untargets)
            losses = self.losses(s_outputs, t_outputs, uns_outputs, unt_outputs, targets)
            #print(losses)
            return losses
        # Eval mode, just conventional reid feature extraction
        else:
            return super().forward(batched_inputs, unbatched_inputs)


    def losses(self, s_outputs, t_outputs, uns_outputs, unt_outputs, gt_labels):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = super(Distiller, self).losses(s_outputs, gt_labels)
        t_logits = t_outputs["pred_class_logits"].detach() 
        s_logits = s_outputs["pred_class_logits"]
        unt_feat = unt_outputs["features"].detach() 
        uns_feat = uns_outputs["features"]

        unt_logits = unt_outputs["pred_class_logits"].detach() 
        uns_logits = uns_outputs["pred_class_logits"]
        #y = t_logits[:,gt_labels]
        #x = compute_cosine_distance(t_outputs['features'], t_outputs['features'])
        #y = y.view(-1).cpu().numpy().reshape(-1, 1)
        #x = x.view(-1).cpu().numpy().reshape(-1, 1)
        #model = LinearRegression(fit_intercept = True)
        #model.fit(x,y)
        #dist_xy = compute_cosine_distance(t_outputs['features'], unt_outputs['features'])
        #dist_yx = compute_cosine_distance(unt_outputs['features'], t_outputs['features'])
        #last_result1 = torch.tensor(model.coef_[0]).cuda() * dist_xy + torch.tensor(model.intercept_).cuda()
        #last_result2 = torch.tensor(model.coef_[0]).cuda() * dist_yx + torch.tensor(model.intercept_).cuda()
        #pdb.set_trace()
        t_dist = compute_cosine_distance(unt_feat, unt_feat)
        s_dist = compute_cosine_distance(uns_feat, uns_feat)
        #loss_dict = {}
        loss_dict['ukl_loss'] = self.kl_loss(uns_logits, unt_logits, gt_labels, 6)
        #loss_dict['udist_loss'] = self.loss(s_dist, t_dist) #self.cross_entropy_loss(uns_logits, unt_logits, s_logits, t_logits)

        loss_dict["loss_kldiv"] = self.kl_loss(s_logits, t_logits, gt_labels, 16)

        return loss_dict

    @classmethod
    def kl_loss(cls, y_s, y_t, gt_labels, t):       
        p_s = F.log_softmax(y_s / t, dim=1)
        p_t = F.softmax(y_t / t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * (t ** 2) / y_s.shape[0]
        return loss

    def cross_entropy_loss(self, pred_class_outputs, gt_classes):
        num_classes = pred_class_outputs.size(1)
        log_probs = F.log_softmax(pred_class_outputs/3, dim=1)
        with torch.no_grad():
            targets = gt_classes
        loss = (-F.softmax(targets/3, dim = 1) * log_probs).sum(dim=1)
        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        loss = loss.sum() / non_zero_cnt

        return loss

    def updata_parameter(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone.parameters(), self.model_t[0].parameters()):
            param_k.data = param_k.data * 0.99 + param_q.data * 0.01
            
        for param_q, param_k in zip(self.heads.parameters(), self.model_t[1].parameters()):
            param_k.data = param_k.data * 0.99 + param_q.data * 0.01
    