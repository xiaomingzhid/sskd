# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from fastreid.modeling.losses import *
from fastreid.modeling.meta_arch import Baseline
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class CamAwareBaseline(Baseline):

    def forward(self, batched_inputs):
        outputs = super().forward(batched_inputs)
        if self.training:
            camids = batched_inputs["camids"].long().to(self.device)
            outputs["camids"] = camids
            return outputs
        else:
            return outputs

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        outputs         = outs["outputs"]
        cam_labels      = outs["camids"]
        cam_cls_outputs = outputs["cam_cls_outputs"]
        # fmt: on

        loss_dict = super().losses(outs)
        # Camera-aware loss
        loss_dict['loss_cam'] = cross_entropy_loss(
            cam_cls_outputs, cam_labels, 0.1
        ) * 0.1
        return loss_dict
