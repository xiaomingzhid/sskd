import unittest

import torch

import sys
sys.path.append('.')
from fastreid.config import cfg
from fastreid.modeling.backbones import build_resnet_backbone
from fastreid.modeling.meta_arch import build_model


class MyTestCase(unittest.TestCase):
    def test_se_resnet101(self):
        cfg.MODEL.BACKBONE.NAME = 'build_resnet_backbone'
        cfg.MODEL.BACKBONE.DEPTH = '50x'
        cfg.MODEL.BACKBONE.WITH_IBN = False
        cfg.MODEL.BACKBONE.WITH_SE = False
        cfg.MODEL.BACKBONE.PRETRAIN_PATH = "/export/home/lxy/.cache/torch/checkpoints/resnet50-19c8e357.pth"

        net1 = build_model(cfg)
        net1.eval()

        # net2.cuda()
        x = torch.randn(64, 3, 256, 128)
        while True:
            net1(x)
        # y2 = net2(x)
        # assert y1.sum() == y2.sum(), 'train mode problem'
        # net1.eval()
        # net2.eval()
        # y1 = net1(x)
        # y2 = net2(x)
        # assert y1.sum() == y2.sum(), 'eval mode problem'


if __name__ == '__main__':
    unittest.main()
