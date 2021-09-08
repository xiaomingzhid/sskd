# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .activation import *
from .arc_softmax import ArcSoftmax
from .circle_softmax import CircleSoftmax
from .cos_softmax import CosSoftmax
from .batch_drop import BatchDrop
from .batch_norm import *
from .context_block import ContextBlock
from .frn import FRN, TLU
from .non_local import Non_local
from .pooling import *
from .helpers import to_ntuple, to_2tuple, to_3tuple, to_4tuple, make_divisible
from .se_layer import SELayer
from .splat import SplAtConv2d, DropBlock2D
from .gather_layer import GatherLayer
from .drop import DropPath, DropBlock2d, drop_block_2d, drop_path
from .weight_init import (
    trunc_normal_, variance_scaling_, lecun_normal_, weights_init_kaiming, weights_init_classifier
)