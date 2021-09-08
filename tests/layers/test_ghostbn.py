import unittest

import torch
from fastreid.layers import BatchNorm, GhostBatchNorm
import torch.nn.functional as F


class TestGhostBN(unittest.TestCase):
    def test_ghostbn(self):
        dummy_inputs = torch.randn(64, 3, 64, 32)

        mean_sp1 = torch.mean(dummy_inputs[::2], dim=[0, 2, 3])
        var_sp1 = torch.var(dummy_inputs[::2], dim=[0, 2, 3])
        std_sp1 = torch.std(dummy_inputs[::2], dim=[0, 2, 3])
        mean_sp2 = torch.mean(dummy_inputs[1::2], dim=[0, 2, 3])

        mean = 0.1 * torch.mean(dummy_inputs, dim=[0, 2, 3])
        var = 0.9 + 0.1 * torch.var(dummy_inputs, dim=[0, 2, 3])
        ghost_bn = GhostBatchNorm(3, num_splits=2)

        outputs = ghost_bn(dummy_inputs)
        ghost_mean = ghost_bn.running_mean
        ghost_var = ghost_bn.running_var

        self.assertAlmostEqual(ghost_mean[0], mean[0], delta=1e-6)
        self.assertAlmostEqual(ghost_mean[1], mean[1], delta=1e-6)
        self.assertAlmostEqual(ghost_mean[2], mean[2], delta=1e-6)
        self.assertAlmostEqual(ghost_var[0], var[0], delta=1e-6)
        self.assertAlmostEqual(ghost_var[1], var[1], delta=1e-6)
        self.assertAlmostEqual(ghost_var[2], var[2], delta=1e-6)


if __name__ == '__main__':
    unittest.main()
