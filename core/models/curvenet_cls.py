"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_cls.py
@Time: 2021/01/21 3:10 PM
"""

import torch.nn as nn
import torch.nn.functional as F
from .curvenet_util import *


curve_config = {
        'default': [[100, 5], [100, 5], None, None],
        'long':  [[10, 30], None,  None,  None]
    }

class CurveNet(nn.Module):
    def __init__(self, num_classes=40, k=20, setting='default'): # k=20
        super(CurveNet, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder #* default=0.08-0.08-0.08-0.16
        self.cic11 = CIC(npoint=478, radius=0.08, k=k, in_channels=additional_channel, output_channels=32, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=478, radius=0.08, k=k, in_channels=32, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])
        
        self.cic21 = CIC(npoint=478, radius=0.08, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=478, radius=0.16, k=k, in_channels=128, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        self.conv0 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
            )
        
    def forward(self, xyz):
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        x = self.conv0(l2_points)
        
        return x
