import torch.nn as nn
import torch.nn.functional as F
from ..module.conv import ConvModule
from .fpn import FPN
from torchvision.ops import roi_align
import torch
class res_block(nn.Module):
  def __init__(self,inc,outc,shortcut=True):
    super(res_block, self).__init__()
    self.conv1 = nn.Conv2d(inc,outc,3,1,1)
    self.bn1 = nn.BatchNorm2d(outc)
    self.act = nn.ReLU()
    self.conv2 = nn.Conv2d(outc,outc,3,1,1)
    self.bn2 = nn.BatchNorm2d(outc)
    self.res = shortcut
  def forward(self,x):
    nx = x.shape[1]
    out1 = self.act(self.bn1(self.conv1(x)))
    if self.res:
      out1[:,:nx] = out1[:,:nx] + x
    out = self.act(self.bn2(self.conv2(out1)))
    if self.res:
      out = out + out1
    return out
class PAN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 task=None):
        super(PAN,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, conv_cfg, norm_cfg, activation)
        
        self.transpose_1 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                             kernel_size=(2,2), stride=(2,2))
        self.up_conv_2 = self.conv_block(128,64)
        self.transpose_2 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                             kernel_size=(2,2), stride=(2,2))
        self.up_conv_4 = self.conv_block(64,32)
        self.transpose_3 = nn.ConvTranspose2d(in_channels=32, out_channels=32,
                                             kernel_size=(5,2), stride=(1,2),dilation=(5,1))
        self.up_conv_6 = self.conv_block(32,16)
        self.up_conv_8 = self.conv_block(16,8)
        self.output = nn.Conv2d(in_channels=8, out_channels=2,
                                kernel_size=1)
        self.init_weights()
        self.task=task
    @staticmethod   
    def conv_block(in_channels, out_channels):
      block = nn.Sequential(
              nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1),
              nn.ReLU(),
              nn.Conv2d(in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1),
              nn.ReLU())
      return block

      
    def forward(self, inputs):
    
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        origin_H = 180
        origin_W = 320

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='bilinear')

        # build outputs
        # part 1: from original levels
        inter_outs = [
            laterals[i] for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += F.interpolate(inter_outs[i], scale_factor=0.5, mode='bilinear')
            

        outs = []
        outs.append(inter_outs[0])
        
        outs.extend([
            inter_outs[i] for i in range(1, used_backbone_levels)
        ])
        u5=self.transpose_1(inter_outs[0])
        u5 = self.up_conv_2(u5)
        u7=self.transpose_2(u5)
        u7 = self.up_conv_4(u7)
        u8=self.transpose_3(u7)
        u8 = self.up_conv_6(u8)
        u9 = self.up_conv_8(u8)
        output = self.output(u9)
        if self.task == "multi":
            return tuple(outs),output
        elif self.task == "detection":
            return tuple(outs)
        else :
            return output
