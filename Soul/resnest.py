##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""

import torch
from resnet import ResNet, Bottleneck

def resnest50(pretrained=False, root="/home/xjy/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/code/OCTA-Net/", **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    model_path="/home/xjy/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/code/OCTA-Net/resnest50-528c19ca.pth"  # /home/imed/OCT-A_segmentation/
    if pretrained:
        model.load_state_dict(torch.load(model_path))
    return model

def resnest101(pretrained=False, root='/home/xjy/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/code/OCTA-Net/', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    model_path = "/home/xjy/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/code/OCTA-Net/resnest101-22405ba7.pth"
    if pretrained:
        model.load_state_dict(torch.load(model_path))
    return model

def resnest200(pretrained=False, root='/home/xjy/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/code/OCTA-Net/', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    model_path = "/home/xjy/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/code/OCTA-Net/"
    if pretrained:
        model.load_state_dict(torch.load(model_path))
    return model

def resnest269(pretrained=False, root='/home/xjy/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/code/OCTA-Net/', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    model_path = "/home/xjy/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/code/OCTA-Net/resnest269-0cc87c48.pth"
    if pretrained:
        model.load_state_dict(torch.load(model_path))
    return model