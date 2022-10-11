from collections import OrderedDict
from typing import Any, Dict

from torch import nn, Tensor
from torch.nn import functional as F

import torchvision

"""
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
"""
class kd_model(nn.Module):
    def __init__(self, num_classes: int = 19):
        super(kd_model, self).__init__()
        self.backbone = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=False).backbone
        self.classifier = LRASPPHead(low_channels=40, high_channels=960, num_classes=num_classes, inter_channels=128)
        self.conv = nn.Conv2d(960, 2048, 1, bias=False)
        
    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(out, size=input.shape[-2:], mode="bilinear", align_corners=False)

        #if self.training:
        intermidia = F.interpolate(features["high"], size=(90,90), mode="bilinear", align_corners=False)
        intermidia = self.conv(intermidia)
        return out, intermidia

        #return out


class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AvgPool2d(kernel_size=(8,8), stride=(2,3)),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)

