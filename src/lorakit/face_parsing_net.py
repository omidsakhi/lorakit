"""BiSeNet face-parsing network (19-class CelebAMask-HQ).

Vendored from https://github.com/yakhyo/face-parsing (MIT License, Copyright (c)
2024 Yakhyokhuja Valikhujaev).  Only the architecture is reproduced here; the
upstream repo also ships a copy of torchvision's ResNet whose ``resnet34()``
defaults to ``ResNet34_Weights.DEFAULT`` and therefore downloads ImageNet weights
on every construction.  We build the backbone from ``torchvision`` with
``weights=None`` instead, which keeps checkpoint key names identical
(``fpn.backbone.conv1.weight`` ...) while staying offline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet

NUM_CLASSES = 19
_BACKBONE_LAYERS = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3]}


class _FeatureResNet(ResNet):
    """torchvision ResNet that returns the 1/8, 1/16 and 1/32 feature maps."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.norm(self.conv(x)))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1)
        self.conv = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.conv_block(x))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.conv_block(x)
        pool = F.avg_pool2d(feat, [int(size) for size in feat.size()[2:]])
        return torch.mul(feat, self.attention(pool))


class ContextPath(nn.Module):
    def __init__(self, backbone_name: str = "resnet18") -> None:
        super().__init__()
        try:
            layers = _BACKBONE_LAYERS[backbone_name]
        except KeyError as error:
            supported = ", ".join(sorted(_BACKBONE_LAYERS))
            raise ValueError(
                f"Unsupported face-parsing backbone {backbone_name!r}; use one of: {supported}"
            ) from error
        self.backbone = _FeatureResNet(BasicBlock, layers)

        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, kernel_size=3, stride=1)
        self.conv_head16 = ConvBNReLU(128, 128, kernel_size=3, stride=1)
        self.conv_avg = ConvBNReLU(512, 128, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        feat8, feat16, feat32 = self.backbone(x)
        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]
        h32, w32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, [int(size) for size in feat32.size()[2:]])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode="nearest")

        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = self.conv_head32(F.interpolate(feat32_sum, (h16, w16), mode="nearest"))

        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = self.conv_head16(F.interpolate(feat16_sum, (h8, w8), mode="nearest"))

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp: Tensor, fcp: Tensor) -> Tensor:
        feat = self.conv_block(torch.cat([fsp, fcp], dim=1))
        attention = F.avg_pool2d(feat, [int(size) for size in feat.size()[2:]])
        attention = self.sigmoid(self.conv2(self.relu(self.conv1(attention))))
        return torch.mul(feat, attention) + feat


class BiSeNet(nn.Module):
    """Face-parsing BiSeNet; ``forward`` returns the main plus two auxiliary logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, backbone_name: str = "resnet18") -> None:
        super().__init__()
        self.fpn = ContextPath(backbone_name=backbone_name)
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, num_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        height, width = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.fpn(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        outputs = (
            self.conv_out(feat_fuse),
            self.conv_out16(feat_cp8),
            self.conv_out32(feat_cp16),
        )
        return tuple(
            F.interpolate(output, (height, width), mode="bilinear", align_corners=True)
            for output in outputs
        )


def load_bisenet(
    checkpoint_path,
    *,
    backbone_name: str = "resnet34",
    num_classes: int = NUM_CLASSES,
    device: str = "cpu",
) -> BiSeNet:
    """Build a BiSeNet and load a plain upstream ``state_dict`` checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f"face-parsing checkpoint is not a state dict: {checkpoint_path}")
    # Some redistributions wrap the weights; accept the common wrappers.
    for key in ("state_dict", "model"):
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break

    model = BiSeNet(num_classes=num_classes, backbone_name=backbone_name)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model.to(torch.device(device))


__all__ = ["NUM_CLASSES", "BiSeNet", "load_bisenet"]
