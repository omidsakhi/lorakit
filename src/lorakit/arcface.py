"""PyTorch InsightFace ArcFace IResNet backbones used for loss export."""

from pathlib import Path

import torch
from torch import nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)


class _IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                _conv1x1(inplanes, planes, stride), nn.BatchNorm2d(planes, eps=1e-5)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class ArcFaceIResNet(nn.Module):
    """InsightFace ``arcface_torch`` IResNet, fixed to 112px RGB inputs."""

    def __init__(self, layers: list[int], embedding_size: int = 512) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-5)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad_(False)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [_IBasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.features(x)


def iresnet50() -> ArcFaceIResNet:
    """The R50 backbone used by ``ms1mv3_arcface_r50_fp16``."""
    return ArcFaceIResNet([3, 4, 14, 3])


class ArcFaceR50Encoder(nn.Module):
    """Export wrapper that accepts normalized RGB and feeds ArcFace its BGR order."""

    def __init__(self, backbone: ArcFaceIResNet) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("rgb_to_bgr", torch.tensor([2, 1, 0], dtype=torch.long))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.backbone(rgb.index_select(1, self.rgb_to_bgr))


def load_arcface_encoder(path: str | Path) -> nn.Module:
    """Load a frozen ArcFace R50 encoder from TorchScript or ``backbone.pth``."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ArcFace model not found: {model_path}")
    suffix = model_path.suffix.lower()
    loaded: nn.Module | None = None
    if suffix in {".pt", ".pth"}:
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            checkpoint = None
        if (
            isinstance(checkpoint, dict)
            and checkpoint
            and all(isinstance(key, str) for key in checkpoint)
            and any(key.startswith("conv1.") or key.startswith("layer1.") for key in checkpoint)
        ):
            backbone = iresnet50().float().eval()
            incompatible = backbone.load_state_dict(checkpoint, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(f"ArcFace backbone is incompatible: {model_path}")
            loaded = ArcFaceR50Encoder(backbone).eval()
    if loaded is None:
        loaded = torch.jit.load(str(model_path), map_location="cpu").eval()
    for parameter in loaded.parameters():
        parameter.requires_grad_(False)
    return loaded


def normalize_arcface_rgb(images: torch.Tensor) -> torch.Tensor:
    """Map RGB in ``[0, 1]`` or ``[-1, 1]`` into ArcFace's expected ``[-1, 1]`` RGB."""
    images = images.float()
    if float(images.min()) >= -1e-3 and float(images.max()) <= 1.0 + 1e-3:
        images = images * 2.0 - 1.0
    return images.clamp(-1.0, 1.0)
