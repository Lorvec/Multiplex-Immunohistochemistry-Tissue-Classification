"""Exact proposed architectures and comparator constructors used in the paper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.pool(x)
        gate = F.silu(self.reduce(gate), inplace=True)
        gate = torch.sigmoid(self.expand(gate))
        return x * gate


class MultiScaleUIB(nn.Module):
    """Inverted bottleneck with 3x3/5x5 depthwise branches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 3.0,
        multi_scale: bool = True,
    ) -> None:
        super().__init__()
        hidden = make_divisible(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.multi_scale = bool(multi_scale)
        self.expand = ConvBNAct(in_channels, hidden, kernel_size=1)

        if self.multi_scale:
            first = hidden // 2
            second = hidden - first
            self.first_channels = first
            self.dw3 = ConvBNAct(
                first,
                first,
                kernel_size=3,
                stride=stride,
                groups=first,
            )
            self.dw5 = ConvBNAct(
                second,
                second,
                kernel_size=5,
                stride=stride,
                groups=second,
            )
        else:
            self.dw = ConvBNAct(
                hidden,
                hidden,
                kernel_size=3,
                stride=stride,
                groups=hidden,
            )

        self.se = SqueezeExcite(hidden, reduction=4)
        self.project = ConvBNAct(
            hidden, out_channels, kernel_size=1, activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.expand(x)
        if self.multi_scale:
            x1, x2 = torch.split(
                x,
                [self.first_channels, x.shape[1] - self.first_channels],
                dim=1,
            )
            x = torch.cat([self.dw3(x1), self.dw5(x2)], dim=1)
        else:
            x = self.dw(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x


class GlobalLocalSingleHeadMixer(nn.Module):
    """Tested attention ablation; excluded from the frozen proposed family."""

    def __init__(self, channels: int, attention_ratio: float = 0.25) -> None:
        super().__init__()
        attention_channels = min(
            channels,
            make_divisible(max(8, channels * attention_ratio)),
        )
        self.attention_channels = attention_channels
        self.scale = attention_channels**-0.5
        self.local = ConvBNAct(
            channels,
            channels,
            kernel_size=3,
            groups=channels,
            activation=False,
        )
        self.qkv = nn.Conv2d(
            attention_channels, attention_channels * 3, 1, bias=False
        )
        self.project = ConvBNAct(
            channels, channels, kernel_size=1, activation=False
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        local = self.local(x)
        attention_input = x[:, : self.attention_channels]
        q, k, v = self.qkv(attention_input).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)
        weights = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(
            batch, self.attention_channels, height, width
        )
        mixed = torch.cat(
            [
                local[:, : self.attention_channels] + attended,
                local[:, self.attention_channels :],
            ],
            dim=1,
        )
        return x + self.gamma * self.project(mixed)


MODERN_FAMILY_CONFIGS = {
    "nano": {
        "widths": (24, 40, 80, 160),
        "depths": (1, 2, 3, 1),
        "expand_ratio": 3.0,
        "attention_stages": (3,),
        "dropout": 0.15,
    },
    "tiny": {
        "widths": (24, 48, 96, 192),
        "depths": (1, 2, 3, 2),
        "expand_ratio": 3.0,
        "attention_stages": (2, 3),
        "dropout": 0.20,
    },
    "small": {
        "widths": (32, 64, 128, 256),
        "depths": (2, 3, 4, 2),
        "expand_ratio": 4.0,
        "attention_stages": (2, 3),
        "dropout": 0.25,
    },
}


class ModernMIHICNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        variant: str = "tiny",
        use_attention: bool = False,
        multi_scale: bool = True,
    ) -> None:
        super().__init__()
        if variant not in MODERN_FAMILY_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}")
        cfg = MODERN_FAMILY_CONFIGS[variant]
        widths = cfg["widths"]
        depths = cfg["depths"]
        expand_ratio = cfg["expand_ratio"]
        attention_stages = set(
            cfg["attention_stages"] if use_attention else ()
        )

        self.variant = variant
        self.use_attention = bool(use_attention)
        self.multi_scale = bool(multi_scale)
        self.stem = ConvBNAct(3, widths[0], kernel_size=3, stride=2)

        stages: list[nn.Module] = []
        mixers: list[nn.Module] = []
        in_channels = widths[0]
        for stage_index, (out_channels, depth) in enumerate(
            zip(widths, depths)
        ):
            blocks: list[nn.Module] = []
            for block_index in range(depth):
                blocks.append(
                    MultiScaleUIB(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2 if block_index == 0 else 1,
                        expand_ratio=expand_ratio,
                        multi_scale=multi_scale,
                    )
                )
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
            if stage_index in attention_stages:
                mixers.append(GlobalLocalSingleHeadMixer(out_channels))
            else:
                mixers.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.mixers = nn.ModuleList(mixers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(widths[-1], num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage, mixer in zip(self.stages, self.mixers):
            x = mixer(stage(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class TinyCNNClean(nn.Module):
    """Clean single-forward implementation of the lightweight control."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for index, out_channels in enumerate((32, 64, 128, 256)):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if index < 3:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).flatten(1))


def count_parameters(model: nn.Module) -> int:
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )


def unwrap_logits(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        tensors = [item for item in output if torch.is_tensor(item)]
        if not tensors:
            raise TypeError("Model returned a tuple/list without tensors")
        return torch.stack(tensors, dim=0).mean(dim=0)
    if isinstance(output, dict):
        for key in ("logits", "out", "prediction"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"Unsupported model output type: {type(output)}")
