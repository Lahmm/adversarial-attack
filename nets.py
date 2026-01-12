import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, vgg16, densenet121

def build_resnet18(num_classes: int, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """
    直接使用 torchvision 的 resnet18 实现，并适配分类数和输入通道数。
    """
    model = resnet18(weights="DEFAULT" if pretrained else None)

    # 1) 修改第一层卷积以适配输入通道数
    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=in_channels, #改变输入通道数
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

    # 2) 修改最后全连接层以适配类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes) #改变输出类别数

    return model


def build_resnet34(num_classes: int, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """
    直接使用 torchvision 的 resnet34 实现，并适配分类数和输入通道数。
    """
    model = resnet34(weights="DEFAULT" if pretrained else None)

    # 1) 修改第一层卷积以适配输入通道数
    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=in_channels, #改变输入通道数
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

    # 2) 修改最后全连接层以适配类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes) #改变输出类别数

    return model

def build_vgg16(num_classes: int, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """
    适配 CIFAR10(32x32) / MNIST(28x28) 的 VGG 模型。
    """
    model = vgg16(weights="DEFAULT" if pretrained else None)

    # 1) 让 VGG 的所有池化在小图输入下不发生尺寸坍塌（尤其 MNIST 28x28）
    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.MaxPool2d):
            model.features[i] = nn.MaxPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                ceil_mode=True,
            )

    # 2) 修改第一层卷积以适配输入通道数，并在可能时迁移预训练权重
    if in_channels != 3:
        old_conv: nn.Conv2d = model.features[0]
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        with torch.no_grad():
            if pretrained:
                # old_conv.weight: [64, 3, 3, 3]
                w = old_conv.weight

                if in_channels == 1:
                    # 灰度输入：用 RGB 均值初始化（更稳）
                    new_conv.weight.copy_(w.mean(dim=1, keepdim=True))
                else:
                    # 多通道：循环拷贝，并做缩放保持量级
                    for c in range(in_channels):
                        new_conv.weight[:, c:c+1].copy_(w[:, (c % 3):(c % 3) + 1])
                    new_conv.weight.mul_(3.0 / float(in_channels))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)

        model.features[0] = new_conv

    # 3) 小图分类头：全局池化 + 轻量分类器（避免 ImageNet VGG 的 4096 大头不匹配）
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
    )

    # 4) 初始化新分类层（无论是否 pretrained，都需要重新训最后层）
    last_fc: nn.Linear = model.classifier[-1]
    nn.init.normal_(last_fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(last_fc.bias)

    return model


def build_densenet121(num_classes: int, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """
    适配 CIFAR10(32x32) / MNIST(28x28) 的 DenseNet121。
    """
    model = densenet121(weights="DEFAULT" if pretrained else None)

    # 1) 小图 stem：替换 conv0，并把 stride 改为 1，kernel 改为 3
    # DenseNet121: features.conv0 / features.norm0 / features.relu0 / features.pool0
    old_conv0: nn.Conv2d = model.features.conv0
    new_conv0 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv0.out_channels,   # 通常为 64
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # 初始化策略：
    # - 若 stem 结构变化（7x7 -> 3x3），不强行从预训练 conv0 迁移（形状也不匹配）
    # - 直接用 Kaiming 初始化，通常对 CIFAR/MNIST 更稳
    nn.init.kaiming_normal_(new_conv0.weight, mode="fan_out", nonlinearity="relu")

    model.features.conv0 = new_conv0

    # 2) 移除/旁路 maxpool（小图不建议过早下采样）
    model.features.pool0 = nn.Identity()

    # （可选但通常有益）保持 norm0/relu0 不变即可

    # 3) 修改最后分类层以适配类别数
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    # 初始化新分类层
    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.classifier.bias)

    return model