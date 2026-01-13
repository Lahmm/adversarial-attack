import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, densenet121

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


def build_lenet(num_classes: int, in_channels: int = 1, pretrained: bool = False) -> nn.Module:
    """
    LeNet-style network for MNIST-sized inputs (28x28).
    """
    if pretrained:
        raise ValueError("LeNet does not support pretrained weights.")

    model = nn.Sequential(
        nn.Conv2d(in_channels, 6, kernel_size=5),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120),
        nn.ReLU(inplace=True),
        nn.Linear(120, 84),
        nn.ReLU(inplace=True),
        nn.Linear(84, num_classes),
    )

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