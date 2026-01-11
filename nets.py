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
    直接使用 torchvision 的 vgg16 实现，并适配分类数和输入通道数。
    """
    model = vgg16(weights="DEFAULT" if pretrained else None)

    # 1) 修改第一层卷积以适配输入通道数
    if in_channels != 3:
        old_conv = model.features[0]  # VGG16 的第一层是 Conv2d(3, 64, 3, 1, 1)
        model.features[0] = nn.Conv2d(
            in_channels=in_channels,  # 改变输入通道数
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

    # 2) 修改最后分类层以适配类别数
    # VGG 的 classifier 最后一层是 Linear(4096, 1000)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)  # 改变输出类别数

    return model

def build_densenet121(num_classes: int, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """
    直接使用 torchvision 的 densenet121 实现，并适配分类数和输入通道数。
    """
    model = densenet121(weights="DEFAULT" if pretrained else None)

    # 1) 修改第一层卷积以适配输入通道数
    # DenseNet 的第一层卷积在 features.conv0
    if in_channels != 3:
        old_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            in_channels=in_channels,  # 改变输入通道数
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

    # 2) 修改最后分类层以适配类别数
    # DenseNet 的分类层是 classifier（Linear）
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)  # 改变输出类别数

    return model