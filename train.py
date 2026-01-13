import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from nets import (
    build_densenet121,
    build_resnet18,
    build_resnet34,
)
from utils import get_device, load_dataset


MODEL_BUILDERS = {
    "resnet18": build_resnet18,
    "resnet34": build_resnet34,
    "densenet121": build_densenet121,
}

DATASET_CHANNELS = {
    "mnist": 1,
    "cifar10": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classifier model.")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--save-dir", type=str, default="model_weights")
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


def build_model(name: str, num_classes: int, in_channels: int, pretrained: bool) -> torch.nn.Module:
    key = name.strip().lower()
    if key not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model name: {name}")
    
    return MODEL_BUILDERS[key](
        num_classes=num_classes, in_channels=in_channels, pretrained=pretrained
    )


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train(args: argparse.Namespace) -> str:
    device = get_device()
    os.makedirs(args.data_dir, exist_ok=True)

    loader, num_classes = load_dataset(
        args.dataset,
        batch_size=args.batch_size,
        train=True,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    dataset_key = args.dataset.strip().lower()
    in_channels = DATASET_CHANNELS.get(dataset_key, 3)

    model = build_model(
        args.model,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=args.pretrained,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{dataset_key}_{args.model.strip().lower()}.pth"
    save_path = os.path.join(save_dir, save_name)
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, loader, device, optimizer, criterion)
        print(f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f} - acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

    if best_acc < 0:
        torch.save(model.state_dict(), save_path)
    return save_path


def main() -> None:
    args = parse_args()
    save_path = train(args)
    print(f"Saved weights to {save_path}")


if __name__ == "__main__":
    main()
