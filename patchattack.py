import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from train import DATASET_CHANNELS, build_model
from utils import get_device


DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial patch attack (square only).")
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--weights-dir", type=str, default="model_weights")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--save-dir", type=str, default="output/patch/cifar10_resnet34")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-type", type=str, default="square")
    parser.add_argument("--patch-size", type=float, default=0.1, help="对抗块尺寸占比，默认0.1")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--conf-target", type=float, default=0.9)
    parser.add_argument("--max-count", type=int, default=500)
    parser.add_argument("--max-images",type=int,default=5,help="Max number of correctly classified samples to attack (-1 for all)",)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1338)
    return parser.parse_args()

# 解析模型权重路径
def resolve_weights_path(args: argparse.Namespace) -> str:
    if args.weights:
        return os.path.abspath(args.weights)
    dataset_key = args.dataset.strip().lower()
    model_key = args.model.strip().lower()
    filename = f"{dataset_key}_{model_key}.pth"
    return os.path.abspath(os.path.join(args.weights_dir, filename))


def get_norm_stats(dataset_key: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    key = dataset_key.strip().lower()
    if key not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset_key}")
    return DATASET_STATS[key]


def build_clamp_tensors(
    mean: Tuple[float, ...], std: Tuple[float, ...], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    clamp_min = (0.0 - mean_t) / std_t
    clamp_max = (1.0 - mean_t) / std_t
    return clamp_min, clamp_max

# 构建测试数据集
def build_test_dataset(data_dir: str) -> datasets.CIFAR10:
    mean, std = DATASET_STATS["cifar10"]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return test_dataset

# 构建攻击数据加载器
def build_attack_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

# 构建数据加载器
def build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )

# 初始化正方形对抗块
def init_patch_square(image_size: int, patch_size: float) -> Tuple[np.ndarray, Tuple[int, ...]]:
    noise_size = int(image_size * image_size * patch_size)
    noise_dim = max(1, int(noise_size**0.5))
    patch = np.random.rand(1, 3, noise_dim, noise_dim).astype(np.float32)
    return patch, patch.shape

def square_transform(
    patch: np.ndarray, data_shape: Tuple[int, ...], patch_shape: Tuple[int, ...], image_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.zeros(data_shape, dtype=np.float32)
    m_size = patch_shape[-1]

    for i in range(x.shape[0]):
        rot = np.random.choice(4)
        for j in range(patch[i].shape[0]):
            patch[i][j] = np.rot90(patch[i][j], rot)

        random_x = np.random.choice(image_size)
        while random_x + m_size > x.shape[-1]:
            random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        while random_y + m_size > x.shape[-1]:
            random_y = np.random.choice(image_size)

        x[i, 0, random_x : random_x + m_size, random_y : random_y + m_size] = patch[i, 0]
        x[i, 1, random_x : random_x + m_size, random_y : random_y + m_size] = patch[i, 1]
        x[i, 2, random_x : random_x + m_size, random_y : random_y + m_size] = patch[i, 2]

    mask = np.copy(x)
    mask[mask != 0] = 1.0
    return x, mask

# 提取对抗块
def extract_patch(masked_patch: np.ndarray, patch_dim: int) -> np.ndarray:
    mask = masked_patch[0, 0] != 0
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return masked_patch[:, :, :patch_dim, :patch_dim]
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return masked_patch[:, :, y0:y1, x0:x1]

# 保存对抗样本和原始样本对
def save_pair(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    save_dir: str,
    split: str,
    index: int,
    ori_label: int,
    adv_label: int,
) -> None:
    mean_t = torch.tensor(mean, device=adversarial.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=adversarial.device).view(1, -1, 1, 1)
    ori = torch.clamp(original * std_t + mean_t, 0.0, 1.0)
    adv = torch.clamp(adversarial * std_t + mean_t, 0.0, 1.0)

    split_dir = os.path.join(save_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    save_image(ori, os.path.join(split_dir, f"{index:06d}_{ori_label}_original.png"))
    save_image(adv, os.path.join(split_dir, f"{index:06d}_{adv_label}_adversarial.png"))

# 定义对抗补丁攻击函数
def attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    patch: torch.Tensor,
    mask: torch.Tensor,
    clamp_min: torch.Tensor,
    clamp_max: torch.Tensor,
    target: int,
    conf_target: float,
    max_count: int,
    step_size: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()  # 将模型切换到评估模式，避免训练态的随机性影响攻击效果
    with torch.no_grad():
        target_prob = F.softmax(model(x), dim=1)[0][target].item()  # 计算目标类别的初始概率

    adv_x = (1 - mask) * x + mask * patch  # 应用对抗补丁生成初始对抗样本
    count = 0  # 初始化迭代计数器

    while target_prob < conf_target and count < max_count:  # 当目标概率未达到阈值且未超过最大迭代次数时循环
        count += 1  # 迭代计数器加一
        adv_x = adv_x.clone().detach().requires_grad_(True)  # 克隆对抗样本并启用梯度计算
        adv_out = model(adv_x)  # 计算模型对对抗样本的输出
        loss = -adv_out[0][target]  # 计算目标类别的负对数概率作为损失
        model.zero_grad()  # 清零模型梯度
        loss.backward()  # 反向传播计算梯度

        adv_grad = adv_x.grad.detach()  # 获取对抗样本的梯度
        patch = patch - step_size * adv_grad  # 更新对抗补丁
        adv_x = (1 - mask) * x + mask * patch  # 重新应用对抗补丁生成对抗样本
        adv_x = torch.max(torch.min(adv_x, clamp_max), clamp_min)  # 裁剪对抗样本到合法输入范围

        with torch.no_grad():
            target_prob = F.softmax(model(adv_x), dim=1)[0][target].item()  # 更新目标类别的概率

    return adv_x.detach(), patch.detach()  # 返回最终对抗样本和对抗补丁

# 评估模型在数据集上的准确率
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, List[int]]:
    model.eval()
    correct = 0
    total = 0
    correct_indices: List[int] = []
    index_offset = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            matches = predicted.eq(targets)

            correct += matches.sum().item()
            total += targets.size(0)

            for i in range(targets.size(0)):
                if matches[i]:
                    correct_indices.append(index_offset + i)

            index_offset += targets.size(0)

    accuracy = correct / max(total, 1)
    return accuracy, correct_indices

# 主攻击流程
def main() -> None:
    args = parse_args()
    dataset_key = args.dataset.strip().lower()
    if dataset_key != "cifar10":
        raise ValueError("Only cifar10 is supported without extra changes.")
    if args.patch_type.strip().lower() != "square":
        raise ValueError("Only square patch is supported when scipy is unavailable.")
    if args.batch_size != 1:
        raise ValueError("Batch size must be 1 for patch updates.")

    device = get_device()

    weights_path = resolve_weights_path(args)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    test_dataset = build_test_dataset(args.data_dir)
    test_loader = build_loader(
        test_dataset, args.batch_size, args.num_workers, args.seed, shuffle=False
    )

    num_classes = 10
    in_channels = DATASET_CHANNELS.get(dataset_key, 3)
    model = build_model(
        args.model, num_classes=num_classes, in_channels=in_channels, pretrained=False
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    mean, std = get_norm_stats(dataset_key)
    clamp_min, clamp_max = build_clamp_tensors(mean, std, device)

    clean_acc, correct_indices = evaluate(model, test_loader, device)
    print(
        f"Clean acc: {clean_acc:.4f} ({len(correct_indices)}/{len(test_loader.dataset)})"
    )
    if not correct_indices:
        print("No correct samples found.")
        return

    if args.max_images > 0:
        attack_indices = correct_indices[: args.max_images]
    else:
        attack_indices = correct_indices

    subset = Subset(test_loader.dataset, attack_indices)
    attack_loader = build_attack_loader(
        subset, args.batch_size, args.num_workers
    )

    os.makedirs(args.save_dir, exist_ok=True)

    total = 0
    adv_correct = 0
    progress = tqdm(attack_loader, desc="Attack", unit="img")
    for batch_idx, (data, labels) in enumerate(progress):
        data = data.to(device)
        labels = labels.to(device)

        data_shape = tuple(data.shape)
        patch, patch_shape = init_patch_square(args.image_size, args.patch_size)
        patch_np, mask_np = square_transform(patch, data_shape, patch_shape, args.image_size)
        patch_t = torch.from_numpy(patch_np).to(device)
        mask_t = torch.from_numpy(mask_np).to(device)

        adv_x, patch_t = attack(
            model,
            data,
            patch_t,
            mask_t,
            clamp_min,
            clamp_max,
            args.target,
            args.conf_target,
            args.max_count,
            args.step_size,
        )

        with torch.no_grad():
            adv_label = model(adv_x).argmax(1)[0].item()
            ori_label = labels[0].item()

        total += 1
        if adv_label == ori_label:
            adv_correct += 1

        sample_index = attack_indices[batch_idx]
        save_pair(
            data,
            adv_x,
            mean,
            std,
            args.save_dir,
            "test",
            sample_index,
            ori_label,
            adv_label,
        )

    adv_acc = adv_correct / max(total, 1)
    print(f"Adversarial acc: {adv_acc:.4f} ({adv_correct}/{total})")


if __name__ == "__main__":
    main()
