import argparse
import os
import warnings
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision.utils import save_image

from train import DATASET_CHANNELS, build_model
from utils import get_device, load_dataset

# 数据集归一化参数
DATASET_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}


# 定义命令行参数解析
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PGD attack (Linf)")
    # PGD 约束半径（Linf），与 FGSM 的 epsilon 含义一致
    parser.add_argument("--epsilon", type=float, default=0.031, help="Linf 约束半径，默认8/255")
    # PGD 步长（每步更新幅度）
    parser.add_argument("--alpha", type=float, default=0.007, help="PGD步长，默认约2/255")
    # PGD 迭代步数
    parser.add_argument("--steps", type=int, default=50, help="PGD迭代步数，默认10")
    parser.add_argument("--random-start", action="store_true", help="是否进行随机初始化（默认关闭）")
    parser.add_argument("--max-images",type=int,default=100,help="最大攻击样本数量，默认100，设置为-1则攻击所有正确分类样本")
    parser.add_argument("--save-dir",type=str,default="output/pgd/cifar10_resnet18")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--weights-dir", type=str, default="model_weights")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--use-cuda", action="store_true")
    return parser.parse_args()


# 组装模型权重路径
def resolve_weights_path(args: argparse.Namespace) -> str:
    if args.weights:
        return os.path.abspath(args.weights)
    dataset_key = args.dataset.strip().lower()
    model_key = args.model.strip().lower()
    filename = f"{dataset_key}_{model_key}.pth"
    return os.path.abspath(os.path.join(args.weights_dir, filename))


# 数据集归一化参数获取
def get_norm_stats(dataset_key: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    key = dataset_key.strip().lower()
    if key not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset_key}")
    return DATASET_STATS[key]


# 规范 normalized 空间张量范围
def build_clamp_tensors(
    mean: Tuple[float, ...], std: Tuple[float, ...], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    clamp_min = (0.0 - mean_t) / std_t
    clamp_max = (1.0 - mean_t) / std_t
    return clamp_min, clamp_max


# 对 test 数据集进行评估，返回准确率和正确样本索引
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


def pgd_attack(model: torch.nn.Module,inputs: torch.Tensor,targets: torch.Tensor,epsilon: float,alpha: float,steps: int,
    random_start: bool,
    clamp_min: torch.Tensor,
    clamp_max: torch.Tensor,
) -> torch.Tensor:
    """
    PGD(Linf)
    输入：白盒模型、inputs、targets、epsilon(半径)、alpha(步长)、steps(迭代步数)、random_start
    输出：对抗样本 adv（在 normalized 空间中，且满足像素范围与Linf约束）
    """
    model.eval()

    #准备干净样本基准 选择随机启动
    x0 = inputs.detach()
    if random_start:
        adv = x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)
        adv = torch.max(torch.min(adv, clamp_max), clamp_min)
    else:
        adv = x0.clone()
    # 主迭代循环
    for _ in range(max(steps, 1)):
        adv = adv.clone().detach().requires_grad_(True) # 确保对抗样本可求梯度

        outputs = model(adv) #获取模型预测结果
        loss = F.cross_entropy(outputs, targets) # 计算真实标签与预测标签之间的交叉熵损失

        model.zero_grad()
        if adv.grad is not None:
            adv.grad.zero_()
        loss.backward()

        # 梯度上升（最大化 loss）
        adv = adv + alpha * adv.grad.sign()

        # 1) 投影到 L∞ ball: ||adv - x0||_∞ <= epsilon
        delta = torch.clamp(adv - x0, min=-epsilon, max=epsilon)
        adv = x0 + delta

        # 2) clamp 到合法像素范围（normalized 空间下）
        adv = torch.max(torch.min(adv, clamp_max), clamp_min)

    return adv.detach()



def save_adversarial_images(
    adv: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    save_dir: str,
    indices: List[int],
    labels: torch.Tensor,
    preds: torch.Tensor,
) -> None:
    mean_t = torch.tensor(mean, device=adv.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=adv.device).view(1, -1, 1, 1)
    imgs = adv * std_t + mean_t
    imgs = torch.clamp(imgs, 0.0, 1.0).cpu()

    for i in range(imgs.size(0)):
        filename = f"{indices[i]:06d}_label{int(labels[i])}_pred{int(preds[i])}.png"
        save_path = os.path.join(save_dir, filename)
        save_image(imgs[i], save_path)


def attack(args: argparse.Namespace) -> None:
    device = get_device(args.use_cuda)

    test_loader, num_classes = load_dataset(
        args.dataset,
        batch_size=args.batch_size,
        train=False,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    dataset_key = args.dataset.strip().lower()
    in_channels = DATASET_CHANNELS.get(dataset_key, 3)

    model = build_model(
        args.model, num_classes=num_classes, in_channels=in_channels, pretrained=False
    )
    weights_path = resolve_weights_path(args)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    clean_acc, correct_indices = evaluate(model, test_loader, device)
    print(
        f"Clean acc: {clean_acc:.4f} ({len(correct_indices)}/{len(test_loader.dataset)})"
    )

    if not correct_indices:
        print("No correct samples found for attack.")
        return

    if args.max_images > 0:
        attack_indices = correct_indices[: args.max_images]
    else:
        attack_indices = correct_indices

    subset = Subset(test_loader.dataset, attack_indices)
    attack_loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    mean, std = get_norm_stats(dataset_key)
    clamp_min, clamp_max = build_clamp_tensors(mean, std, device)

    os.makedirs(args.save_dir, exist_ok=True)

    total = 0
    correct = 0
    offset = 0

    model.eval()
    for inputs, targets in tqdm(attack_loader, desc="PGD", unit="batch"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        adv = pgd_attack(
            model=model,
            inputs=inputs,
            targets=targets,
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.steps,
            random_start=True,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )

        with torch.no_grad():
            outputs = model(adv)
            _, preds = outputs.max(1)

        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        batch_indices = attack_indices[offset : offset + targets.size(0)]
        save_adversarial_images(
            adv, mean, std, args.save_dir, batch_indices, targets, preds
        )
        offset += targets.size(0)

    adv_acc = correct / max(total, 1)
    asr = 1.0 - adv_acc
    print(f"PGD acc: {adv_acc:.4f} ({correct}/{total})")
    print(f"ASR: {asr:.4f} ({total - correct}/{total})")
    print(f"Saved adversarial images to {os.path.abspath(args.save_dir)}")


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    attack(args)


if __name__ == "__main__":
    main()
