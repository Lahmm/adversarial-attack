import argparse
import os
import random
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One Pixel Attack (Differential Evolution) - Untargeted only")

    # One Pixel / DE 超参（按论文默认设置）
    parser.add_argument("--pixels", type=int, default=1, help="修改像素数量 d，默认1（One Pixel）")
    parser.add_argument("--popsize", type=int, default=400, help="DE 种群规模，默认400")
    parser.add_argument("--iters", type=int, default=50, help="DE 迭代代数，默认50")
    parser.add_argument("--F", type=float, default=0.5, help="DE 缩放系数 F，默认0.5")
    parser.add_argument("--stop-true-prob",type=float,default=0.05,help="早停阈值：真实类概率低于该值且已误分类则停止")
    parser.add_argument("--max-images", type=int, default=100, help="最大攻击样本数量，默认100，-1则攻击所有正确分类样本")
    parser.add_argument("--save-dir", type=str, default="output/onepixel/cifar10_resnet18")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--weights-dir", type=str, default="model_weights")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="dataset")
    return parser.parse_args()


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


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, List[int]]:
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

# 定义 One Pixel Attack（无目标）攻击函数
def one_pixel_attack_untargeted(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    pixels: int = 1,
    popsize: int = 400,
    iters: int = 50,
    F_scale: float = 0.5,
    stop_true_prob: float = 0.05,
    clamp_min: torch.Tensor = None,
    clamp_max: torch.Tensor = None,
) -> torch.Tensor:
    """
    One Pixel Attack（无目标）攻击说明
    攻击输入为：一个黑盒/仅需前向查询的模型，目标样本（此处为inputs，以张量形式读入），对应的标签targets，
            数据标准化参数mean/std，以及可修改像素数pixels；搜索超参数包括种群大小popsize、迭代次数iters、
            差分进化缩放系数F_scale与提前停止阈值stop_true_prob，并通过clamp_min/clamp_max约束输入范围
    输出为：通过修改少量像素点（默认1个像素）得到的对抗样本adv_batch
    """
    model.eval()  # 设置模型为评估模式，关闭dropout等训练特定行为
    device = inputs.device  # 获取输入张量所在设备
    b, c, h, w = inputs.shape  # 获取输入张量的批量大小、通道数、高度和宽度

    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)  # 数据集均值张量，用于反归一化
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)  # 数据集标准差张量，用于反归一化

    def to_unnorm_01(x_norm: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x_norm * std_t + mean_t, 0.0, 1.0)  # 将归一化张量转换回[0,1]范围的图像
    def to_norm(x_01: torch.Tensor) -> torch.Tensor:
        return (x_01 - mean_t) / std_t  # 将[0,1]范围的图像转换回归一化张量

    def clip_candidate(cand: torch.Tensor) -> torch.Tensor:
        cand = cand.clone()  # 克隆候选解，避免修改原始张量
        for p in range(pixels):  # 遍历每个像素点
            xi = 5 * p + 0  # x 坐标索引
            yi = 5 * p + 1  # y 坐标索引
            cand[:, xi] = torch.clamp(cand[:, xi], 0.0, float(w - 1))  # 限制 x 坐标在图像宽度范围内
            cand[:, yi] = torch.clamp(cand[:, yi], 0.0, float(h - 1))  # 限制 y 坐标在图像高度范围内
            cand[:, 5 * p + 2 : 5 * p + 5] = torch.clamp(
                cand[:, 5 * p + 2 : 5 * p + 5], 0.0, 255.0
            )  # 限制 RGB 通道值在有效范围内
        return cand  # 返回裁剪后的候选解
    def init_population() -> torch.Tensor:
        dim = 5 * pixels  # 每个像素点包含5个参数（x坐标、y坐标、R、G、B）
        pop = torch.empty((popsize, dim), device=device)  # 初始化种群张量
        for p in range(pixels):  # 遍历每个像素点
            pop[:, 5 * p + 0] = torch.randint(low=0, high=w, size=(popsize,), device=device).float()
            pop[:, 5 * p + 1] = torch.randint(low=0, high=h, size=(popsize,), device=device).float()
            rgb = torch.normal(mean=128.0, std=127.0, size=(popsize, 3), device=device)
            rgb = torch.clamp(rgb, 0.0, 255.0)
            pop[:, 5 * p + 2 : 5 * p + 5] = rgb
        return clip_candidate(pop)  # 返回初始化种群，确保所有候选解合法

    def apply_candidates(x0_norm: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        n = cand.shape[0]  # 获取候选解数量
        x01 = to_unnorm_01(x0_norm)  # 将归一化张量转换回[0,1]范围的图像
        x01_adv = x01.expand(n, -1, -1, -1).clone()  # 扩展输入图像以匹配候选解数量，并克隆用于修改
        idx = torch.arange(n, device=device)  # 生成索引张量，用于批量操作

        for p in range(pixels):  # 遍历每个像素点
            x_pos = torch.round(cand[:, 5 * p + 0]).clamp(0, w - 1).long()  # 计算 x 坐标索引
            y_pos = torch.round(cand[:, 5 * p + 1]).clamp(0, h - 1).long()  # 计算 y 坐标索引
            rgb01 = torch.clamp(cand[:, 5 * p + 2 : 5 * p + 5] / 255.0, 0.0, 1.0)  # RGB 通道值归一化到 [0,1]

            if c == 1:  # 单通道图像
                x01_adv[idx, 0, y_pos, x_pos] = rgb01[:, 0]  # 修改单通道图像的像素值
            else:  # 多通道图像
                x01_adv[idx, 0, y_pos, x_pos] = rgb01[:, 0]  # 修改 R 通道
                x01_adv[idx, 1, y_pos, x_pos] = rgb01[:, 1]  # 修改 G 通道
                x01_adv[idx, 2, y_pos, x_pos] = rgb01[:, 2]  # 修改 B 通道

        x01_adv = torch.clamp(x01_adv, 0.0, 1.0)  # 限制对抗样本像素值在 [0,1] 范围内
        x_adv_norm = to_norm(x01_adv)  # 将对抗样本转换回归一化张量
        x_adv_norm = torch.max(torch.min(x_adv_norm, clamp_max), clamp_min)  # 裁剪对抗样本到合法输入范围
        return x_adv_norm  # 返回归一化的对抗样本

    @torch.no_grad()
    def evaluate_population(x0_norm: torch.Tensor, true_y: int, cand: torch.Tensor):
        x_adv_norm = apply_candidates(x0_norm, cand)  # 应用候选解生成对抗样本
        logits = model(x_adv_norm)  # 计算模型对对抗样本的输出
        prob = F.softmax(logits, dim=1)  # 计算预测概率分布
        p_true = prob[:, true_y]  # 获取真实类别的预测概率
        preds = prob.argmax(dim=1)  # 获取预测类别
        fit = -p_true  # 适应度为真实类别概率的负值，目标是降低该概率
        return fit, preds, p_true  # 返回适应度、预测类别和真实类别概率
    def sample_distinct_indices(n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.arange(n, device=device)  # 生成索引张量，用于批量操作
        r1 = torch.randint(0, n, (n,), device=device)  # 随机索引 1
        r2 = torch.randint(0, n, (n,), device=device)  # 随机索引 2
        r3 = torch.randint(0, n, (n,), device=device)  # 随机索引 3
        mask = (r1 == idx) | (r2 == idx) | (r3 == idx) | (r1 == r2) | (r1 == r3) | (r2 == r3)
        while mask.any():  # 重新采样，确保索引互不相同
            count = int(mask.sum().item())
            r1[mask] = torch.randint(0, n, (count,), device=device)
            r2[mask] = torch.randint(0, n, (count,), device=device)
            r3[mask] = torch.randint(0, n, (count,), device=device)
            mask = (r1 == idx) | (r2 == idx) | (r3 == idx) | (r1 == r2) | (r1 == r3) | (r2 == r3)
        return r1, r2, r3  # 返回三个互不相同的随机索引

    adv_batch = inputs.clone()  # 克隆输入张量，作为对抗样本批次的初始值

    for i in range(b):  # 遍历批次中的每个样本
        x0 = inputs[i : i + 1].detach()  # 获取当前样本的输入张量
        y_true = int(targets[i].item())  # 获取当前样本的真实标签

        pop = init_population()  # 初始化种群
        fit_vals, preds, p_true = evaluate_population(x0, y_true, pop)  # 评估种群适应度
        best_idx = int(torch.argmax(fit_vals).item())  # 获取适应度最高的个体索引
        best_fit = float(fit_vals[best_idx].item())  # 获取最高适应度值
        best_cand = pop[best_idx].clone()  # 克隆最佳个体

        for _ in range(max(iters, 1)):  # 遍历最大迭代次数
            _, pred_best, p_true_best = evaluate_population(x0, y_true, best_cand.unsqueeze(0))  # 评估当前最佳个体
            if (float(p_true_best[0].item()) <= stop_true_prob) and (int(pred_best[0].item()) != y_true):  # 检查停止条件
                break  # 满足停止条件，退出循环

            r1, r2, r3 = sample_distinct_indices(popsize)  # 采样三个互不相同的随机索引
            trial = pop[r1] + F_scale * (pop[r2] - pop[r3])  # 生成试验个体
            trial = clip_candidate(trial)  # 裁剪试验个体到合法范围

            fit_trial, _, _ = evaluate_population(x0, y_true, trial)  # 评估试验个体适应度
            improve = fit_trial > fit_vals  # 判断试验个体是否优于当前个体
            if improve.any():  # 如果有改进
                pop[improve] = trial[improve]  # 更新种群中的个体
                fit_vals[improve] = fit_trial[improve]  # 更新适应度值

            new_best_idx = int(torch.argmax(fit_vals).item())  # 获取新的最佳个体索引
            new_best_fit = float(fit_vals[new_best_idx].item())  # 获取新的最佳适应度值
            if new_best_fit > best_fit:  # 如果新的适应度更好
                best_fit = new_best_fit  # 更新最佳适应度值
                best_cand = pop[new_best_idx].clone()  # 更新最佳个体
        adv_batch[i : i + 1] = apply_candidates(x0, best_cand.unsqueeze(0)).detach()  # 应用最佳个体生成对抗样本

    return adv_batch.detach()  # 返回对抗样本批次


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
    device = get_device()

    test_loader, num_classes = load_dataset(
        args.dataset,
        batch_size=args.batch_size,
        train=False,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    dataset_key = args.dataset.strip().lower()
    in_channels = DATASET_CHANNELS.get(dataset_key, 3)

    model = build_model(args.model, num_classes=num_classes, in_channels=in_channels, pretrained=False)
    weights_path = resolve_weights_path(args)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    clean_acc, correct_indices = evaluate(model, test_loader, device)
    print(f"Clean acc: {clean_acc:.4f} ({len(correct_indices)}/{len(test_loader.dataset)})")

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
    for inputs, targets in tqdm(attack_loader, desc="OnePixel(DE) Untargeted", unit="batch"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        adv = one_pixel_attack_untargeted(
            model=model,
            inputs=inputs,
            targets=targets,
            mean=mean,
            std=std,
            pixels=max(args.pixels, 1),
            popsize=max(args.popsize, 1),
            iters=max(args.iters, 1),
            F_scale=float(args.F),
            stop_true_prob=float(args.stop_true_prob),
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )

        with torch.no_grad():
            outputs = model(adv)
            _, preds = outputs.max(1)

        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        batch_indices = attack_indices[offset : offset + targets.size(0)]
        save_adversarial_images(adv, mean, std, args.save_dir, batch_indices, targets, preds)
        offset += targets.size(0)

    adv_acc = correct / max(total, 1)
    asr = 1.0 - adv_acc
    print(f"Adv acc: {adv_acc:.4f} ({correct}/{total})")
    print(f"ASR: {asr:.4f} ({total - correct}/{total})")
    print(f"Saved adversarial images to {os.path.abspath(args.save_dir)}")


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    attack(args)


if __name__ == "__main__":
    main()
