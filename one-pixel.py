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
    parser.add_argument("--iters", type=int, default=100, help="DE 迭代代数，默认100")
    parser.add_argument("--F", type=float, default=0.5, help="DE 缩放系数 F，默认0.5")
    parser.add_argument("--stop-true-prob",type=float,default=0.05,help="早停阈值：真实类概率低于该值且已误分类则停止（论文 ImageNet 非定向设置）",)
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


def one_pixel_attack_untargeted(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    pixels: int,
    popsize: int,
    iters: int,
    F_scale: float,
    stop_true_prob: float,
    clamp_min: torch.Tensor,
    clamp_max: torch.Tensor,
) -> torch.Tensor:
    """
    One Pixel Attack（非定向，仅基于概率输出），严格遵循论文的 DE 框架与初始化方式：

    1) 编码：每个像素一个 tuple (x, y, r, g, b)，RGB 在 [0,255]
       多像素攻击为 d 个 tuple 串联，维度 5*d。

    2) 初始化（论文设置）：
       - 坐标 (x,y)：均匀随机采样
       - RGB：高斯采样 N(128, 127)，再裁剪到 [0,255]

    3) 差分进化（不使用 crossover）：
       对个体 j：trial = pop[r1] + F * (pop[r2] - pop[r3])，r1,r2,r3 两两不同且不等于 j
       若 trial 更优则替换 pop[j]（逐元素 1-1 竞争）。

    4) 适应度（非定向）：
       以最小化真实类概率 p_true 为目标。为了使用“越大越好”的比较，定义 fitness = -p_true。
    """
    model.eval()
    device = inputs.device
    b, c, h, w = inputs.shape

    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)

    def to_unnorm_01(x_norm: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x_norm * std_t + mean_t, 0.0, 1.0)

    def to_norm(x_01: torch.Tensor) -> torch.Tensor:
        return (x_01 - mean_t) / std_t

    def clip_candidate(cand: torch.Tensor) -> torch.Tensor:
        cand = cand.clone()
        for p in range(pixels):
            xi = 5 * p + 0
            yi = 5 * p + 1
            cand[..., xi] = torch.clamp(cand[..., xi], 0.0, float(w - 1))
            cand[..., yi] = torch.clamp(cand[..., yi], 0.0, float(h - 1))
            cand[..., 5 * p + 2 : 5 * p + 5] = torch.clamp(cand[..., 5 * p + 2 : 5 * p + 5], 0.0, 255.0)
        return cand

    def init_population() -> torch.Tensor:
        dim = 5 * pixels
        pop = torch.empty((popsize, dim), device=device)
        for p in range(pixels):
            pop[:, 5 * p + 0] = torch.randint(low=0, high=w, size=(popsize,), device=device).float()
            pop[:, 5 * p + 1] = torch.randint(low=0, high=h, size=(popsize,), device=device).float()
            rgb = torch.normal(mean=128.0, std=127.0, size=(popsize, 3), device=device)
            rgb = torch.clamp(rgb, 0.0, 255.0)
            pop[:, 5 * p + 2 : 5 * p + 5] = rgb
        return clip_candidate(pop)

    def apply_candidate(x0_norm: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        x01 = to_unnorm_01(x0_norm)  # (1,C,H,W) in [0,1]
        x01_adv = x01.clone()
        for p in range(pixels):
            x_pos = int(torch.round(cand[5 * p + 0]).item())
            y_pos = int(torch.round(cand[5 * p + 1]).item())
            x_pos = max(0, min(w - 1, x_pos))
            y_pos = max(0, min(h - 1, y_pos))

            rgb255 = cand[5 * p + 2 : 5 * p + 5]
            rgb01 = torch.clamp(rgb255 / 255.0, 0.0, 1.0)

            if c == 1:
                x01_adv[0, 0, y_pos, x_pos] = rgb01[0]
            else:
                x01_adv[0, 0, y_pos, x_pos] = rgb01[0]
                x01_adv[0, 1, y_pos, x_pos] = rgb01[1]
                x01_adv[0, 2, y_pos, x_pos] = rgb01[2]

        x01_adv = torch.clamp(x01_adv, 0.0, 1.0)
        x_adv_norm = to_norm(x01_adv)
        x_adv_norm = torch.max(torch.min(x_adv_norm, clamp_max), clamp_min)
        return x_adv_norm

    @torch.no_grad()
    def fitness_and_stats(x_adv_norm: torch.Tensor, true_y: int) -> Tuple[float, int, float]:
        logits = model(x_adv_norm)
        prob = F.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(prob).item())
        p_true = float(prob[true_y].item())
        fit = -p_true  # maximize -p_true <=> minimize p_true
        return fit, pred, p_true

    adv_batch = inputs.clone()

    for i in range(b):
        x0 = inputs[i : i + 1].detach()
        y_true = int(targets[i].item())

        pop = init_population()
        fit_vals = torch.empty((popsize,), device=device)

        best_fit = None
        best_cand = None

        for j in range(popsize):
            x_adv = apply_candidate(x0, pop[j])
            fit, _, _ = fitness_and_stats(x_adv, y_true)
            fit_vals[j] = fit
            if best_fit is None or fit > best_fit:
                best_fit = fit
                best_cand = pop[j].clone()

        for _ in range(max(iters, 1)):
            x_best = apply_candidate(x0, best_cand)
            _, pred_best, p_true_best = fitness_and_stats(x_best, y_true)
            if (p_true_best <= stop_true_prob) and (pred_best != y_true):
                break

            for j in range(popsize):
                idxs = list(range(popsize))
                idxs.remove(j)
                r1, r2, r3 = random.sample(idxs, 3)

                trial = pop[r1] + F_scale * (pop[r2] - pop[r3])
                trial = clip_candidate(trial)

                x_trial = apply_candidate(x0, trial)
                fit_trial, _, _ = fitness_and_stats(x_trial, y_true)

                if fit_trial > float(fit_vals[j].item()):
                    pop[j] = trial
                    fit_vals[j] = fit_trial
                    if fit_trial > float(best_fit):
                        best_fit = fit_trial
                        best_cand = trial.clone()

        adv_batch[i : i + 1] = apply_candidate(x0, best_cand).detach()

    return adv_batch.detach()


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
