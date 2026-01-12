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

# Dataset normalization stats
DATASET_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal VMIFGSM attack")
    parser.add_argument("--epsilon", type=float, default=0.031, help="Linf constraint radius, default 8/255")
    parser.add_argument("--alpha", type=float, default=0.007, help="Step size, default ~2/255")
    parser.add_argument("--steps", type=int, default=10, help="Number of attack iterations")
    parser.add_argument("--decay", type=float, default=1.0, help="Momentum decay factor")
    parser.add_argument("--beta",type=float,default=1.5,help="Variance term weight (extra attack hyper-parameter for VMIFGSM)")
    parser.add_argument("--max-images", type=int, default=100, help="Max number of correctly classified samples to attack (-1 for all)")
    parser.add_argument("--save-dir", type=str, default="output/vmifgsm/cifar10_resnet18", help="Directory to save adversarial images")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--weights-dir", type=str, default="model_weights")
    parser.add_argument("--blackbox-model", type=str, default="", help="Optional black-box model name for transfer evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--use-cuda", action="store_true")
    return parser.parse_args()


def resolve_weights_path_for(
    model_name: str, dataset_name: str, weights_dir: str, blackmodel_name: str
) -> str:
    dataset_key = dataset_name.strip().lower()
    model_key = model_name.strip().lower()
    black_model_key = blackmodel_name.strip().lower()
    filename = f"{dataset_key}_{model_key}.pth"
    black_filename = f"{dataset_key}_{black_model_key}.pth"
    return os.path.abspath(os.path.join(weights_dir, filename)), os.path.abspath(os.path.join(weights_dir, black_filename))


def resolve_weights_path(args: argparse.Namespace) -> str:
    return resolve_weights_path_for(
        model_name=args.model,
        dataset_name=args.dataset,
        blackmodel_name=args.blackbox_model,
        weights_dir=args.weights_dir,
    )


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


def vmifgsm_attack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    decay: float,
    beta: float,
    clamp_min: torch.Tensor,
    clamp_max: torch.Tensor,
    noise_scale: float = 1e-3,
    noise_samples: int = 5,
) -> torch.Tensor:
    """
    Variance-guided Momentum Iterative FGSM (Linf)
    Inputs: white-box model, normalized inputs/targets, epsilon (radius), alpha (step), steps, decay, beta, clamp bounds.
    Output: adversarial examples in normalized space.
    """
    model.eval()
    adv = inputs.detach()
    momentum = torch.zeros_like(inputs)
    prev_grad = torch.zeros_like(inputs)

    for _ in range(max(steps, 1)):
        # Estimate gradient with small random smoothing to stabilize variance.
        grad_acc = torch.zeros_like(adv)
        for _ in range(max(noise_samples, 1)):
            noisy_adv = adv + torch.randn_like(adv) * noise_scale
            noisy_adv = torch.max(torch.min(noisy_adv, clamp_max), clamp_min)
            noisy_adv = noisy_adv.clone().detach().requires_grad_(True)

            outputs = model(noisy_adv)
            loss = F.cross_entropy(outputs, targets)

            model.zero_grad()
            loss.backward()

            grad_acc += noisy_adv.grad.detach()

        grad = grad_acc / float(max(noise_samples, 1))
        variance = grad - prev_grad
        mixed_grad = grad + beta * variance
        grad_norm = torch.mean(torch.abs(mixed_grad), dim=(1, 2, 3), keepdim=True)
        normalized_grad = mixed_grad / torch.clamp(grad_norm, min=1e-8)

        momentum = decay * momentum + normalized_grad
        adv = adv + alpha * momentum.sign()

        delta = torch.clamp(adv - inputs, min=-epsilon, max=epsilon)
        adv = inputs + delta
        adv = torch.max(torch.min(adv, clamp_max), clamp_min)

        prev_grad = grad.detach()
        model.zero_grad()

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
    
    model_weights_path, black_model_weights_path = resolve_weights_path(args)

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Weights not found: {model_weights_path}")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)

    blackbox_model = None
    if args.blackbox_model:
        if not os.path.exists(black_model_weights_path):
            raise FileNotFoundError(f"Black-box weights not found: {black_model_weights_path}")

        blackbox_model = build_model(
            args.blackbox_model,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=False,
        )
        blackbox_model.load_state_dict(torch.load(black_model_weights_path, map_location=device))
        blackbox_model.to(device)
        blackbox_model.eval()

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
    transfer_correct = 0
    offset = 0

    model.eval()
    for inputs, targets in tqdm(attack_loader, desc="VMIFGSM", unit="batch"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        adv = vmifgsm_attack(
            model=model,
            inputs=inputs,
            targets=targets,
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.steps,
            decay=args.decay,
            beta=args.beta,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )

        with torch.no_grad():
            outputs = model(adv)
            _, preds = outputs.max(1)

        if blackbox_model is not None:
            with torch.no_grad():
                bb_outputs = blackbox_model(adv)
                _, bb_preds = bb_outputs.max(1)
            transfer_correct += bb_preds.eq(targets).sum().item()

        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        batch_indices = attack_indices[offset : offset + targets.size(0)]
        save_adversarial_images(
            adv, mean, std, args.save_dir, batch_indices, targets, preds
        )
        offset += targets.size(0)

    adv_acc = correct / max(total, 1)
    asr = 1.0 - adv_acc
    print(f"VMIFGSM acc: {adv_acc:.4f} ({correct}/{total})")
    print(f"ASR: {asr:.4f} ({total - correct}/{total})")

    if blackbox_model is not None:
        transfer_acc = transfer_correct / max(total, 1)
        transfer_asr = 1.0 - transfer_acc
        print(
            f"Black-box acc: {transfer_acc:.4f} ({transfer_correct}/{total}) - transfer ASR: {transfer_asr:.4f} ({total - transfer_correct}/{total})"
        )

    print(f"Saved adversarial images to {os.path.abspath(args.save_dir)}")


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    attack(args)


if __name__ == "__main__":
    main()
