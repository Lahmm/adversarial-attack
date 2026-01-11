# 使用说明
---
## 1.环境搭建
本项目使用conda创建并管理环境，具体的指令为:
`conda env create -f environment.yml`
如果读者使用其他方式进行环境创建和管理，本项目所需的主要依赖如下:
```
certifi==2026.1.4
charset-normalizer==3.4.4
filelock==3.20.0
idna==3.11
jinja2==3.1.6
markupsafe==3.0.2
mpmath==1.3.0
networkx==3.4.2
numpy==2.0.1
pillow==11.1.0
pyyaml==6.0.3
requests==2.32.5
sympy==1.14.0
typing_extensions==4.15.0
urllib3==2.6.3

torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
```
此外`python`版本为`python=3.10.19`, `cuda`版本为`cuda=11.8`

---
## 2.实践准备工作
本实践的数据集依托于`torch`库中提供的`mnist`和`cifar10`, 在初次使用时会自行下载
本实践的模型依托于`torch`内置的实现，在首次运行攻击前需要训练得到模型分类权重，具体的训练指令如下：
| Model           | Dataset  | Command |
|-----------------|----------|---------|
| resnet18        | cifar10  | `python train.py --model resnet18 --dataset cifar10 --epochs 10 --batch-size 64` |
| resnet18        | mnist    | `python train.py --model resnet18 --dataset mnist --epochs 10 --batch-size 64` |
| resnet34        | cifar10  | `python train.py --model resnet34 --dataset cifar10 --epochs 10 --batch-size 64` |
| resnet34        | mnist    | `python train.py --model resnet34 --dataset mnist --epochs 10 --batch-size 64` |
| vgg16           | cifar10  | `python train.py --model vgg16 --dataset cifar10 --epochs 10 --batch-size 64` |
| vgg16           | mnist    | `python train.py --model vgg16 --dataset mnist --epochs 10 --batch-size 64` |
| densenet121     | cifar10  | `python train.py --model densenet121 --dataset cifar10 --epochs 10 --batch-size 64` |
| densenet121     | mnist    | `python train.py --model densenet121 --dataset mnist --epochs 10 --batch-size 64` |

---
## 3.白盒对抗攻击