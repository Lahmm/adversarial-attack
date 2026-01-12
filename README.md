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
tqdm==4.67.1

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
| resnet18        | cifar10  | `python train.py --model resnet18 --dataset cifar10 --epochs 5 --batch-size 64` |
| resnet18        | mnist    | `python train.py --model resnet18 --dataset mnist --epochs 5 --batch-size 64` |
| resnet34        | cifar10  | `python train.py --model resnet34 --dataset cifar10 --epochs 5 --batch-size 64` |
| resnet34        | mnist    | `python train.py --model resnet34 --dataset mnist --epochs 5 --batch-size 64` |
| vgg16           | cifar10  | `python train.py --model vgg16 --dataset cifar10 --epochs 5 --batch-size 64` |
| vgg16           | mnist    | `python train.py --model vgg16 --dataset mnist --epochs 5 --batch-size 64` |
| densenet121     | cifar10  | `python train.py --model densenet121 --dataset cifar10 --epochs 5 --batch-size 64` |
| densenet121     | mnist    | `python train.py --model densenet121 --dataset mnist --epochs 5 --batch-size 64` |

---
## 3.白盒对抗攻击
本实践基本实现了快速梯度符号法（FGSM）和投影梯度下降法（PGD）的攻击流程，读者可分别在`fgsm.py`和`pgd.py`下进行复现
> 请注意：在复现攻击前，请保证已通过`train.py`训练得到所需模型和对应数据集的模型权重

具体攻击指令如下：
| Method           | Dataset/Model  | Command |
|-----------------|----------|---------|
| fgsm        | cifar10/resnet18  | `python fgsm.py --model resnet18 --dataset cifar10 --epsilon 0.15 --max-images 50 --save-dir output/fgsm/cifar10_resnet18` |
| pgd        | cifar10/resnet18    | `python pgd.py --model resnet18 --dataset cifar10 --epsilon 0.2 --alpha 0.05 --steps 50 --max-images 50 --save-dir output/pgd/cifar10_resnet18` |

读者应根据所需数据集和修改对应名称，参数解释详见代码注释

---
