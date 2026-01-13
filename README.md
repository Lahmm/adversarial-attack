# 使用说明
---
本项目于用于提供教材第二章对抗样本攻击实践部分的基本代码实现
---
## 1.环境搭建
本项目使用conda创建并管理环境，之后使用pip下载器安装依赖:
`conda create -n adv-att python=3.10`
`pip install -r requirements.txt`
如果读者使用其他方式进行环境创建和管理，本项目所需的主要依赖如下:
```
numpy==2.0.1
sympy==1.14.0
mpmath==1.3.0
networkx==3.4.2
jinja2==3.1.6
markupsafe==3.0.2
pillow==11.1.0
tqdm==4.67.1
typing_extensions==4.15.0
filelock==3.20.0
pyyaml==6.0.3
requests==2.32.5
urllib3==2.6.3
charset-normalizer==3.4.4
idna==3.11
certifi==2026.1.4
colorama==0.4.6
pysocks==1.7.1
cffi==2.0.0
pycparser==2.23

# PyTorch ecosystem (cross-platform declaration)
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1

```
此外， `cuda`版本为`cuda=11.8`
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

---
## 2.实践准备工作
本实践的数据集依托于`torch`库中提供的`mnist`和`cifar10`, 在初次使用时会自行下载
本实践的模型依托于`torch`内置的实现，在首次运行攻击前需要训练得到模型分类权重，具体的训练指令如下：
| Model           | Dataset  | Command |
|-----------------|----------|---------|
| resnet18        | cifar10  | `python train.py --model resnet18 --dataset cifar10 --epochs 5 --batch-size 64` |
| resnet34        | cifar10  | `python train.py --model resnet34 --dataset cifar10 --epochs 5 --batch-size 64` |
| densenet121     | cifar10  | `python train.py --model densenet121 --dataset cifar10 --epochs 5 --batch-size 64` |
| lenet           | mnist    | `python train.py --model lenet --dataset mnist --epochs 5 --batch-size 64` |

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
## 4.黑盒对抗攻击
本实践基本实现了VMIFGSM和One-pixel攻击，读者可分别在`vmifgsm.py`和`one-pixel.py`下进行复现
> 请注意：在复现攻击前，请保证已通过`train.py`训练得到所需模型和对应数据集的模型权重

具体攻击指令如下
| Method           | Dataset/Model  | Command |
|-----------------|----------|---------|
| vmifgsm | cifar10/resnet18/resnet34  | `python vmifgsm.py --model resnet18 --dataset cifar10 --epsilon 0.031 --alpha 0.007 --steps 10 --decay 1.0 --beta 1.5 --max-images 100 --save-dir output/vmifgsm/cifar10_resnet18 --blackbox-model resnet34` |
| pgd | mnist/lenet    | `python one-pixel.py --model lenet --dataset mnist --pixels 1 --popsize 400 --iters 100 --F 0.5 --max-images 100 --save-dir output/onepixel/mnist_lenet` |

读者应根据所需数据集和修改对应名称，参数解释详见代码注释
由于VMIFGSM是迁移攻击，使用攻击指令时 **必须添加黑盒模型**，One-pixel模型和数据集固定，无需改动

---
## 5.对抗性补丁攻击
本实践基于论文<a href="https://arxiv.org/pdf/1712.09665.pdf">Adversarial Patch paper</a>和GitHub仓库<a href="https://github.com/jhayes14/adversarial-patch">adversarial-patch</a>实现，读者可在`patchattack.py`下进行复现
> 请注意：在复现攻击前，请保证已通过`train.py`训练得到所需模型和对应数据集的模型权重

具体攻击指令如下：
`python patchattack.py --model resnet34 --dataset cifar10 --target 0 --save-dir output/patch/cifar10_resnet34`