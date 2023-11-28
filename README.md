<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

In this repository, I implemented two basic diffusion models (DDPM & DDIM). Both models were trained on the MNIST, FashionMNIST, and CIFAR10 datasets. The code for this implementation was referenced from [here](https://github.com/awjuliani/pytorch-diffusion).

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/PAD2003/base_diffusion_model.git
cd base_diffusion_model

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/PAD2003/base_diffusion_model.git
cd base_diffusion_model

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

You can override any parameter from command line like this

```bash
python -m src.train trainer=gpu \
data.dataset_name="CIFAR10" data.train_val_test_split="[50_000, 10_000, 0]" \
model.img_depth=3 \
logger.wandb.group="ddpm" logger.wandb.name="CIFAR10" \
trainer.max_epochs=50
```

## Results

I have trained a model with three datasets: MNIST, FashionMNIST, and CIFAR10. You can view the training report [here](https://api.wandb.ai/links/pad_team/sh4nigod)

### MNIST

![MNIST Generation](/imgs/mnist.gif)

### Fashion-MNIST

![Fashion MNIST Generation](/imgs/fashion.gif)

### CIFAR

![CIFAR Generation](/imgs/cifar.gif)