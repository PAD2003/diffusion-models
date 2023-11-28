export CUDA_VISIBLE_DEVICES=0

# python -m src.train trainer=gpu data.dataset_name="FashionMNIST" logger.wandb.group="ddpm" logger.wandb.name="FashionMNIST"
# python -m src.train trainer=gpu data.dataset_name="MNIST" logger.wandb.group="ddpm" logger.wandb.name="MNIST"
python -m src.train trainer=gpu data.dataset_name="CIFAR10" data.train_val_test_split="[50_000, 10_000, 0]" model.img_depth=3 logger.wandb.group="ddpm" logger.wandb.name="CIFAR10" trainer.max_epochs=50