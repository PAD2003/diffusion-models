from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from src.models.ddpm_module import DDPMModule
from src.models.ddim_sample import DDIMSample
import torch
import imageio
import numpy as np
from PIL import Image

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

def inference(cfg: DictConfig):
    DDPM_model = DDPMModule.load_from_checkpoint(cfg.ckpt_path)
    sample = DDIMSample(
        model=DDPM_model,
        S=10,
        ddim_discretize="uniform",
        ddim_eta=0
    )
    torch.manual_seed(100)
    gen_samples = sample.generate_sample()
    gen_samples_np = gen_samples.to("cpu").numpy()

    frames = []
    for frame_idx in range(gen_samples_np.shape[0]):
        # Arrange sequences in a 3x3 grid for each frame
        grid = np.vstack([np.hstack(gen_samples_np[frame_idx, i, j, :, :, 0] for j in range(3)) for i in range(3)])
        frames.append(grid)

    # Save the GIF
    file_path = "data/ddim_test_2.gif"
    imageio.mimsave(file_path, frames, fps=3)

@hydra.main(version_base="1.3", config_path="../configs", config_name="ddim_sample.yaml")
def main(cfg: DictConfig):
    inference(cfg)

if __name__ == "__main__":
    main()
