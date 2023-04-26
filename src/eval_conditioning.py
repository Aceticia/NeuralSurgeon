from typing import List

from pathlib import Path

import torch

import hydra
import pyrootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from tqdm import tqdm
from itertools import product

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
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
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> None:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Check device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Move model to device
    model = model.to(device)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
    }

    log.info("Starting testing!")
    model.eval()

    # Get all layer names
    layer_names = list(model.net.layer_sizes().keys())

    # Keep a matrix of scores. 10 classes
    size_last = model.net.layer_sizes()[layer_names[-1].num_channels]
    pre_conditioning_sensitivity_all = torch.zeros(size_last, device=device)
    post_conditioning_sensitivity_all = torch.zeros((len(layer_names), len(layer_names), size_last), device=device)

    count_pre = 0
    count_post = 0

    # Initialize datamodule
    datamodule.setup(stage="test")

    with torch.no_grad():
        # Iterate over all test data
        for batch in tqdm(datamodule.test_dataloader()):
            # Get the data
            x, target = batch

            # Move to device
            x = x.to(device)
            target = target.to(device)

            # Forward pass
            out_x, res_dict = model(x)

            # Use the rolled inputs for 2nd round of forward
            y = x.roll(1, 0)
            out_y = out_x.roll(1, 0)

            # Find the conditioning rank
            conditioning_rank = torch.argsort(out_x, dim=1, descending=False)

            # Find the pre-conditioning sensitivity by adjusting the logit dimension order
            pre_conditioning_sensitivity = out_y.gather(1, conditioning_rank)
            pre_conditioning_sensitivity_all += pre_conditioning_sensitivity.sum(dim=0)
            count_pre += x.shape[0]

            # Iterate over pairs of layers
            for (idx_from, layer_from), (idx_to, layer_to) in product(enumerate(layer_names), repeat=2):
                out_y_tilde, _ = model.net.conditioned_forward_single(
                    x=y,
                    condition_dict=res_dict,
                    layer_conditions=[(layer_from, layer_to)],
                    alpha=cfg.alpha
                )

                # Find the post-conditioning sensitivity by adjusting the logit dimension order
                post_conditioning_sensitive = out_y_tilde.gather(1, conditioning_rank)
                post_conditioning_sensitivity_all[idx_from, idx_to] += post_conditioning_sensitive.sum(dim=0)
                count_post += x.shape[0]

    # Divide by the number of test batches
    pre_conditioning_sensitivity_all /= count_pre
    post_conditioning_sensitivity_all /= count_post

    # Move to cpu
    pre_dist = pre_conditioning_sensitivity_all.cpu()
    post_dist = post_conditioning_sensitivity_all.cpu()

    # Store the matrix and name with the checkpoint
    p = Path(cfg.store_path)

    # Make the directory if not existing
    p.mkdir(parents=True, exist_ok=True)

    torch.save((pre_dist, post_dist, layer_names), p / f"conditioning_alpha{cfg.alpha}.pt")
    metric_dict = {"pre": pre_dist, "post": post_dist}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_conditioning.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
