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

    # Do we have device?
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Move model to device
    model = model.to(device)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    model.eval()

    # Get all layer names
    layer_names = list(model.net.layer_sizes().keys())

    # Keep a matrix of scores
    scores = torch.zeros(len(layer_names), len(layer_names), device=device)
    count = 0

    # Initialize datamodule
    datamodule.setup(stage="test")

    with torch.no_grad():
        # Iterate over all test data
        for batch in tqdm(datamodule.test_dataloader()):
            # Get the data
            x, _ = batch

            # Move to device
            x = x.to(device)

            # Forward pass
            _, res_dict = model(x)

            # Iterate over pairs of layers
            for (idx_from, layer_from), (idx_to, layer_to) in product(enumerate(layer_names), repeat=2):
                target = res_dict[layer_to]
                pred = model.net.get_pairwise_predictions(res_dict, layer_from, layer_to)

                # Remove the target 0's
                zero_mask = target == 0
                target = target[~zero_mask]
                pred = pred[~zero_mask]

                # Find MAPE
                diff = (pred - target).abs() / target.abs()

                # Add to the matrix
                scores[idx_from, idx_to] += diff.mean()
                count += 1
    
    # Divide by the number of test batches
    scores /= count

    # Move to CPU
    scores = scores.cpu()

    # Store the matrix and name with the checkpoint
    p = Path(cfg.ckpt_path)
    torch.save((scores, layer_names), p.parent / f"{p.stem}_scores.pt")

    metric_dict = {"matrix": scores}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_all_pairs.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
