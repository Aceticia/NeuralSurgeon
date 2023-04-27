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

    # Initialize datamodule
    datamodule.setup(stage="test")

    # Keep a matrix of scores
    score_a_increase = torch.zeros(len(layer_names), len(layer_names), device=device)
    score_b_decrease = torch.zeros(len(layer_names), len(layer_names), device=device)
    count = 0

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
            out_x = torch.softmax(model.classifier(out_x), dim=1)
            out_y = out_x.roll(1, 0)

            # Use the rolled inputs for 2nd round of forward
            y = x.roll(1, 0)
            target_y = target.roll(1, 0)

            # Find the logits for original label and condition label in the original pass
            a_x = out_x[torch.arange(len(out_x)), target]
            b_x = out_x[torch.arange(len(out_x)), target_y]

            # Find the logits for original label and condition label in the unconditioned pass
            a_y = out_y[torch.arange(len(out_y)), target]
            b_y = out_y[torch.arange(len(out_y)), target_y]

            # Iterate over pairs of layers
            for (idx_from, layer_from), (idx_to, layer_to) in product(enumerate(layer_names), repeat=2):
                out_y_tilde, _ = model.net.conditioned_forward_single(
                    x=y,
                    condition_dict=res_dict,
                    layer_conditions=[(layer_from, layer_to)],
                )
                out_y_tilde = torch.softmax(model.classifier(out_y_tilde), dim=1)

                # Find the logits for original label and condition label in the conditioned pass
                a_y_c = out_y_tilde[torch.arange(len(out_y_tilde)), target]
                b_y_c = out_y_tilde[torch.arange(len(out_y_tilde)), target_y]

                # Find the increase in conditioned label logit, excluding equal
                mask_a_equal = (a_x == a_y)
                inc_a = (a_y_c[~mask_a_equal] - a_y[~mask_a_equal]) / (a_x[~mask_a_equal] - a_y[~mask_a_equal])

                # Find the decrease in original label logit
                mask_b_equal = (b_y == b_x)
                dec_b = (b_y_c[~mask_b_equal] - b_x[~mask_b_equal]) / (b_y[~mask_b_equal] - b_x[~mask_b_equal])

                # Add to the matrix
                score_a_increase[idx_from, idx_to] += inc_a.mean()
                score_b_decrease[idx_from, idx_to] += dec_b.mean()
                count += 1

    # Divide by the number of test batches
    score_a_increase /= count
    score_b_decrease /= count

    # Move to cpu
    score_a_increase = score_a_increase.cpu()
    score_b_decrease = score_b_decrease.cpu()

    # Store the matrix and name with the checkpoint
    p = Path(cfg.store_path)

    # Make the directory if not existing
    p.mkdir(parents=True, exist_ok=True)

    torch.save((score_a_increase, score_b_decrease, layer_names), p / f"conditioning_{model.net.modulator}")
    metric_dict = {"sharpen": score_a_increase, "dampen": score_b_decrease}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
