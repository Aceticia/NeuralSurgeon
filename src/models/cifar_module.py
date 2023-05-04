from typing import Any

from itertools import chain

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class CIFARLitModule(LightningModule):
    """Example of LightningModule for CIFAR classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        modulator: callable
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net(modulator=modulator)

        # classifier
        final_size = self.net.get_final_output_size()
        self.classifier = torch.nn.Linear(final_size, 10)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.pred_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_pred_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 4:
            # B, C, H, W -> B, C2
            return self.net(x)
        elif len(x.shape) == 5:
            # B, T, C, H, W -> B, T, C2
            # Step by step condition
            out_x, res_dict = self.net(x[:, 0])

            # Returns
            rets = [out_x]

            # Iterate over time
            for t in range(x.shape[1]):
                out_x, res_dict = self.net.conditioned_forward_single(
                    x=x[:, t],
                    condition_dict=res_dict,
                    layer_condition=[(1,1)]   # TODO
                )
                rets.append(out_x)

            return torch.stack(rets, dim=1)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        outs, res_dict = self.forward(x)
        logits = self.classifier(outs)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, res_dict

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int=0):
        loss, preds, targets, res_dict = self.model_step(batch)
        pair_loss = self.net.sample_layer_pair_loss(res_dict, n_samples=10)

        # update and log metrics
        self.train_loss(loss)
        self.pred_loss(pair_loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/paired_pred_loss", self.pred_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False)

        # return loss or backpropagation will fail
        return loss+pair_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, res_dict = self.model_step(batch)
        pair_loss = self.net.sample_layer_pair_loss(res_dict, n_samples=100)

        # update and log metrics
        self.val_loss(loss)
        self.val_pred_loss(pair_loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/paired_pred_loss", self.val_pred_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            params=chain(
                self.classifier.parameters(),
                self.net.get_mapping_params()
            )
        )
        return optimizer


if __name__ == "__main__":
    _ = CIFARLitModule(None, None, None)
