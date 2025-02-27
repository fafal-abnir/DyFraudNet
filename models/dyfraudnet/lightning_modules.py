import torch
import hydra.utils
from dataclasses import dataclass
from torchmetrics.classification import BinaryAveragePrecision,BinaryAUROC
from torch.nn import BCEWithLogitsLoss

import pytorch_lightning as L


class LightningGNN(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()
        self.loss_fn = BCEWithLogitsLoss()
        # self.loss_fn = BinaryAUROC()
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = False

    def reset_loss(self, loss):
        self.loss_fn = loss()

    # def forward(self, x, edge_index, edge_label_index, previous_embeddings=None, num_current_edges=None,
    #             num_previous_edges=None):
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        pred= self.model(x, edge_index)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer

    def _shared_step(self, batch):
        pred = self.forward(batch)
        loss = self.loss_fn(pred[batch.node_mask], batch.y[batch.node_mask].type_as(pred))
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        auc_roc = self.metric_auroc(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        return loss, avg_pr, auc_roc

    def training_step(self, batch, batch_idx):
        loss, avg_pr, auc_roc = self._shared_step(batch)
        optimizer = self.optimizers()  # Get the optimizer
        self.manual_backward(loss, retain_graph=True)  # Manually handle backward pass
        optimizer.step()  # Update the model parameters
        optimizer.zero_grad()  # Zero the gradients for the next step
        self.log("train_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_au_roc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, avg_pr, auc_roc = self._shared_step(batch)
        self.log("val_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_au_roc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, avg_pr, auc_roc = self._shared_step(batch)
        self.log("test_avg_pr", avg_pr)
        self.log("test_au_roc", auc_roc)
        self.log("test_loss", loss)
