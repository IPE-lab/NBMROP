from typing import Iterator
import pytorch_lightning as pl
from .nbm import ConceptNBMNary
from .mlp import MLP
import torch 
import torch.nn as nn

import torch.nn.functional as F


class BaseModel(pl.LightningModule):
    def __init__(self, 
                 modeltype,
                 learning_rate = 0.0,
                 weight_decay = 0.0,
                 **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if modeltype == 'nbm':
            self.model = ConceptNBMNary(**kwargs)
        elif modeltype == 'mlp':
            self.model = MLP()

        self.model_init()

    def _loss(self, outputs, targets):
        outputs = outputs.squeeze(-1)
        targets = targets.squeeze(-1)
        return nn.MSELoss()(outputs, targets)
    
    def _get_output_and_losses(self, inputs, targets):
        outputs, _ = self.model(inputs)
        loss = self._loss(outputs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        loss = self._get_output_and_losses(inputs, targets)
        self.log('train_loss', loss, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        loss = self._get_output_and_losses(inputs, targets)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        loss = self._get_output_and_losses(inputs, targets)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    
    # initialize model's parameter through Ming's method
    def model_init(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:  # Check if the parameter has more than 1 dimension
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)