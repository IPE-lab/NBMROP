from dataset import ROPContinousLearningDataset
import yaml
from models import BaseModel
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

import os
def main():
    # load configuration

    for test_well in ['well_0', 'well_1', 'well_2', 'well_3', 'well_4', 'well_5'][:1]:
    # load dataset
        config = {}
        config['modelconfig'] = {}
        config['modelconfig']['num_concepts'] = 10
        config['modelconfig']['num_classes'] = 1
        config['modelconfig']['modeltype'] = 'nbm'
        config['modelconfig']['dropout'] = 0.3
        config['modelconfig']['batchnorm'] = True
        config['modelconfig']['num_bases'] = 100
        config['modelconfig']['hidden_dims'] = (32, 32, 16)


        weight_decay = 1e-5
        batch_size = 256
        config['modelconfig']['learning_rate'] = 0.0001
        config['modelconfig']['weight_decay'] = weight_decay

        model = BaseModel(**config['modelconfig'])

        train_dataset = ROPContinousLearningDataset(test_well, 'train', rigind = 0)
        max_rig = train_dataset.max_rigind

        train_dataset = ROPContinousLearningDataset(test_well, 'train', rigind = max_rig)
        val_dataset = ROPContinousLearningDataset(test_well, 'validation', rigind = max_rig)
        test_dataset = ROPContinousLearningDataset(test_well, 'test', rigind = max_rig)

        model = BaseModel(**config['modelconfig'])

        datamodule = pl.LightningDataModule.from_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=val_dataset,
            batch_size = batch_size,
        )

        model_save_path = f'temp_test/{test_well}_last_second/'
        # training
        modelcheckpoint = ModelCheckpoint(
            dirpath=model_save_path,
            filename = 'best_model',
            monitor='val_loss', 
            save_top_k=1,
            save_last=True,
            mode='min'
            )
        
        logger = pl.loggers.TensorBoardLogger(model_save_path)
        earlystop = EarlyStopping(monitor='val_loss', patience=500)
        
        trainer = Trainer(accelerator='mps', 
                        devices=1,
                        max_epochs=10000,
                        callbacks=[modelcheckpoint,earlystop],
                        logger=logger,
                        log_every_n_steps = 5,
                        )
        
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)
        

if __name__ == '__main__':
    main()