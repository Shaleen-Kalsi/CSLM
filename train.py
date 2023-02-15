"""
Training File
Usage:
    train.py --config=<config-file>

Options:
    --config=<config-file>   Path to config file containing hyperparameter info
"""
import os
import json
import logging
from docopt import docopt
import torch.utils.data as data
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from config import CSLMConfig
from dataset import CSLMDataset
from model import LightningModel
from config import CSLMConfig

# set to online to use wandb
os.environ["WANDB_MODE"] = "offline"
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    args = docopt(__doc__)
    config = CSLMConfig(args["--config"])

    seed_everything(42)

    # set logger
    logger = WandbLogger(
        name=config.run_name,
        project='CSLM'
    )

    # Set up appropriate dataloaders 
    # Train
    logging.info("Setting up dataloaders..")
    train_set = CSLMDataset(
        CSVPath = config.train_path,
        hparams = config,
        is_train=True
    )
    train_loader = data.DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.n_workers,
    )
    # Validation
    valid_set = CSLMDataset(
        CSVPath = config.val_path,
        hparams = config,
        is_train=False
    )
    val_loader = data.DataLoader(
        valid_set, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.n_workers,
    )
    # Test
    test_set = CSLMDataset(
        CSVPath = config.test_path,
        hparams = config,
        is_train=False
    )
    test_loader = data.DataLoader(
        test_set, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.n_workers,
    )

    logging.info("Setting up the model..")
    model = LightningModel(config, config.num_labels)

    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        monitor='val/loss', 
        mode='min',
        verbose=1)

    early_stopping = EarlyStopping(
                monitor='val/loss',
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode='min'
                )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        fast_dev_run=False,
        log_every_n_steps=1,
        devices=config.devices, 
        max_epochs=config.epochs, 
        callbacks=[
            # early_stopping, uncomment if you want to enable early stopping
            # model_checkpoint_callback,
            #lr_monitor
        ],
        #logger=logger,
        resume_from_checkpoint=config.model_checkpt,
        accelerator=config.accelerator # If your machine has GPUs, it will use the GPU Accelerator for training
    )

    logging.info("Training the model..")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logging.info("Testing the model..")
    trainer.test(model, dataloaders=test_loader, verbose=True)
