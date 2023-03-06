"""
Testing File
Usage:
    test.py --config=<config-file>

Options:
    --config=<config-file>   Path to config file containing hyperparameter info used in training
"""
import os
import logging
from docopt import docopt
import torch.utils.data as data
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from cslm.config import CSLMConfig
from cslm.dataset import CSLMDataset
from cslm.model import LightningModel

# set to online to use wandb
os.environ["WANDB_MODE"] = "offline"
logging.basicConfig(level=logging.INFO)

def main():

    args = docopt(__doc__)
    config = CSLMConfig(args["--config"])

    seed_everything(42)

    # set logger
    logger = WandbLogger(
        name=config.run_name,
        project='CSLM'
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

    model = LightningModel.load_from_checkpoint(config.save_dir + "/" + config.run_name + "-epoch="+ str(config.epochs - 1) + ".ckpt")
    trainer = Trainer(
        logger=logger,
        accelerator=config.accelerator)
    logging.info("Testing the model..")
    trainer.test(model, dataloaders=test_loader, verbose=True)