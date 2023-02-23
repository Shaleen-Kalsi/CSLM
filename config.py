import os
import json


class CSLMConfig():

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as jsonfile:
            config = json.load(jsonfile)

        self.dir = config['dataDir']['dir']
        self.data_dir = config['dataDir']['data']
        self.dataset = config['dataDir']['dataset_path'].replace('$dir', self.dir)
        self.train_path = config['dataDir']['train_path'].replace('$dir', self.dir)
        self.test_path = config['dataDir']['test_path'].replace('$dir', self.dir)
        self.val_path = config['dataDir']['val_path'].replace('$dir', self.dir)

        self.batch_size = int(config['hparams']['batch_size'])
        self.epochs = int(config['hparams']['epochs'])
        
        self.upstream_model = config['hparams']['upstream_model']

        # No of GPUs for training and no of workers for datalaoders
        self.accelerator = config['accelerator']
        self.devices = int(config['devices'])
        self.n_workers = int(config['n_workers'])

        # model checkpoint to continue from
        self.model_checkpt = config["hparams"]["model_checkpt"]

        #model params saving
        self.model_params = config["model_params"]

        self.run_name = config['run_name']
        
        # LR of optimizer
        self.lr = float(config['hparams']['lr'])
        self.weight_decay = float(config['hparams']['weight_decay'])
        self.num_classes = config["num_classes"]
