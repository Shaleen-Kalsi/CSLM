import json


class CSLMConfig():

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as jsonfile:
            config = json.load(jsonfile)

        self.dir = config['proj_dir']
        self.data_dir = config['data']['dir']
        self.train_path = config['data']['train_path'].replace('$dir', self.data_dir)
        self.hindi_mono_path = config['data']['hindi_mono_path'].replace('$dir', self.data_dir)
        self.test_path = config['data']['test_path'].replace('$dir', self.data_dir)
        self.val_path = config['data']['val_path'].replace('$dir', self.data_dir)

        self.batch_size = int(config['hparams']['batch_size'])
        self.epochs = int(config['hparams']['epochs'])
        
        self.upstream_model = config['hparams']['upstream_model']
        self.freeze = config['hparams']['freeze']

        # No of GPUs for training and no of workers for datalaoders
        self.accelerator = config['accelerator']
        self.devices = int(config['devices'])
        self.n_workers = int(config['n_workers'])

        # model checkpoint to continue from
        self.model_checkpt = config["hparams"]["model_checkpt"]
        self.save_dir = config["hparams"]["save_dir"].replace('$proj_dir', self.dir)

        # data augmentation technique to use - dynamic/static
        self.apply_mixup = config["hparams"]["apply_mixup"]

        self.run_name = config['run_name']
        
        # LR of optimizer
        self.lr = float(config['hparams']['lr'])
        self.weight_decay = float(config['hparams']['weight_decay'])
        self.num_classes = config["num_classes"]
