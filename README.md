# Making multilingual language models robust to code-switched text

## Setup instructions

> Note: Changing the directory structure of 'cslm' package might break poetry scripts.
Using poetry  
```
pip install poetry
poetry install
```

Alternate way to install pytorch + GPU on Google Cloud Platform using pip
```
poetry remove torch
pip install torch>=1.12.0+cu116 torchvision>=0.13.0+cu116 torchaudio>=0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

`pip install flair` [breaks](https://github.com/flairNLP/flair/issues/2969) for some python versions. So install it separately.
```
poetry run pip install git+https://github.com/flairNLP/flair  
```


## Usage

### Dataset splits

To create dataset splits for,

1. Hate Speech Detection -
   ``` poetry run datasplit_HS ```
2. Sentiment Analysis
    ``` poetry run datasplit_SA ```

While training on different datasets, make sure to change `num_classes` in `config.json` and update the dict `labels2num` in `dataset.py`.
    
### Training
To train file, make appropriate changes to config file (Example config file at `config.json`) and run,
```
poetry run train --config <config-json>
```
### Testing
To test file, it is easiest keep the config file (Example config file at `config.json`) the same as during training and run,
```
poetry run test --config <config-json>
```


### Logging

Authorization for wandb,
```
poetry run python -m wandb login
```

To enable wandb, set `os.environ["WANDB_MODE"] = "online"`

if the online mode [fails](https://github.com/ultralytics/yolov5/issues/5498) on the cluster,
run it in offline mode `os.environ["WANDB_MODE"] = "offline"`, then sync it separately using,
```
wandb sync <wandb-run-path>
```



