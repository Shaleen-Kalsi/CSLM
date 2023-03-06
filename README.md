# Making multilingual language models robust to code-switched text

## Setup instructions

> Note: Changing the directory structure of 'cslm' package might break poetry scripts.

Using poetry  
```
pip install poetry
poetry install
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

if the online mode fails on the cluster,
run it in offline mode `os.environ["WANDB_MODE"] = "offline"`, then sync it separately using,
```
wandb sync <wandb-run-path>
```



