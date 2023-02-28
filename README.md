# Making multilingual language models robust to code-switched text

## Setup instructions

> Note: Changing the directory structure of 'cslm' package might break poetry scripts.

Using poetry  
```
pip install poetry
poetry install
```

## Usage

### Dataset splits

To create dataset splits for,

1. Hate Speech Detection -
   ``` poetry run datasplit_HS ```
2. Sentiment Analysis

### Training
To train file, make appropriate changes to config file (Example config file at `config.json`) and run,
```
poetry run train --config <config-json>
```

### Logging

Authorization for wandb,
```
poetry run python -m wandb login
```

To enable wandb, set `os.environ["WANDB_MODE"] = "online"`



