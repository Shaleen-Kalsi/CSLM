# Using dynamic data augmentation for Sentiment Analysis on Hinglish Code-Switched text

Refer to `report.pdf` for details about the project.

## Setup instructions

> Note: Changing the directory structure of 'cslm' package might break poetry scripts.

[Poetry issues](https://github.com/python-poetry/poetry/issues/4231)

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



