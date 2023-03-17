# Making multilingual language models robust to code-switched text

## Setup instructions

> Note: Changing the directory structure of 'cslm' package might break poetry scripts.

POETRY BEING A PAIN IN THE [ASS](https://github.com/python-poetry/poetry/issues/4231)

Create separate environments for running static augmentation and normal training process. We ran into a lot of dependency conflicts.

Create an environment with the Python version 3.10.0
```
conda create -n cslm-env python==3.10.0
conda activate cslm-env
pip install -r requirements.txt
```

Use a separate environment for running static data agumentation

```
pip install -r flair-requirements.txt
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



