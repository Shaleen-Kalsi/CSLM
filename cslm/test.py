"""
Testing File
Usage:
    test.py --config=<config-file>

Options:
    --config=<config-file>   Path to config file containing hyperparameter info used in training
"""
import torch
import ast
import numpy as np
import logging
from docopt import docopt
import torch.utils.data as data
from sklearn.metrics import classification_report
from tqdm import tqdm

from cslm.config import CSLMConfig
from cslm.dataset import CSLMDataset
from cslm.model import LightningModel
from netcal.metrics import ECE
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO)

def numpy_one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def ECEMetric(y, y_softmax_scores):
    y_softmax_scores = np.stack(y_softmax_scores, axis=0)
    # 10 bins
    ece = ECE(10)
    return ece.measure(y_softmax_scores, y)

def main():
    args = docopt(__doc__)
    config = CSLMConfig(args["--config"])

    # test data
    test_set = CSLMDataset(
        CSVPath = config.test_path,
        hparams = config,
        is_train=False
    )

    test_df = test_set.data
    logging.info("Loading model from checkpoint..")
    model = LightningModel.load_from_checkpoint(config.model_checkpt, config=config)
    if torch.cuda.is_available():
        model.to('cuda')
    print("cuda available", torch.cuda.is_available())
    model.eval() # evaluation mode
    test_df["predictions"] = ""
    test_df["probs"] = ""
    num2labels = dict([(val, key) for key, val in test_set.labels2num.items()])

    logging.info("Getting Predictions..")
    for i in tqdm(range(len(test_df))):
        data = test_set[i]
        input_ids = data["input_ids"].view(1, -1)
        attention_mask = data["attention_mask"].view(1, -1)
        if torch.cuda.is_available():
            input_ids = input_ids.to(device = 'cuda')
            attention_mask = attention_mask.to(device = 'cuda')
        outputs = model.basic_forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs[0] 
        probs = logits.view(-1, config.num_classes).detach().cpu().numpy().astype(float)[0]
        preds = logits.view(-1, config.num_classes).argmax(dim=1).detach().cpu().numpy().astype(int)[0]
        test_df.loc[i, 'predictions'] = num2labels[preds]
        test_df.at[i, 'probs'] = probs
    
    labels = list(test_df['label'])
    preds = list(test_df['predictions'])

    logging.info("Saving predictions..")
    test_df.to_csv('test_predictions.csv', index=False)

    print(classification_report(y_true=labels, y_pred=preds, target_names=test_set.labels2num.keys(), zero_division='warn'))

    # conver to numpy array for ECE
    # convert to one hot
    y = numpy_one_hot(np.array([test_set.labels2num[l] for l in labels]), config.num_classes)
    y_pred_probs = np.array(list(test_df["probs"]))
    print("ECE metric: ", ECEMetric(y, y_pred_probs))

    