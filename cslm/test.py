"""
Testing File
Usage:
    test.py --config=<config-file>

Options:
    --config=<config-file>   Path to config file containing hyperparameter info used in training
"""
import logging
from docopt import docopt
import torch.utils.data as data
from sklearn.metrics import classification_report
from tqdm import tqdm

from cslm.config import CSLMConfig
from cslm.dataset import CSLMDataset
from cslm.model import LightningModel

logging.basicConfig(level=logging.INFO)

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
    #model.to('cuda')
    model.eval() # evaluation mode
    test_df["predictions"] = ""
    num2labels = dict([(val, key) for key, val in test_set.labels2num.items()])

    logging.info("Getting Predictions..")
    for i in tqdm(range(len(test_df))):
        data = test_set[i]
        input_ids = data["input_ids_x"].view(1, -1)
        attention_mask = data["attention_mask_x"].view(1, -1)
        logits = model.basic_forward(input_ids, attention_mask)
        preds = logits.view(-1, config.num_classes).argmax(dim=1).detach().cpu().numpy().astype(int)[0]
        test_df.loc[i, 'predictions'] = num2labels[preds]
    
    labels = list(test_df['label'])
    preds = list(test_df['predictions'])

    logging.info("Saving predictions..")
    test_df.to_csv('test_predictions.csv', index=False)

    print(classification_report(y_true=labels, y_pred=preds, target_names=test_set.labels2num.keys(), zero_division='warn'))

    