"""
Script for static data augmentation. Currently supports only monolingual English texts.
Usage:
    static_da --csv=<csv-path> --type=<type> --save_path=<save-path>

Options:
    --csv=<csv-path>           Input CSV file path
    --type=<type>         Choose from ['lang_agnostic']
    --save_path=<save-path>    Save path for output csv
"""
import os
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
from docopt import docopt
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load('pos')

def mask_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to mask tokens in a sentence based on its parts of speech.
    Args:
        sentence: Input sentence to be masked
    Returns:
        masked sentence
    """
    tagger = SequenceTagger.load('pos')
    
    masked_df = pd.DataFrame(columns=df.columns)
    idx = 0
    MASK = "<GIB>"
    for i in tqdm(range(len(df))):
        sentence = Sentence(df.loc[i, 'sentence'])
        masked_sent_NN = sentence.text.split()
        masked_sent_VB = sentence.text.split()
        masked_sent_JJ = sentence.text.split()

        tagger.predict(sentence)

        for i, label in enumerate(sentence.get_labels('pos')):
            if re.match('VB*', label.value) is not None:
                masked_sent_NN[i] = MASK
            if re.match('JJ*', label.value) is not None:
                masked_sent_VB[i] = MASK
            if re.match('NN*', label.value) is not None:
                masked_sent_JJ[i] = MASK
                
        masked_df.loc[idx, 'sentence'] = " ".join(masked_sent_NN)
        masked_df.loc[idx, 'label'] = df.loc[i, 'label']
        idx += 1
        masked_df.loc[idx, 'sentence'] = " ".join(masked_sent_VB)
        masked_df.loc[idx, 'label'] = df.loc[i, 'label']
        idx += 1
        masked_df.loc[idx, 'sentence'] = " ".join(masked_sent_JJ)
        masked_df.loc[idx, 'label'] = df.loc[i, 'label']
        idx += 1
        
    return masked_df


def augment_data(csv_path:str, type: str, save_path:str):
    """
    Function for static data augmentation
    Args:
        csv_path: Path to input CSV
        type: type of static data augmentation (currently supports only 'lang_agnostic')
        save_path: Save path for augmented CSV
    """
    dataset = pd.read_csv(csv_path)
    if type == 'lang_agnostic':
        masked_df = mask_tokens(dataset)
        masked_df.to_csv(save_path, index=False)


def main():
    args = docopt(__doc__)
    augment_data(args['--csv'], args['--type'], args['--save_path'])