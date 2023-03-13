"""
Script for static data augmentation. Currently supports only monolingual English texts.
Usage:
    static_da --csv=<csv-path> --type=<type> --save_path=<save-path>

Options:
    --csv=<csv-path>           Input CSV file path
    --type=<type>         Choose from ['lang_agnostic']
    --save_path=<save-path>    Save path for output csv
"""
import torch
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
from docopt import docopt
import flair
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

    if torch.cuda.is_available():
        flair.device = torch.device('gpu')
    else:
        flair.device = torch.device('cpu') 
    
    masked_df = pd.DataFrame(columns=df.columns)
    idx = 0
    MASK = "<GIB>"
    batch_size = 10
    for i in tqdm(range(0, len(df), batch_size)):
        sentences = []
        for k in range(batch_size):
            sentences.append(Sentence(df.loc[i+k, 'sentence']))
        tagger.predict(sentences)

        for sentence in sentences:
            masked_sent_NN = []
            masked_sent_VB = []
            masked_sent_JJ = []
            for j, label in enumerate(sentence.get_labels('pos')):
                if re.match('VB*', label.value) is not None:
                    masked_sent_NN.append(MASK)
                if re.match('JJ*', label.value) is not None:
                    masked_sent_VB.append(MASK)
                if re.match('NN*', label.value) is not None:
                    masked_sent_JJ.append(MASK)
                else:
                    masked_sent_JJ.append(sentence.tokens[j].text)
                    masked_sent_VB.append(sentence.tokens[j].text)
                    masked_sent_JJ.append(sentence.tokens[j].text)
                    
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