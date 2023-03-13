"""
Script to make data splits for the Hate Speech Detection dataset.
Usage:
    preprocess_sa_hindi --csv=<csv> --out-dir=<dir>

Options:
    --csv=<csv>    Input CSV file path
    --out-dir=<dir>    Save directory
"""
import os
import pandas as pd
import numpy as np
from docopt import docopt
from ai4bharat.transliteration import XlitEngine


def preprocess_df(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Function to preprocess sentences : Remove urls, convert to lowercase, remove punctuations.
    Args:
        df: Input Dataframe
        col_name:   column name where sentences are present
    Returns:
        Preprocessed Dataframe
    """
    del_rows = []
    for i in range(len(df)):
        df.columns = ['sentence', 'label']
        sent = df.loc[i, col1]
        label = df.loc[i, col2]

        #convert to romanized hindi
        #e = XlitEngine(src_script_type="indic", beam_width=10, rescore=False)
        #out = e.translit_sentence(sent, lang_code="hi")
        #print(out)
        # Handle null rows
        if sent is np.nan:
            del_rows.append(i)
            continue
        if label == 0:
            df.loc[i, col2] = "negative"
        elif label == 1:
            df.loc[i, col2] = "neutral"
        elif label == 2:
            df.loc[i, col2] = "positive"
        df.loc[i, col1] = sent

    # Delete empty rows
    df = df.drop(index=del_rows)
    return df

def main():
    args = docopt(__doc__)
    csv_path = args['--csv']
    out_dir = args['--out-dir']

    #read_file = pd.read_csv (r'sa_hindi_rom_tweets.txt', sep='\t')
    #read_file.to_csv (r'sa_hindi_rom_tweets.csv', index=None)
    
    dataset = pd.read_csv(csv_path)
    dataset = preprocess_df(dataset, 'sentence', 'label')

    train_path = os.path.join(out_dir, "sa_hindi_rom_preprocessed_tweets.csv")
    dataset.to_csv(train_path, index=False)