"""
Script to make data splits for the Hate Speech Detection dataset.
Usage:
    preprocess_sa_english --csv=<csv> --out-dir=<dir>

Options:
    --csv=<csv>    Input CSV file path
    --out-dir=<dir>    Save directory
"""
import os
import pandas as pd
import numpy as np
from docopt import docopt


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
        label = df.loc[i, col1]
        sent = df.loc[i, col2]
        # Handle null rows
        if sent is np.nan:
            del_rows.append(i)
            continue

        df.loc[i, col1] = sent
        df.loc[i, col2] = label

    # Delete empty rows
    df = df.drop(index=del_rows)
    return df

def main():
    args = docopt(__doc__)
    csv_path = args['--csv']
    out_dir = args['--out-dir']

    dataset = pd.read_csv(csv_path)
    dataset = preprocess_df(dataset, 'sentence', 'label')

    train_path = os.path.join(out_dir, "sa_english_preprocessed_train.csv")
    dataset.to_csv(train_path, index=False)