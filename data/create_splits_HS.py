import os
import re
import pandas as pd
import numpy as np
from config import CSLMConfig
from sklearn.model_selection import train_test_split


def preprocess_df(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Function to preprocess sentences : Remove urls, convert to lowercase, remove punctuations.
    Args:
        df: Input Dataframe
        col_name:   column name where sentences are present
    Returns:
        Preprocessed Dataframe
    """
    for i in range(len(df)):
        sent = df.loc[i, col_name]
        sent = sent.lower()
        sent = re.sub(r"\shttps://.*\s", "", sent).strip()
        df.loc[i, col_name] = sent

def split_dataset(csv_path, out_dir):
    """
    Function to split dataset into train, test and validation sets
    Args:
        csv_path: Path to input CSV
        out_dir: Path to dir to save the splits
    """
    dataset = pd.read_csv(csv_path, sep='\t')
    dataset = preprocess_df(dataset, 'sentence')
    #80, 10, 10 split
    train, validate, test = np.split(dataset.sample(frac=1, random_state=42), [int(.6*len(dataset)), int(.8*len(dataset))])

    train_path = os.path.join(out_dir, '/train.csv')
    train.to_csv(train_path, index=False)

    validate_path = os.path.join(out_dir, '/val.csv')
    validate.to_csv(validate_path, index=False)

    test_path = os.path.join(out_dir, '/test.csv')
    test.to_csv(test_path, index=False)