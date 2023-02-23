import sys
import pandas as pd
import numpy as np
from config import CSLMConfig
from sklearn.model_selection import train_test_split

def split_dataset(csvPath, savePath):
    dataset = pd.read_csv(csvPath, sep='\t')
    #80, 10, 10 split
    train, validate, test = np.split(dataset.sample(frac=1, random_state=42), [int(.6*len(dataset)), int(.8*len(dataset))])

    train_path = savePath + '/train.csv'
    train.to_csv( train_path, index=False)

    validate_path = savePath + '/val.csv'
    validate.to_csv( validate_path, index=False)

    test_path = savePath + '/test.csv'
    test.to_csv(test_path, index=False)
