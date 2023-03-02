"""
Script to format datasets into appropriate sentence|label form.

Usage:
    format_data --csv=<csv> 

Options:
    --csv=<csv>    Input CSV file path
"""

import os
import re
import pandas as pd
import numpy as np
from docopt import docopt

def sa_to_sentence_label(csv_path):
    """
    Function to split dataset into train, test and validation sets
    Args:
        csv_path: Path to input CSV
        out_dir: Path to dir to save the splits
    """

    dataset = pd.read_csv(csv_path, sep='\t', names = ["text", "lang", "label"])
    dataset = dataset.dropna(subset = "text").reset_index(drop=True) #three cases of spaces which are coded as nans
    formatted = pd.DataFrame(columns = ["sentence", "label"]) #new dataste to store in

    #main code to combine tokenisied data into space separated sentences, including special characters and punctuation
    i = 0
    count = 0
    while i < dataset.shape[0]:
        if dataset.loc[i, "text"] == "meta":
            if i > 0:
                formatted = pd.concat([formatted, pd.DataFrame({"sentence": sentence, "label": label}, index = [count])], axis = 0)
                count += 1
            sentence = dataset.loc[i+1, "text"]
            label = dataset.loc[i, "label"]
            i+=2
        else:
            sentence = sentence + " " + dataset.loc[i, "text"]
            if i == dataset.shape[0] - 1:
                formatted = pd.concat([formatted, pd.DataFrame({"sentence": sentence, "label": label}, index = [count])], axis = 0)
            i+=1    

    #print(formatted.iloc[99, :]) 
    formatted = formatted.drop(99) #corrupted data, removed 
    out_path = csv_path[:csv_path.rfind(".")] + "_reformatted.csv"
    formatted.to_csv(out_path, index=False)


def main():
    args = docopt(__doc__)
    path = args['--csv']
    data = path[path.rfind("/")+1:path.rfind(".")]
    if data == "sa_hinglish":
        sa_to_sentence_label(args['--csv']) 