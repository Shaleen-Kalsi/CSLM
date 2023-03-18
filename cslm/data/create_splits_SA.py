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
import emoji

def sa_hinglish_to_sentence_label(csv_path):
    """
    Function to convert sa_hinglish CoNLL to sentence label format
    Args:
        csv_path: Path to input CSV
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

    return formatted

def sa_english_to_sentence_label(csv_path):
    """
    Function to convert sa_english dataframe with extra columns to sentence label
    Args:
        csv_path: Path to input CSV
    """
    dataset = pd.read_csv(csv_path, sep=',', names = ["label", "id", "date", "query", "name", "sentence"], encoding = "latin-1")
    dataset = dataset.drop(columns = ["id", "date", "query", "name"])
    dataset = dataset.dropna(subset = "sentence").reset_index(drop=True)
    dataset = dataset.replace({"label":{4: "positive", 0: "negative"}})
    dataset = dataset.reindex(columns = ["sentence", "label"])
    dataset = dataset.sample(n = 100000, random_state=42).reset_index(drop=True) #16 million samples are too heavy
    return dataset

def sa_drav_to_sentence_label(csv_path):
    """
    Function to convert sa_tamil and sa_mlayalam dataframe to sentence label removing redundant labels
    Args:
        csv_path: Path to input CSV
    """
    dataset = pd.read_csv(csv_path, sep=',')
    #dropping samples that are not in language of choice
    dataset = dataset.drop(dataset[(dataset.label == "not-Tamil ") | (dataset.label == "not-malayalam ")].index)
    #changing unknown and mixed to neutral 
    dataset = dataset.replace({"label":{"unknown_state ":"neutral", "Mixed_feelings ":"neutral", "Positive ": "positive", "Negative ": "negative"}})
    dataset = dataset.dropna(subset = "sentence").reset_index(drop=True)

    return dataset
    
def preprocess(dataset):
    """
    Function to preprocess sentences: Remove urls, removes hashtag symbols, converts emojis to text versions as in paper.
    Args:
        dataset: Input Dataframe
    Returns:
        Preprocessed Dataframe
    """

    for i in range(dataset.shape[0]):
        sent = dataset.loc[i, "sentence"]
        #not changed to lowercase in paper
        #sent = sent.lower()
        #removing URLs
        #specific commands because of CoNLL formatting
        sent = re.sub(r"\shttps // t . co / \S*\s", "", sent)
        sent = re.sub(r"\shttps // t co / \S*\s", "", sent)
        sent = re.sub(r"\shttps // tco / \S*\s", "", sent)
        sent = re.sub(r"\shttps // t . co / \S*", "", sent)
        sent = re.sub(r"\shttps // t co / \S*", "", sent)
        sent = re.sub(r"\shttps // tco / \S*", "", sent)
        sent = re.sub(r"https // t . co / \S*", "", sent)
        sent = re.sub(r"https // t co / \S*", "", sent)
        sent = re.sub(r"https // tco / \S*", "", sent)
        #already sentence label dataset only requires the next command but above commands should not effect anything
        sent = re.sub(r"https://\S*\s", "", sent)
        sent = re.sub(r"http://\S*\s", "", sent)
        sent = re.sub(r"www.\S*\s", "", sent)
        sent = re.sub(r"\s\S*.com\S*\s", "", sent)
        #edge cases for CoNLL
        sent = re.sub(r"http.*.", "", sent)
        #punctuation not removed in the paper, only hastag symbols
        #sent = re.sub(r'[^\w\s]', '', sent).strip() 
        sent = re.sub(r'#\s', '', sent)
        sent = re.sub(r'#', '', sent)
        #emojis converted ro text versions
        sent = emoji.demojize(sent, version = 13.0)
        #adding extra to make mentions no spaced in CoNLL
        sent = re.sub(r'@\s', r"@", sent)
        dataset.loc[i, "sentence"] = sent
    return dataset


def main():
    args = docopt(__doc__)
    path = args['--csv']
    data = path[path.rfind("/")+1:path.rfind(".")]
    if data == "sa_hinglish_raw":
        dataset = sa_hinglish_to_sentence_label(args['--csv']) 
        dataset = preprocess(dataset)
        out_path = path[:path.rfind("_")] + ".csv"
        dataset.to_csv(out_path, index=False)
    elif data == "sa_english_raw":
        dataset = sa_english_to_sentence_label(args['--csv']) 
        dataset = preprocess(dataset)
        out_path = path[:path.rfind("_")] + ".csv"
        dataset.to_csv(out_path, index=False)
    elif data == "sa_tamil_raw":
        dataset = sa_drav_to_sentence_label(args['--csv']) 
        dataset = preprocess(dataset)
        out_path = path[:path.rfind("_")] + ".csv"
        dataset.to_csv(out_path, index=False)
    elif data == "sa_malayalam_raw":
        dataset = sa_drav_to_sentence_label(args['--csv']) 
        dataset = preprocess(dataset)
        out_path = path[:path.rfind("_")] + ".csv"
        dataset.to_csv(out_path, index=False)

    #creating splits
    #80, 10, 10 split
    train, validate, test = np.split(dataset.sample(frac=1, random_state=42), [int(.8*len(dataset)), int(.9*len(dataset))])

    out_dir = path[:path.rfind("_")]
    
    train_path = out_dir + "_train.csv"
    train.to_csv(train_path, index=False)

    validate_path = out_dir + "_val.csv"
    validate.to_csv(validate_path, index=False)

    test_path = out_dir + "_test.csv"
    test.to_csv(test_path, index=False)