import pandas as pd
import torch
from torch.nn.functional import one_hot
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import csv
import random
"""
Change this to the appropriate dataset
"""
class CSLMDataset(Dataset):
    def __init__(self, CSVPath, hparams, is_train:True, CSVPathMixup=None, apply_mixup=False, same_class_mixup=False):
        self.data = pd.read_csv(CSVPath)
        self.hparams = hparams
        self.apply_mixup = apply_mixup
        self.same_class_mixup = same_class_mixup
        if self.apply_mixup:
            self.data = pd.read_csv(CSVPath)
            #used if we are performing mixup across languages
            #self.mixup = pd.read_csv(CSVPathMixup)
            #if we are choosing sentences to mixup from the same class
            if self.same_class_mixup:
                self.data_pos = self.data.loc[self.data["label"]== "positive"]
                self.data_neg = self.data.loc[self.data["label"]== "negative"]
                self.data_neu = self.data.loc[self.data["label"]== "neutral"]
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.upstream_model, use_fast=True)
        self.labels2num = {"positive": 0, "negative": 1, "neutral": 2}
        self.mixup_file_path = self.hparams.mixup_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.apply_mixup == True:
            mixup_data = None            
            sentence = self.data.iloc[idx, 0]
            #convert label to one-hot
            label = self.data.iloc[idx, 1]
            label_num = self.labels2num[label]

            if self.same_class_mixup:
                if label_num ==0:
                    mixup_data = self.data_pos.sample(n= 1)
                elif label_num ==1:
                    mixup_data = self.data_neg.sample(n= 1)
                elif label_num==2:
                    mixup_data = self.data_neu.sample(n= 1)
            else:
                mixup_data = self.data.sample(n= 1)

            target_label = one_hot(torch.tensor(label_num, dtype=torch.int64), self.hparams.num_classes).float() # throws an error that an index tensor is required with torch.int8, cross entropy requires float tensors
            encoding = self.tokenizer.encode_plus(sentence, max_length=128, padding = 'max_length', truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors='pt')

            sentence_mixup = mixup_data.iloc[0,0]
            label_mixup = mixup_data.iloc[0,1]
            label_num_mixup = self.labels2num[label_mixup]
            target_label_mixup = one_hot(torch.tensor(label_num_mixup, dtype=torch.int64), self.hparams.num_classes).float() # throws an error that an index tensor is required with torch.int8, cross entropy requires float tensors
            encoding_mixup = self.tokenizer.encode_plus(sentence_mixup, max_length=128, padding = 'max_length', truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors='pt')

            #saving sentences being mixed up into a file
            mixup_sentences = [sentence, label, sentence_mixup, label_mixup]
            # open the file in the write mode
            with open(self.mixup_file_path, 'a', encoding='UTF8') as f:
            # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file
                writer.writerow(mixup_sentences)

            return {"input_ids_x": encoding['input_ids'].flatten(), "input_ids_mixup_x": encoding_mixup['input_ids'].flatten(), "attention_mask_x": encoding['attention_mask'].squeeze(), "attention_mask_mixup_x": encoding_mixup['attention_mask'].squeeze(), "labels_x": target_label, "labels_mixup_x": target_label_mixup} # flatten to 1D to make it work with BERT forward function
        
        else:
            sentence = self.data.iloc[idx, 0]

            #convert label to one-hot
            label_num = self.labels2num[self.data.iloc[idx, 1]]
            target_label = one_hot(torch.tensor(label_num, dtype=torch.int64), self.hparams.num_classes).float() # throws an error that an index tensor is required with torch.int8, cross entropy requires float tensors
            encoding = self.tokenizer.encode_plus(sentence, max_length=128, padding = 'max_length', truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
            return {"input_ids": encoding['input_ids'].flatten(), "attention_mask": encoding['attention_mask'].squeeze(), "labels": target_label} # flatten to 1D to make it work with BERT forward function
