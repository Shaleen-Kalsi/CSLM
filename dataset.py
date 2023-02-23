import pandas as pd
import torch
from torch.nn.functional import one_hot
from transformers import AutoTokenizer
from torch.utils.data import Dataset
"""
Change this to the appropriate dataset
"""
class CSLMDataset(Dataset):
    def __init__(self, CSVPath, hparams, is_train:True):
        self.data = pd.read_csv(CSVPath)
        self.hparams = hparams
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.upstream_model, use_fast=True)
<<<<<<< HEAD
        self.labels2num = {"positive": 0, "negative": 1}

=======
        self.config = hparams
        
>>>>>>> ffe98dffc2ac0fac519f1e6278998a359e25d5b9
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx, 0]
<<<<<<< HEAD

        #convert label to one-hot
        label_num = self.labels2num[self.data.iloc[idx, 1]]
        target_label = one_hot(torch.tensor(label_num, dtype=torch.int64), self.hparams.num_classes).float() # throws an error that an index tensor is required with torch.int8, cross entropy requires float tensors
        encoding = self.tokenizer.encode_plus(sentence, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        print(encoding)
        print(target_label)
        return {"input_ids": encoding['input_ids'].flatten(), "attention_mask": encoding['attention_mask'], "labels": target_label} # flatten to 1D to make it work with BERT forward function
=======
        label = self.data.iloc[idx, 1]
        features = self.tokenizer.encode(sentence, max_length=self.config.max_seq_length, pad_to_max_length=True, truncation=True)
        return features, label
>>>>>>> ffe98dffc2ac0fac519f1e6278998a359e25d5b9
