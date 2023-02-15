import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
"""
Change this to the appropriate dataset
"""
class CSLMDataset(Dataset):
    def __init__(self, CSVPath, hparams, is_train:True):
        self.data = pd.read_csv(CSVPath)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.upstream_model, use_fast=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx, 0]
        label = self.iloc[idx, 1]
        features = self.tokenizer.encode(sentence, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True)
        return features, label