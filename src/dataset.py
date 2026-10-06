import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SimCSEDataset(Dataset):
    def __init__(self, csv_path, split="train", model_name="bert-base-uncased", max_length=64):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.sentences = self.df["sentence"].tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Tokenize the same sentence twice.
        # Different dropout masks during training will create the two views.
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }