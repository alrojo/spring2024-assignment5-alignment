import torch, os, random
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from typing import Any
from glob import glob
import pandas as pd
from cs336_alignment.utils.io import load_jsonl
from cs336_alignment.utils.string_format import format_instruct_inference, format_instruct_training

SFT_PATH = "/scratch/users/arjo/spring2024-assignment5-alignment/data/ultrachat"

class SFTDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            dataset_path: str | os.PathLike,
            seq_length: int,
            shuffle: bool
        ):
        # load data
        dataset_path = os.path.splitext(dataset_path)[0]
        samples = self.load_data(dataset_path)
        # shuffle data
        if shuffle:
            random.shuffle(samples)
        # concat into large string
        all_samples = "<|end_of_text|><|begin_of_text|>".join(samples)
        # tokenize data
        all_samples_encoded = tokenizer.encode(all_samples)
        # clip into seq_length sized intervals 
        num_samples = (len(all_samples_encoded)-1)//seq_length
        self.inputs = [all_samples_encoded[seq_length*i:seq_length*(i+1)] for i in range(num_samples)]
        self.outputs = [all_samples_encoded[seq_length*i+1:seq_length*(i+1)+1] for i in range(num_samples)]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        if not (i < self.__len__()):
            raise IndexError(f"Index {i} is out of range for dataset of length {len(self)}.") 
        return {"input_ids": torch.tensor(self.inputs[i], dtype=torch.long),
                "labels": torch.tensor(self.outputs[i], dtype=torch.long)}

    def format_training(self, prompt, answer):
        formatted = format_instruct_training(prompt, answer)
        return f"{formatted}"

    def format_inference(self, prompt):
        formatted = format_instruct_inference(prompt)
        return formatted 

    def load_data(self, path):
        rows = load_jsonl(path)
        samples = []
        for row in rows:
            sample = self.format_training(row["prompt"], row["response"])
            samples.append(sample)
        return samples

def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
