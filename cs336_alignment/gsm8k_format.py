from typing import Any
import re, os
from glob import glob
import pandas as pd
from cs336_alignment.utils.io import load_jsonl
from cs336_alignment.utils.string_format import format_instruct_inference

GSM8K_PATH = "/scratch/users/arjo/spring2024-assignment5-alignment/data/gsm8k"

class gsm8k:
    def load_data(set_type, model_name="zero"):
        assert set_type in ["train", "test"] 
        assert model_name in ["zero", "instruct"]
        string_format = string_formats[model_name]
        path = os.path.join(GSM8K_PATH, set_type)
        rows = load_jsonl(path)
        samples = []
        questions = []
        answers = []
        labels = []
        for row in rows:
            question = row["question"]
            answer = row["question"]
            label = gsm8k.parse_gsm8k_response(answer)
            sample = format_sample(row)
            samples.append(sample)
            questions.append(question)
            answers.append(answer)
            labels.append(label)
        return samples, questions, answers, labels 

    def format_zero(question, answer):
        return (
            f"{question}\n"
            f"Answer:"
        )

    def format_instruct(question):
        return format_instruct_inference(question)

    def eval_metrics(labels, predictions):
        n = len(labels)
        k = sum([int(label==prediction) for (label, prediction) in zip(labels, predictions)])
        acc = k/n
        return acc

    def parse_gsm8k_response(
        model_output: str,
    ) -> str | None:
        match = re.search(r'.*\b(\d+(\.\d+)?)\b(?!.*\b\d+(\.\d+)?\b)', model_output)
        return match.group(1) if match else None

gsm8k.string_formats = {"zero": gsm8k.format_zero, "instruct": gsm8k.format_instruct}

if __name__=="__main__":
    gsm8k.load_data("test")
