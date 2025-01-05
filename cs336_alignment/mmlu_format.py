from typing import Any
import re, os
from glob import glob
import pandas as pd
from cs336_alignment.utils.string_format import format_instruct_inference

MMLU_PATH = "/scratch/users/arjo/spring2024-assignment5-alignment/data/mmlu"
class mmlu:
    def load_data(set_type, model_name="zero"):
        assert set_type in ["dev", "test", "val"] 
        assert model_name in ["zero", "instruct"]
        string_format = string_formats[model_name]
        paths = glob("%s/%s/*" % (MMLU_PATH, set_type) )
        samples_per_path = {}
        for path in paths:
            file_name = os.path.basename(path)
            subject = os.path.splitext(file_name)[0]
            df = pd.read_csv(path, header=None)
            samples = []
            labels = []
            rows = []
            for index, row in df.iterrows():
                question = str(row[0])
                options = list(row[1:5])
                answer = str(row[5])
                sample = string_format(subject, question, options, answer)
                samples.append(sample)
                labels.append(answer)
                rows.append(row)
            samples_per_path[file_name] = (labels, samples, rows)
        return samples_per_path 

    def format_zero(subject, question, options, answer):
        return (
            f"Answer the following multiple choice question about {subject}. Respond with a single\n"
            f"- sentence of the form \"The correct answer is _\", filling the blank with the letter\n"
            f"- corresponding to the correct answer (i.e., A, B, C or D).\n"
            f"\n"
            f"Question: {question}\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}\n"
            f"C. {options[2]}\n"
            f"D. {options[3]}\n"
            f"Answer:"
        )

    def format_instruct(subject, question, options, answer):
        prompt = (
            f"Answer the following multiple choice question about {subject}. Respond with a single\n"
            f"- sentence of the form \"The correct answer is _\", filling the blank with the letter\n"
            f"- corresponding to the correct answer (i.e., A, B, C or D).\n"
            f"\n"
            f"Question: {question}\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}\n"
            f"C. {options[2]}\n"
            f"D. {options[3]}\n"
        )
        return format_instruct_inference(prompt)

    def eval_metrics(labels, predictions):
        n = len(labels)
        k = sum([int(label==prediction) for (label, prediction) in zip(labels, predictions)])
        acc = k/n
        return acc

    def parse_mmlu_response(
        model_output: str,
    ) -> str | None:
        answer_options = ["A", "B", "C", "D"]
        tokens = re.findall(r'\w+|[^\w\s]', model_output)
        for token in tokens:
            if token in answer_options:
                return token
        return None

mmlu.string_formats = {"zero": mmlu.format_zero, "instruct": mmlu.format_instruct}
if __name__=="__main__":
    mmlu.load_data("val")
