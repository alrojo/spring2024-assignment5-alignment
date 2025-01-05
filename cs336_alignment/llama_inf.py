import subprocess, argparse
from time import time
from vllm import LLM, SamplingParams
from cs336_alignment.mmlu_format import mmlu
from cs336_alignment.gsm8k_format import gsm8k
from cs336_alignment.alpaca_format import alpaca
from cs336_alignment.sst_format import sst
from utils.io import *
from utils.string_format import format_instruct_inference

parser = argparse.ArgumentParser(description="training script: instruction tuning")
parser.add_argument("--model_path", type=str, help="zero or path to instruct", required=True)
parser.add_argument("--max_tokens", type=int, help="slice size", default=1024)
parser.add_argument("--device", type=int, help="gpu device", default=250)
args = parser.parse_args()
if args.model == "zero":
    model_path = LLAMA_8B_PATH
    model_name = "zero"
else:
    model_path = args.model_path
    model_name = "instruct"

out_folder = "/scratch/users/arjo/spring2024-assignment5-alignment/out"
alpaca_save_path = "/scratch/users/arjo/spring2024-assignment5-alignment/scripts/alpaca_eval_vllm_llama3_70b_fn/annotations_seed0_configs"

class llama_inf:
    def __init__(self, model_name, model_path, max_tokens=1024):
        self.model_name = model_name
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop["\n"])
        self.model_name = os.path.basename(model_path)
        self.llm = LLM(model=model_path)

    def sst_inf(self):
        out_path = "%s/simple_safety_tests/%s_simple_safety_tests" % (out_folder, self.model_name)
        df = sst.load_samples("test", model_name=self.model_name)
        predictions = []
        for row in df.itertuples():
            text = row.prompts_final
            prompt = text if self.model_name="zero" else format_instruct_inference(text)
            generation, time_elapsed = self.generate(prompt)
            prediction = {"prompts_final": text, "output": generation}
            times.append(time_elapsed)
            predictions.append(prediction)
        to_dump = predictions
        save_list_to_jsonl(out_path, to_dump)
        return times, out_path

    def sst_report(self):
        print("model: %" % self.model_name)
        print("dataset: sst")
        times, out_path = self.alpaca_inf()
        avg_time = sum(times)/len(times)
        print("the average amount of time per example is: %.2f" avg_time)
        results_path = "%s/simple_safety_tests/results" % (out_folder)
        command = [
            "scripts/evaluate_safety.py",
            "--input_path", out_path,
            "--model-name-or-path", LLAMA_70B_PATH
            "--num-gpus", "2",
            "--output-path", results_path, 
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Command executed successfully.")
            print("Output:", result.stdout)
            print("Errors (if any):", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Command failed with error:")
            print(e.stderr)
        results = load_json(results_path)
        print(results)

    def alpaca_inf(self):
        out_path = "%s/alpaca_eval/%s_alpaca_eval" % (out_folder, self.model_name)
        examples = gsm8k.load_samples("test")
        for example in examples:
            text = example["instruction"]
            prompt = text if self.model_name="zero" else format_instruct_inference(text)
            generation, time_elapsed = self.generate(prompt)
            example["output"] = generation
            example["generator"] = self.model_name
            times.append(time_elapsed)
        alpaca_results = load_json(alpaca_save_path)
        to_dump = examples
        save_dict_to_json(out_path, to_dump)
        return times, out_path

    def alpaca_report(self):
        print("model: %" % self.model_name)
        print("dataset: alpaca")
        times, out_path = self.alpaca_inf()
        avg_time = sum(times)/len(times)
        print("the average amount of time per example is: %.2f" avg_time)
        command = [
            "scripts/alpaca_eval",
            "--model_outputs", "out/alpaca_eval/alpaca_eval.json",
            "--annotators_config", "scripts/alpaca_eval_vllm_llama3_70b_fn",
            "--base-dir", "."
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Command executed successfully.")
            print("Output:", result.stdout)
            print("Errors (if any):", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Command failed with error:")
            print(e.stderr)
        alpaca_results = load_json(alpaca_save_path)
        print(alpaca_results)

    def mmlu_inf(self, set_type="val"):
        out_path_ext = "%s/mmlu/%s_%s" % (out_folder, self.model_name, set_type)
        samples_per_path = mmlu.load_samples("val")
        total_predictions = []
        total_labels = []
        failed_attempts = []
        times = []
        incorrect_samples = []
        for file_name, (rows, labels, samples) in samples_per_path.items():
            out_path = os.join.path(out_path_ext, file_name)
            predictions = []
            for label, sample in zip(labels, samples):
                generation, time_elapsed = self.generate(sample)
                times.append(time_elapsed)
                prediction = mmlu.parse_mmlu_response(generation)
                if prediction is None:
                    failed_attemps.append((generation, sample))
                elif prediction != label:
                    incorrect_samples.append((generation, sample)) 
                predictions.append(prediction)
            to_dump = {"rows": rows, "labels": labels, "predictions": predictions}
            save_dict_to_json(out_path, to_dump)
            total_predictions+=predictions
            total_labels+=labels
        accuracy = eval_metrics(total_labels, total_predictions)
        return accuracy, failed_attemps, times, incorrect_samples

    def report(self, dataset)
        print("model: %" % self.model_name)
        times, out_path = self.alpaca_inf()
        assert dataset in ["mmlu", "gsm8k"]
        print("dataset: " % dataset)
        data = llama.mmlu_inf("val") if dataset == "mmlu" else llama.gsm8k("test")
        accuracy, failed_attemps, times = data 
        print("%s_accuracy: %.3f." % (dataset, accuracy))
        print("below are some failed attempts")
        for i in range(min(len(failed_attemps), 3)):
            generation, sample = failed_attemps[i]
            print("%d @@@@@" %(i+1))
            print(sample)
            print("@@@@@")
            print(generation)
            print()
        avg_time = sum(times)/len(times)
        print("the average amount of time per example is: %.2f" avg_time)
        for i in range(min(len(incorrect_samples), 10)):
            generation, sample = incorrect_samples[i]
            print("%d @@@@@" %(i+1))
            print(sample)
            print("@@@@@")
            print(generation)
            print()

    def gsm8k_inf(self, set_type="val"):
        out_path = "%s/gsm8k/%s_%s" % (out_folder, self.model_name, set_type)
        samples, questions, answers, labels = gsm8k.load_samples("test")
        predictions = []
        failed_attempts = []
        times = []
        incorrect_samples = []
        for (sample, question, answer, label) in zip(samples, questions, answers, labels):
            generation, time_elapsed = self.generate(sample)
            times.append(time_elapsed)
            prediction = mmlu.parse_mmlu_response(generation)
            if prediction is None:
                failed_attemps.append((generation, sample))
            elif prediction != label:
                incorrect_samples.append((generation, sample)) 
            predictions.append(prediction)
        to_dump = {"questions": questions, "labels": labels, "predictions": predictions}
        save_dict_to_json(out_path, to_dump)
        accuracy = eval_metrics(labels, predictions)
        return accuracy, failed_attemps, times, incorrect_samples

    def redteaming(self):
        while True:
            text = input("write a prompt: ")
            prompt = format_instruct_inference(text)
            generation, _ = self.generate(prompt)
            print("generated output: %s" % generation)
            cont = input("continue? y/n")
            if cont == "n":
                break

    def generate(prompts):
        start_time = time()
        generation = self.llm.generate(prompts, self.sampling_params)[0].text
        time_elapsed = time() - start_time
        return generation, time_elapsed 

if __name__=="__main__":
    llama = llama_inf(model_name, model_path, max_tokens=args.max_tokens)
    print(f"---REPORT: mmlu {model_name}---")
    llama.report("mmlu")
    print(f"---REPORT: gsm8k {model_name}---")
    llama.report("gsm8k")
    print(f"---REPORT: alpaca {model_name}---")
    llama.alpaca_report()
    print(f"---REPORT: sst {model_name}---")
    llama.sst_report()
    print(f"---Redteaming: {model_name}---")
    llama.redteaming()
