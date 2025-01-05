import os
from cs336_alignment.utils.io import load_jsonl_gz, save_list_to_jsonl

base_path = "/scratch/users/arjo/spring2024-assignment5-alignment/data" 
full_path = lambda x: os.path.join(base_path, x) 
paths = {
    "harmless-base": full_path("harmless-base/train"),
    "helpful-base": full_path("helpful-base/train"),
    "helpful-online": full_path("helpful-online/train"),
    "helpful-rejection-sampled": full_path("helpful-rejected-sampled/train")
}
out_path = full_path("anthropic/train")

def clean_sample(sample):
    if len(sample["chosen"].split("\n\nHuman: ")) > 2: # multi turn
        return None
    prompt_chosen = sample["chosen"].split("\n\nHuman: ")[1].split("\n\nAssistant: ")
    prompt = prompt_chosen[0]
    chosen = prompt_chosen[1]
    prompt_rejected = sample["rejected"].split("\n\nHuman: ")[1].split("\n\nAssistant: ")
    assert prompt == prompt_rejected[0]
    rejected = prompt_rejected[1]
    return (prompt, chosen, rejected)

datapoints = []
print_mode = False
num_prints=3
for dataset_name, path in paths.items():
    removed = 0
    i = 0
    dataset = load_jsonl_gz(path)
    n = len(dataset)
    print(f"{dataset_name}: num_samples: {n}")
    for sample in dataset:
        cleaned = clean_sample(sample)
        if cleaned is not None:
            prompt, chosen, rejected = cleaned
            datapoint = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "dataset": dataset_name
            }
            datapoints.append(datapoint)
            if print_mode and i < num_prints:
                print(f"@@@@@{dataset_name}@@@@@")
                print(f"propmt: {prompt}")
                print(f"chosen: {chosen}")
                print(f"rejected: {rejected}")
                print(f"----------")
                i+=1
        else:
            removed += 1
    print(f"{dataset_name}: removed: {removed}, kept: {n-removed}")
save_list_to_jsonl(out_path, datapoints)
print()
print(f"saved to: {out_path}.jsonl ...")
