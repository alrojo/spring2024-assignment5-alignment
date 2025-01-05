import gzip
import json
import random
PATH_SFT = "/scratch/users/arjo/spring2024-assignment5-alignment/data/ultrachat/test.jsonl.gz" 

def load_random_sft(file_path, num_examples=10):
    data = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        random_examples = random.sample(data, num_examples)
        return random_examples
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except OSError as e:
        print(f"Error reading the compressed file: {e}")
    return []

# Example usage
if __name__=="__main__":
    samples = load_random_sft(PATH_SFT, num_examples=10)
    for e, sample in enumerate(samples):
        print("%d:--------" % (e+1))
        print("----PROMPT----")
        print(sample["prompt"])
        print("----RESPONSE----")
        print(sample["response"])
