import gzip
import json

LLAMA_8B_PATH = "/scratch/users/arjo/.llama/checkpoints/Llama3.1-8B"
LLAMA_70B_PATH = "/scratch/users/arjo/.llama/checkpoints/Llama3.1-70B"

def save_dict_to_json(file_path, data):
    with open(f"{file_path}.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

def save_list_to_jsonl(file_path, data):
    with open(f"{file_path}.jsonl", 'w', encoding='utf-8') as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + '\n')

def load_json(file_path):
    try:
        with open(f"{file_path}.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path}.json was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return None

def load_jsonl(file_path):
    data = []
    try:
        with open(f"{file_path}.jsonl", 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return data

def load_jsonl_gz(file_path):
    data = []
    try:
        with gzip.open(f"{file_path}.jsonl.gz", 'rt', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except OSError as e:
        print(f"Error reading the compressed file: {e}")
    return []
