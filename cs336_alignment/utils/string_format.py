def format_instruct_inference(prompt: str) -> str:
    return (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n"
        f"{prompt}\n"
        f"\n"
        f"### Response:\n"
    )

def format_instruct_training(
    prompt: str,
    response: str) -> str:
    return (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n"
        f"{prompt}\n"
        f"\n"
        f"### Response:\n"
        f"{response}"
    )
