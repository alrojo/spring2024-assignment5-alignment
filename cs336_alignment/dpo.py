import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from cs336_alignment.utils.string_format import format_instruct_training
format_instruct = format_instruct_training

def get_log_likelihood(lm, text, tokenizer):
    device = next(lm.parameters()).device
    encoding = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    encoding.to(device)
    outputs = lm(**encoding)
    logits = outputs.logits[:, :-1, :]
    targets = encoding["input_ids"][:, 1:].squeeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs[0,torch.arange(len(targets)), targets]
    return selected_log_probs.sum(dim=-1)

def per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    # Encode text
    chosen_text   = format_instruct(prompt, response_chosen)
    rejected_text = format_instruct(prompt, response_rejected)

    eos = tokenizer.eos_token
    if eos is not None:
        chosen_text   += eos
        rejected_text += eos
    
    # inference text
    lm_device = next(lm.parameters()).device
    lm_ref_chosen_logprob = get_log_likelihood(lm_ref, chosen_text, tokenizer).to(lm_device)
    lm_ref_rejected_logprob = get_log_likelihood(lm_ref, rejected_text, tokenizer).to(lm_device)
    lm_chosen_logprob = get_log_likelihood(lm, chosen_text, tokenizer)
    lm_rejected_logprob = get_log_likelihood(lm, rejected_text, tokenizer)
    
    # compute DPO
    chosen_log_prob = lm_chosen_logprob - lm_ref_chosen_logprob
    rejected_log_prob = lm_rejected_logprob - lm_ref_rejected_logprob
    reward = beta * (chosen_log_prob - rejected_log_prob)
    dpo = - F.logsigmoid(reward) 
    return dpo
