import torch, argparse, os, random, copy
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.utils.io import load_jsonl
from cs336_alignment.sft_format import SFTDataset, iterate_batches, SFT_PATH
from cs336_alignment.dpo import per_instance_dpo_loss

BASE_PATH = "/scratch/users/arjo/spring2024-assignment5-alignment"
TRAIN_PATH = os.path.join(BASE_PATH, "data/anthropic")
EXPERIMENT_FOLDER = "/scratch/users/arjo/spring2024-assignment5-alignment/experiments/dpo"

def run_validation(llm, llm_ref, valid_set, tokenizer, args.device_llm, args.device_llm_ref)
    llm.eval()
    valid_loader = iterate_batches(_dataset, slice_size, False)
    losses = []
    with torch.no_grad():
        for sample in valid_set:
            prompt, chosen, rejected = sample["prompt"], sample["chosen"], sample["rejected"]
            loss = per_instance_dpo_loss(lm, lm_ref, tokenizer, args.beta_dpo,
                prompt, chosen, rejected)
            losses.append(loss.item())
    avg_loss = np.mean(losses)
    return avg_loss

###
# 1. Ability to configure and control the various model and optimizer hyperparameters
###

parser = argparse.ArgumentParser(description="training script: instruction tuning")
parser.add_argument("--experiment_name", type=str, help="name of experiment", required=True)
parser.add_argument("--pretrained_path", type=str, help="path to pretrained model", required=True)
parser.add_argument("--lr", type=float, help="learning rate", defaults=1e-6)
parser.add_argument("--weight_decay", type=float, help="weight decay", defaults=0)
parser.add_argument("--beta_dpo", type=float, help="weight decay", defaults=0.1)
parser.add_argument("--device_lm", type=int, help="gpu device", default=0)
parser.add_argument("--device_lm_ref", type=int, help="gpu device", default=1)
parser.add_argument("--print_every", type=int, help="number of iterations to print average training loss", default=25)
parser.add_argument("--eval_every", type=int, help="number of iterations to evaluate", default=250)
args = parser.parse_args()

timestamp = timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s" % (args.experiment_name, timestamp)
experiment_dir = os.path.join(EXPERIMENT_FOLDER, experiment_id)
os.makedirs(experiment_dir)
print("experiment setup at %s" % experiment_dir)

###
# 2. Load instruct tuned LLAMA model and setup dataset
###

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
llm = AutoModelForCausalLM.from_pretrained(
    args.pretrained_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
llm.cpu()
llm_ref = copy.deepcopy(llm)
llm_ref.eval()

llm.to(args.device_lm)
llm_ref.to(args.device_lm_ref)
optimizer = torch.optim.RMSProp(
    llm.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

dataset = load_jsonl(TRAIN_PATH) 
random.seed(1)
random.shuffle(dataset)
valid_set = dataset[:200]
train_set = dataset[200:]

###
# 3. Train model
###
loss_tracker = []
optimal_llm = None
best_valid = 1e9
for i, sample in enumerate(train_set):
    llm.train()
    optimizer.zero_grad()
    prompt, chosen, rejected = sample["prompt"], sample["chosen"], sample["rejected"]
    loss = per_instance_dpo_loss(lm, lm_ref, tokenizer, args.beta_dpo,
        prompt, chosen, rejected)
    loss.backward()
    loss_tracker.append(loss.item())
    optimizer.step() 
    if i % args.print_every == 0:
        avg_train_loss = sum(loss_tracker)/len(loss_tracker)
        print("%d: Train loss: %.3f" % avg_train_loss)
        loss_tracker = []
    if i % args.valid_every == 0:
        valid_loss = run_validation(llm, llm_ref, valid_set, tokenizer, args.device_llm, args.device_llm_ref)
        if best_valid < valid_loss:
            llm.cpu()
            optimal_llm = copy.deepcopy(llm)
            llm.device(args.device_llm)
            print(f"updaing model in {experiment_dir} ...")
        print("%d: --Test loss: %.3f perplexity: %.3f" % (valid_loss, valid_perplexity))

###
# 4. Save fine-tuned model
###
optimal_llm.save_pretrained(save_directory=experiment_dir)
tokenizer.save_pretrained(save_directory=experiment_dir)
