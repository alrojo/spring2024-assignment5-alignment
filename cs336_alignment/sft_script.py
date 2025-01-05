import torch, argparse
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.utils.io import LLAMA_8B_PATH
from cs336_alignment.sft_format import SFTDataset, iterate_batches, SFT_PATH

EXPERIMENT_FOLDER = "/scratch/users/arjo/spring2024-assignment5-alignment/experiments"

def run_validation(_model, _dataset, slice_size, device):
    _model.eval()
    valid_loader = iterate_batches(_dataset, slice_size, False)
    losses = []
    with torch.no_grad():
        for valid_batch in valid_loader:
            input_ids = valid_batch["input_ids"].to(device)
            labels = train_batch["labels"].to(device)
            logits = model(input_ids).logits
            loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())
    avg_loss = np.mean(losses)
    avg_perplexity = math.exp(avg_loss)
    return avg_loss, avg_perplexity


###
# 1. Ability to configure and control the various model and optimizer hyperparameters
###

parser = argparse.ArgumentParser(description="training script: instruction tuning")
parser.add_argument("--experiment_name", type=str, help="name of experiment", required=True)
parser.add_argument("--lr", type=float, help="learning rate", defaults=1e-4)
parser.add_argument("--weight_decay", type=float, help="weight decay", defaults=2e-5)
parser.add_argument("--seq_len", type=int, help="context length", default=512)
parser.add_argument("--batch_size", type=int, help="batch size", default=32)
parser.add_argument("--slice_size", type=int, help="slice size", default=2)
parser.add_argument("--num_epochs", type=int, help="number of training epochs", default=2)
parser.add_argument("--print_every", type=int, help="number of iterations to print average training loss", default=25)
parser.add_argument("--eval_every", type=int, help="number of iterations to evaluate", default=250)
parser.add_argument("--device", type=int, help="gpu device", default=250)
args = parser.parse_args()

timestamp = timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s" % (args.experiment_name, timestamp)
experiment_dir = os.path.join(EXPERIMENT_FOLDER, experiment_id)
os.makedirs(experiment_dir)
print("experiment setup at %s" % experiment_dir)

###
# 2. Load LLAMA model and setup data loader
###

tokenizer = AutoTokenizer.from_pretrained(LLAMA_8B_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_8B_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.to(args.device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

train_path = os.path.join(SFT_PATH, "train.jsonl")
train_path = os.path.join(SFT_PATH, "test.jsonl")
train_dataset = SFTDataset(tokenizer, train_path, args.seq_len, True)
test_dataset = SFTDataset(tokenizer, test_path, args.seq_len, False)
num_slices = args.batch_size // args.slice_size
num_batches_epoch = len(train_dataset//args.batch_size)
num_batches = num_batches_epoch * args.num_epochs
warmup_steps = int(0.03 * num_batches)

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.0,
    end_factor=1.0,
    total_iters=warmup_steps
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=(num_batches - warmup_steps)
)

lr_scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

###
# 3. Train model
###
for e in range(num_epochs):
    train_loader = iterate_batches(train_dataset, slice_size, True)
    loss_tracker = []
    i = 0
    for train_batch in train_loader:
        model.train()
        input_ids = train_batch["input_ids"].to(args.device)
        labels = train_batch["labels"].to(args.device)
        logits = model(input_ids).logits
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        loss_tracker.append(loss.item())
        if (i+1) % num_slices == 0:
            optimizer.step() 
            lr_scheduler.step()
            optimizer.zero_grad()
        if i % args.print_every == 0:
            avg_train_loss = sum(loss_tracker)/len(loss_tracker)
            print("%d: Train loss: %.3f" % avg_train_loss)
            loss_tracker = []
        if i % args.valid_every == 0:
            valid_loss, valid_perplexity = run_validation(model, test_dataset, args.slice_size, args.device)
            print("%d: --Test loss: %.3f perplexity: %.3f" % (valid_loss, valid_perplexity))
        i += 1
            
###
# 4. Save fine-tuned model
###
model.save_pretrained(save_directory=experiment_dir)
tokenizer.save_pretrained(save_directory=experiment_dir)
