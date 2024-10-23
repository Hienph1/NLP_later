# !pip install datasets

"""Import dependencies"""

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

"""Create DataLoader and Train the Tokenizer"""

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt') # Ensure PyTorch Tensor output

        input_ids = encoding['input_ids'].squeeze()

        # Assuming you want to use the input_ids as labels for language modeling


        labels = input_ids.clone()

        labels[:-1] = input_ids[1:]  # Shift labels
        return input_ids, labels  # Return both input_ids and labels

# Function to get training data
def get_training_corpus():
    dataset = load_dataset("text", data_files={"train": "/content/cleaned_data.txt"})
    for i in range(0, len(dataset["train"]), 1000):
        yield dataset["train"][i : i + 1000]["text"]

# Load the base tokenizer
base_tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-v0.3-bnb-4bit")

# Train the new tokenizer
new_tokenizer = base_tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size=1000)

# Save the new tokenizer
new_tokenizer.save_pretrained("new_tokenizer")

# Test the new tokenizer
test_text = "الهجرة كلمة تسمعها بزاف في بلادي "
encoded = new_tokenizer.encode(test_text)
decoded = new_tokenizer.decode(encoded)

print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")


# Add padding token if it doesn't exist
if new_tokenizer.pad_token is None:
    new_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Ensure pad_token_id is valid
if new_tokenizer.pad_token_id is None or new_tokenizer.pad_token_id >= new_tokenizer.vocab_size:
    new_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    new_tokenizer.pad_token_id = new_tokenizer.vocab_size - 1  # Use the last valid token ID as padding

"""Create the Model"""

# Configuration parameters
n_embd = 4096
n_head = 32
n_layer = 32
head_size = 128
dropout = 0.1
block_size = 32768
vocab_size = new_tokenizer.vocab_size
num_experts = 8
top_k = 2
# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Expert(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

"""The Router class manages the allocation of input data to different experts in a mixture-of-experts model, it begins by computing logits that determine how likely it is for each input to be assigned to each expert, and adds noise to these logits to promote robustness and avoid overfitting."""

class Router(nn.Module):
    def __init__(self, n_embed: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output: torch.Tensor):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class MoE(nn.Module):
    def __init__(self, n_embed: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = Router(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output

class Block(nn.Module):
    def __init__(self, n_embed: int, n_head: int, num_experts: int, top_k: int):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = MoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x

class MOELanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, num_experts, top_k) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = MOELanguageModel().to(device)

"""Trainer"""

def train(model: SMoELanguageModel,
          train_data: DataLoader,
          val_data: DataLoader,
          optimizer: torch.optim.Optimizer,
          total_steps: int = 10000,
          device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
          clip_grad_norm: float = 1.0,
          lr_scheduler=None,
          eval_interval: int = 1000,
          save_path: str = "MOE_darija.pt"):

    model = model.to(device)
    model.train()

    print("Training...")
    step = 0
    total_loss = 0.0
    start_time = time.time()

    while step < total_steps:
        for batch in train_data:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(input_ids, labels)

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            # Update weights
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step(loss.detach().item())

            total_loss += loss.item()
            step += 1

            # Print step-wise loss and elapsed time periodically
            if step % eval_interval == 0:
                avg_loss = total_loss / eval_interval
                elapsed_time = time.time() - start_time
                print(f"Step {step}/{total_steps} | Average loss: {avg_loss:.4f} | Elapsed time: {elapsed_time:.2f}s")
                total_loss = 0.0
                start_time = time.time()

                # Evaluation Phase
                model.eval()
                eval_loss = 0
                with torch.no_grad():
                    for val_batch in val_data:
                        input_ids, labels = val_batch
                        input_ids, labels = input_ids.to(device), labels.to(device)
                        logits, loss = model(input_ids, labels)
                        eval_loss += loss.item()

                avg_eval_loss = eval_loss / len(val_data)
                print(f"Step {step}, Evaluation Loss: {avg_eval_loss:.4f}")
                model.train()

            # Stop if total steps reached
            if step >= total_steps:
                break

        # Early stop if steps are already reached within epoch
        if step >= total_steps:
            break

    torch.save(model.state_dict(), save_path)
    print("Training complete!")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

train_dataset = TextDataset('/content/cleaned_data.txt',
                            new_tokenizer,
                            max_length=512)
train_loader = DataLoader(train_dataset,
                          batch_size=4,
                          shuffle=False)

val_dataset = TextDataset('/content/validation.txt',
                            new_tokenizer,
                            max_length=512)
val_loader = DataLoader(val_dataset,
                          batch_size=4,
                          shuffle=False)

train(model,
      train_loader,
      val_loader,
      optimizer)

"""Inference"""

max_length = 30
num_return_sequences = 10

# Encode the input text
tokens = new_tokenizer.encode("راك ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

# Generate sequences
ids = model.generate(tokens,max_length)

# Print the generated text
for i in range(num_return_sequences):
    decoded = new_tokenizer.decode(ids[i], skip_special_tokens=True)
    print(">", decoded)