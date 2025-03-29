import sys
import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 32
block_size = 8
# vocab_size = 65
learning_rate = 1e-3

n_steps = 8000
eval_iters = 500
n_embd = 32
torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")
# read_text
with open("input.txt", "r") as f:
    text = f.read()

# print(f"length of text: {len(text)}")
# print(f"first 100 characters: {text[0:100]}")

#character encoding
char_set = sorted(list(set(text)))
print(f"length of char_set: {len(char_set)}\n")

vocab_size = len(char_set)
idx2char = {i:ch for i, ch in enumerate(char_set)}
char2idx  = {ch:i for i, ch in enumerate(char_set)}
encode = lambda s: [char2idx[c] for c in s]
decode = lambda l: "".join([idx2char[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long).to(device)
print(data.shape, data.dtype)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# batch_size = 4 # how many independent sequences will we process in parallel?
# block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def evaluate():
    model.eval()
    out_loss = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): 
            xval , yval = get_batch(split)
            xval.to(device)
            yval.to(device)
            logits, loss = model(xval, yval)
            losses[k] = loss.item()
        out_loss[split] = losses.mean()
    model.train()
    return out_loss

xb, yb = get_batch('train')

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.query(x)   #(B,T,head_size)
        k = self.key(x)     #(B,T,head_size)
        v = self.value(x)   #(B,T,head_size)
        # print(q.device, k.device, v.device)

        # TODO two n_embd, head_size, which one is used to normalize, ( according to me head_size)
        wei = q @ k.transpose(-2,-1) * C**-0.5    # (B,T,head_size) @ (B, head_size, T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))

        affin = F.softmax(wei, dim=-1)

        out = affin @ v # (B,T,T) @ (B,T,head_size)

        return out

        
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([ Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)  # as each head works independently



class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # as at one moment max token is block size
        self.sa_head = MultiHeadAttention(4, n_embd//4)   
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        #idx: B,T , targets: B,T
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (B,T,n_embd)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))   # (T,n_embd)
        x = token_embeddings + position_embeddings # (B,T,n_embd)
        x = self.sa_head(x)        # (B, T, n_embd)
        logits = self.lm_head(x)   # (B, T, vocab_size)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: B,T
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:, -block_size:])
            #focus on last time step as bigram model
            logits = logits[:, -1, :]  # becomes B, C
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) # B,1
            idx = torch.cat((idx, idx_next), dim=1) # B,T+1
        return idx # B, T+max_new_tokens

model = BigramLanguageModel().to(device)
xb.to(device)
yb.to(device)
logits, loss = model(xb, yb)
print(f"logits.shape: {logits.shape}")
print(f"loss: {loss}") # -ln(1/65) = 4.174387   

print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))



# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for steps in range(n_steps): # increase number of steps for good results...
    # sample a batch of data
    xb, yb = get_batch('train')
    xb.to(device)
    yb.to(device)   

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % 500 == 0:
        eval_loss = evaluate()
        print(f"step {steps}: train_loss {eval_loss['train']:.4f}, val_loss {eval_loss['val']:.4f}")

print(f"final loss: {loss.item()} after {n_steps} steps.\n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
sample_gen = decode(model.generate(idx = context, max_new_tokens=500)[0].tolist())
print(sample_gen)

# step 0: train_loss 4.1546, val_loss 4.1591
# step 500: train_loss 2.6869, val_loss 2.7050
# step 1000: train_loss 2.5354, val_loss 2.5478
# step 1500: train_loss 2.4537, val_loss 2.4661
# step 2000: train_loss 2.3973, val_loss 2.4262
# step 2500: train_loss 2.3756, val_loss 2.3838
# step 3000: train_loss 2.3423, val_loss 2.3600
# step 3500: train_loss 2.3205, val_loss 2.3382
# step 4000: train_loss 2.2933, val_loss 2.3204
# step 4500: train_loss 2.2792, val_loss 2.3013
# step 5000: train_loss 2.2646, val_loss 2.2921
# step 5500: train_loss 2.2381, val_loss 2.2838
# step 6000: train_loss 2.2297, val_loss 2.2735
# step 6500: train_loss 2.2268, val_loss 2.2661
# step 7000: train_loss 2.2125, val_loss 2.2644
# step 7500: train_loss 2.2108, val_loss 2.2606
# final loss: 2.1532678604125977 after 8000 steps.