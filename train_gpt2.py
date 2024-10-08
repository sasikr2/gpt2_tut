from dataclasses import dataclass
import math
import torch
import torch.nn  as nn
from torch.nn import functional as F
import tiktoken
import time 
import numpy as np

class CausalSelfAttentionBlock(nn.Module): # Multi Head Attention
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)   # doubt why we use this 

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # it is not bias, but openai used in gpt , actually it is kind of mask, TODO check whether it is trainable or not 
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # nice way to batch calculation
        qkv = self.c_attn(x) # (B, T, 3*n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)       #q,k,v (B, T, n_embd)

        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)           # (B, T, n_head, hs)-> (B, n_head, T, hs)   hs*n_head==n_embd
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)

        # att = q@k.transpose(-1, -2) * (1.0/math.sqrt(k.size(-1)))               # (B, n_head, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionBlock(config)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
            

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()     #parent class constructor call
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for i in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5  # intresting to avoid resuidal addition 1/sqrt(n_layers) here there are 2 times residual at each layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # print(sd_keys_hf)
        # print(sd_keys)
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def forward(self, idx, target=None):
        B, T = idx.size()
        token_emb = self.transformer.wte(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer['wpe'](pos)

        x = token_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        vocab_size = logits.size(-1)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
        return logits, loss


class DataLoaderLite:
    def __init__(self, B, T):
        with open("./input.txt", "r") as f:
            raw_text = f.read()
            f.close()

        self.B = B
        self.T = T
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(raw_text)
        self.tokens = torch.tensor(tokens)
        self.curr_position_ptr = 0
        print(f"tokens length {len(self.tokens)}")
        print(f"1 epoch ={len(self.tokens)//(B*T)}")
    
    def next_batch(self):
        B, T = self.B, self.T
        current_buff = self.tokens[self.curr_position_ptr: self.curr_position_ptr + B*T+1]
        x = current_buff[:-1].view(B, T)
        y = current_buff[1:].view(B, T)
        self.curr_position_ptr = self.curr_position_ptr + B*T+1

        if self.curr_position_ptr+B*T+1 > len(self.tokens):
            self.curr_position_ptr = 0
        return x, y



if __name__=="__main__":

    

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"using device: {device}")

    model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    model = torch.compile(model)

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    train_loader = DataLoaderLite(B=16, T=1024)
    torch.set_float32_matmul_precision("high")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        to= time.time()
        x, y =  train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            # import code; code.interact(local=locals())
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-to)*1000 # in msec
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1-to)
        print(f"step {i} loss {loss}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:2f}")

    # loss reaches from 11 to 6
