from dataclasses import dataclass
import math
import os
import sys
import torch
import torch.nn  as nn
from torch.nn import functional as F
import tiktoken
import time 
import numpy as np
import inspect

import wandb

import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import wandb


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

        # weight sharing scheme , wirght tying
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
        logger.info("loading weights from pretrained gpt: %s" % model_type)
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
        # logger.info(sd_keys_hf)
        # logger.info(sd_keys)
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params_dict = {pn:p for pn, p in self.named_parameters()}
        params_dict = {pn:p for pn, p in params_dict.items() if p.requires_grad}

        # decay params which have dim 2 and more like embedding and matrix multiplications
        decay_params = [p for p in params_dict.values() if p.dim() >= 2]
        no_decay_params = [p for p in params_dict.values() if p.dim() < 2]

        # params for decay or no decay
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        logger.info(f"Number of decay param tensors: {len(decay_params)} | {num_decay_params} parameters")
        logger.info(f"Number of no decay param tensors: {len(no_decay_params)} | {num_no_decay_params} parameters")

        params_config = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        fused_adam = "fused" in inspect.signature(torch.optim.AdamW).parameters
        fused_adam =  True if device_type == "cuda" else False
        optimizer = torch.optim.AdamW(params_config, lr=learning_rate, betas=betas, eps=1e-8, fused=fused_adam)
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T, split="train", device="cpu"):

        self.device = device
        file_list = os.listdir("./edu_fineweb10B")
        file_list = ["./edu_fineweb10B/"+f for f in file_list]
        shard_list = [f for f in file_list if f.find(split) != -1]
        logger.info(f"{split} Shard length: {len(shard_list)}")

        self.B = B
        self.T = T
        self.shard_list = shard_list
        self.curr_shard_ptr = 0
        self.tokens = self.load_tokens(self.shard_list[self.curr_shard_ptr])
        self.curr_position_ptr = 0

    def reset_shard(self):
        # logger.info(f"Resetting shard: {self.curr_shard_ptr}")
        if self.curr_shard_ptr >= len(self.shard_list):
            self.curr_shard_ptr = 0 
        else:
            self.curr_shard_ptr += 1

        self.tokens = self.load_tokens(self.shard_list[self.curr_shard_ptr])
        self.curr_position_ptr = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        current_buff = self.tokens[self.curr_position_ptr: self.curr_position_ptr + B*T+1]
        x = current_buff[:-1].view(B, T)
        y = current_buff[1:].view(B, T)
        self.curr_position_ptr = self.curr_position_ptr + B*T+1

        if self.curr_position_ptr+B*T+1 > len(self.tokens):
            self.reset_shard()
        return x, y

    def load_tokens(self, shard_path):
        tokens_np = np.load(shard_path)
        tokens_np = tokens_np.astype(np.long)
        tokens_tensor = torch.from_numpy(tokens_np)
        tokens_tensor = tokens_tensor.to(self.device)
        # logger.info(tokens_tensor.dtype)
        return tokens_tensor
    

if __name__=="__main__":

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"Current Device: {device}")

    model = GPT(GPTConfig(vocab_size=50304))        # increase vocab size from 50257 to 50304 just only for hardware optimization
    model.to(device)
    model = torch.compile(model)

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # TODO move it to config
    training_config = {
        "out_dir": "/clearml_agent_cache/storage_manager/dynamic_lm_exp_shashik/self_exp/gpt2_tut",
        "batch_size": 32,
        "total_batch_size": 2**12,   # total batch size for all GPUs, gradient accumulation
        "eval_iters": 200,
        "eval_interval": 500,
        "log_interval": 5,
        "wandb_log": True,

        "learning_rate": 6e-4, # max learning rate
        "max_iters": 60000, # total number of training iterations
        "weight_decay": 1e-1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0, # clip gradients at this value, or disable if == 0.0

        "warmup_steps": 2000
    }
    out_dir = training_config["out_dir"]
    batch_size = training_config["batch_size"]
    eval_iters = training_config["eval_iters"]
    eval_interval = training_config["eval_interval"]
    log_interval = training_config["log_interval"]
    wandb_log = training_config["wandb_log"]

    learning_rate = training_config["learning_rate"] # max learning rate
    max_iters = training_config["max_iters"] # total number of training iterations
    weight_decay = training_config["weight_decay"]
    beta1 = training_config["beta1"]
    beta2 = training_config["beta2"]
    grad_clip = training_config["grad_clip"] # clip gradients at this value, or disable if == 0.0

    # learning rate scheduler config
    warmup_steps = training_config["warmup_steps"]
    lr_min = learning_rate * 0.1 # as in gpt paper it's 10% of max_lr should be ~= learning_rate/10 per Chinchilla
    lr_decay_steps = max_iters # should be ~= max_iters per Chinchilla

    ###### learning rate schedule ######
    # warmup_steps + cosine decay + lr_min
    def learning_rate_cosine_decay_schedule(iter):
        if iter < warmup_steps:
            return (iter+1)*learning_rate/(1+warmup_steps) # +1 to avoid division by zero
        elif iter > lr_decay_steps:
            return lr_min
        else:
            decay_ratio = (iter-warmup_steps)/(lr_decay_steps-warmup_steps)
            curr_lr = lr_min + 0.5*(learning_rate-lr_min)*(1+math.cos(math.pi*decay_ratio))
            return curr_lr

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                if split == "train":
                    X, Y = train_loader.next_batch()
                else:
                    X, Y = val_loader.next_batch()
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    if wandb_log:
        run = wandb.init(
            entity="rc_speech",
            project="llm-test",
            name="gpt2-tut-train-1-2025-05-12",
            config=training_config
        )

    train_loader = DataLoaderLite(B=batch_size, T=1024, split="train", device=device)
    val_loader = DataLoaderLite(B=batch_size, T=1024, split="val", device=device)
    torch.set_float32_matmul_precision("high")
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    # logger.info({pn:p.shape for pn, p in model.named_parameters()})
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=(beta1,beta2), device_type=device)

    total_batch_size = training_config["total_batch_size"]
    grad_accum_steps = total_batch_size // batch_size
    logger.info(f"Total batch size: {total_batch_size} | Grad accum steps: {grad_accum_steps}")

    best_val_loss = float('inf')
    for step in range(max_iters):
        to= time.time()

        # optimizer and lr setup
        lr = learning_rate_cosine_decay_schedule(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # gradient accumulation
        loss_accumalation = 0
        for mini_batch in range(grad_accum_steps):
            x, y =  train_loader.next_batch()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss/grad_accum_steps
            loss.backward()
            loss_accumalation += loss.detach()

        # Returns the original gradient norm before clipping (stored in the variable norm)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        torch.cuda.synchronize()

        # clear the gradient to free memory
        optimizer.zero_grad()
        
        t1 = time.time()
        dt = (t1-to)*1000 # in msec
        tokens_per_sec = (train_loader.B * train_loader.T*grad_accum_steps) / (t1-to)

        if step % log_interval == 0:
            logger.info(f"step: {step:4d} | lr: {lr:.6f} | loss: {loss_accumalation:.4f} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:2f}")
        if step % eval_interval == 0:
            losses = estimate_loss()
            logger.info(f"step: {step:4d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
            if wandb_log:
                    wandb.log({
                        "step": step,
                        "norm": norm,
                        "lr": lr,
                        # "dt": dt,
                        # "tok/sec": tokens_per_sec,
                        "train/loss": losses['train'],
                        "val/loss": losses['val']
                    })
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if step > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model.config,
                        'iter_num': step,
                        'best_val_loss': best_val_loss,
                        'config': training_config,
                        }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            
    # loss reaches from 11 to 6
