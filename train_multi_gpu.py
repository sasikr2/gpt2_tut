"""
This is modification of llm.c training script and nanoGPT training script
https://github.com/karpathy/llm.c/blob/master/train_gpt2.py
https://github.com/karpathy/nanoGPT/blob/master/train.py
"""



import os
import sys
import math
import glob
import random
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn  as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import tiktoken
import time 
import numpy as np
import wandb

import glob

def setup_logging_for_debugging(log_file_base="debug_run"):
    """
    Sets up logging for all ranks, each to its own file.
    The master rank also prints to the console.
    """
    import logging
    process_rank = int(os.environ.get("RANK", 0))
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Include rank in the log format
    formatter = logging.Formatter(
        f'%(asctime)s - RANK {process_rank} - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Each rank gets its own log file
    fh = logging.FileHandler(f"{log_file_base}_rank_{process_rank}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # The master rank also logs to the console
    if process_rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
logger = setup_logging_for_debugging(log_file_base="logs/training_multi_gpu_run.log")

class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


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
        self.gelu = NewGELU()
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
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                std = (2 * self.config.n_layer) ** -0.5  # interesting to avoid residual addition 1/sqrt(n_layers) here there are 2 times residual at each layer
                torch.nn.init.normal_(param, mean=0.0, std=0.02*std, generator=self.init_rng)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02   # this is default init for linear layer,actually it is approx 1/sqrt(num_features) like 1/sqrt(768)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

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

    def forward(self, idx, target=None, return_logits=True):
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
            
        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None
            
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
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


#[TODO] in future add .bin file reading support
def _load_data_shard(filename):
    tokens = np.memmap(filename, dtype=np.uint16, mode='r')
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        # print(f"[Dataloader] filename pattern: {filename_pattern}")
        self.files = [x for x in sorted(glob.glob(filename_pattern)) if x.endswith(".npy") != -1]
        if not self.files:
            raise ValueError(f"No files found for pattern {filename_pattern}")
        # print(f"[Dataloader] Found {len(self.files)} data files.")
        
        self.current_shard_idx = -1 # Start at -1 to ensure the first shard is loaded by reset
        self.tokens = None
        self.current_position = 0

        self.reset()
        
    def reset(self):
        """Resets the dataloader, shuffles shards, and loads the first one."""
        logger.debug("Resetting dataloader and shuffling shards.")
        # Shuffle shards for the new epoch
        random.shuffle(self.files)
        # Reset to the first shard and load it
        self.current_shard_idx = 0
        self.tokens = _load_data_shard(self.files[self.current_shard_idx])
        # Set the starting position for the current process
        self.current_position = self.B * self.T * self.process_rank
        
        
    def next_batch(self):
        B, T = self.B, self.T
        # This loop ensures we find a shard with enough data for the next batch
        while self.current_position + B * T + 1 > len(self.tokens):
            self.advance()
        
        token_buf = self.tokens[self.current_position: self.current_position + B*T + 1]
        buf = torch.tensor(token_buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        self.current_position += B * T * self.num_processes
        return x, y
    
    def advance(self): # advance to next data shard
        """Advances to the next data shard or resets if at the end of an epoch."""
        self.current_shard_idx += 1
        if self.current_shard_idx >= len(self.files):
            self.reset()
            return
        self.current_position = self.process_rank * self.B * self.T
        logger.info(f"loading shard {self.current_shard_idx} for process {self.process_rank}")
        self.tokens = _load_data_shard(self.files[self.current_shard_idx])

# def print0(*args, **kwargs):
#     # modified print that only prints from the master process
#     # if this is not a distributed run, it's just a print
#     if int(os.environ.get("RANK", 0)) == 0:
#         print(*args, **kwargs)       

def args():
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Train a GPT model with multiple GPUs.")
    parser.add_argument('--single', action='store_true', help='Single GPU training')
    # parser.add_argument('--device', type=str, default=None, help='Device to use for training (e.g., "cuda:0").')
    return parser.parse_args()

if __name__=="__main__":
        
    args = args()
    if args.single:
        from training_config import single_gpu_config
        # use single GPU config
        logger.info("Using single GPU configuration.")
        training_config = single_gpu_config
    else:
        # use multi GPU config
        logger.info("Using multi GPU configuration.")
        from training_config import config as training_config
    
    ########## Device Configuration #########    
    zero_stage = 0
    device = None
    training_type = "single_cpu"
    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = 0 #[TODO] what does this mean?
        training_type = "multi_gpu"
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # select the device
        if device:
            # provided explicitly by the user
            device = device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
                training_type = "single_gpu"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                training_type = "single_mps_gpu"
                device = "mps"
    logger.info(f"using device: {device}")
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # wandb logging setupå
    if training_config.get("wandb_log", False) and master_process:
        import wandb
        wandb.init(entity="rc_speech",
                project="llm-test",
                name=training_config.get('run_name', 'gpt2_multi_gpu_4'),
                config=training_config)
    
    ########## Model Loading ##########
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    current_iter = 0
    best_5_val_loss = [float('inf')]*5
    if training_config["training_type"] == "checkpoint":
        cpkt_path = training_config["checkpoint_path"]
        logger.info(f"Loaded checkpoint from {cpkt_path}")
        model_checkpoint = torch.load(cpkt_path, weights_only=False, map_location=device)
        checkpoint_model_args = model_checkpoint["model_args"]
        logger.info("[Checkpoint] model args: %s" % checkpoint_model_args)
        # model_args = {}
        # for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        #     model_args[k] = checkpoint_model_args[k]  
        # gptconf = GPTConfig(checkpoint_model_args)
        model = GPT(checkpoint_model_args) # use the model args from the checkpoint
        # state_dict = model_checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(model_checkpoint["model"].items()):
            if k.startswith(unwanted_prefix):
                model_checkpoint["model"][k[len(unwanted_prefix):]] = model_checkpoint["model"].pop(k)
        model.load_state_dict(model_checkpoint["model"])
        current_iter = model_checkpoint["iter_num"] + 1
        best_5_val_loss = [ model_checkpoint["best_val_loss"] ] + best_5_val_loss[1:] # keep the best val loss from the checkpoint
        logger.info(f"Current iter: {current_iter} | Best val loss: {best_5_val_loss}")
        checkpoint_training_config = model_checkpoint["training_config"]
        # override the training config with the checkpoint training config
        for k, v in checkpoint_training_config.items():
            if k in ['training_type', 'checkpoint_path', 'checkpoint_dir', 'run_name', 'compile', 'wandb_log']:
                continue
            else:
                training_config[k] = v
    elif training_config["training_type"] == "scratch":
        logger.info("Initializing model from scratch.")
        logger.info("setting vocab size to 50304 for hardware optimization")
        gptconf = GPTConfig(vocab_size=50304)    
        model = GPT(gptconf)        # increase vocab size from 50257 to 50304 just only for hardware optimization
    model.train()
    model.to(device)
    
    # [TODO] current it throws error need to fix it 
    if training_config.get('compile', False):
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        logger.info("compiling the model...")
        model = torch.compile(model)
    
    ##### Training setup #####
    
    # training config
    training_data_base_dir = training_config['training_data_base_dir']
    validation_iid_base_dir = training_config['validation_iid_base_dir']
    validation_ood_base_dir = training_config['validation_ood_base_dir']
    B, T = training_config["B"], training_config["block_size"]    
    learning_rate = training_config['learning_rate']
    lr_min = training_config.get('min_learning_rate', learning_rate * 0.1) # default to 10% of max learning rate
    lr_decay_steps = training_config.get('lr_decay_steps', training_config['max_iterations']) # default to max_iterations if not set
    weight_decay = training_config['weight_decay']
    grad_clip = training_config['grad_clip']
    warmup_steps = training_config['warmup_iters']
    val_loss_every = training_config["val_loss_every"]
    log_interval = training_config["log_interval"]
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
    optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)
    if training_config["training_type"] == 'checkpoint':
        optimizer.load_state_dict(model_checkpoint['optimizer'])
    model_checkpoint = None # free up memory
    
    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[training_config.get('dtype', 'bfloat16')]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    
    #train data loading
    enc = tiktoken.get_encoding("gpt2")
    train_loader = DistributedDataLoader(
        filename_pattern=os.path.join(training_data_base_dir, "edufineweb_train_*.npy"),
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size
    )
    val_loader = DistributedDataLoader(
        filename_pattern=os.path.join(validation_iid_base_dir, "edufineweb_val_*.npy"),
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size
    )  
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert training_config['global_batch_size'] % tokens_per_fwdbwd == 0
    gradient_accumulation_steps = training_config['global_batch_size'] // tokens_per_fwdbwd
    logger.info(f"Total desired global batch size: {training_config['global_batch_size']} | per forward-backward batch size: {tokens_per_fwdbwd} | gradient accumulation steps: {gradient_accumulation_steps}")
    training_config['ddp_world_size'] = ddp_world_size
    training_config['gradient_accumulation_steps'] = gradient_accumulation_steps
        
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
    
    timings = []    
    running_mfu = -1.0
    for step in range(current_iter, training_config["max_iterations"]+1):
        
        last_step = (step ==training_config["max_iterations"])
        # evaluate the model every eval_interval steps
        if step > 0 and val_loss_every > 0 and (step% val_loss_every == 0 or last_step) and val_loader is not None and master_process:
            model.eval()
            val_loader.reset()  # reset the validation loader to the start
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(training_config.get('val_max_steps', 100)):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= training_config.get('val_max_steps', 100)
            logger.info(f"Validation loss at step {step}: {val_loss:.6f}")
            if training_config.get('wandb_log', False):
                wandb.log({"val/loss": val_loss, "step": step})
            
            if step > 0 and  val_loss < min(best_5_val_loss):
                # if the validation loss is better than the worst in the top 5, update the list
                best_5_val_loss.append(val_loss)
                best_5_val_loss.sort()
                best_5_val_loss = best_5_val_loss[:5]
                # save the model checkpoint
                checkpoint = {
                    "model_args": raw_model.config,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": step,
                    "best_val_loss": val_loss,
                    "training_config": training_config
                }
                cpkt_path = os.path.join(training_config["checkpoint_dir"], f"ckpt_{training_type}_step_{step}_valloss_{val_loss:.4f}.pt")
                torch.save(checkpoint, cpkt_path)
            
        # once in a while perform model inference on the master process
        if (training_config['sample_every'] > 0 \
            and (step % training_config['sample_every'] == 0 or last_step)) \
            and master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            start_ids = [enc.eot_token]
            xg = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            max_new_tokens = 48
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
            logger.info(f'------Output at step {step}------')
            logger.info(enc.decode(yg[0].tolist()))
            logger.info('---------------')
            
        if last_step:
            logger.info("Reached the last step, exiting training loop.")
            break
            
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(gradient_accumulation_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / gradient_accumulation_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = learning_rate_cosine_decay_schedule(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)
        timings.append(dt*1000) # in milliseconds
        tokens_per_second = gradient_accumulation_steps * ddp_world_size * B * T / (t1-t0)
        logger.info(f"""step {step+1:4d}/{training_config["max_iterations"]} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)""")
        if step % log_interval == 0 and master_process:
            mfu = raw_model.estimate_mfu(fwdbwd_per_iter=B * gradient_accumulation_steps, dt=dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            logger.info(f"step {step+1:4d}/{training_config['max_iterations']} | running mfu: {running_mfu*100:.2f}%")
            if training_config.get('wandb_log', False):
                wandb.log({
                    'train/loss': lossf,
                    "lr": lr,
                    "grad_norm": norm,
                    "step": step,
                    "tokens_per_second": tokens_per_second,
                    "average_time_per_step": np.mean(timings[-log_interval:]) if len(timings) > log_interval else dt,
                    "running_mfu": running_mfu*100
                })
            
    logger.info(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    logger.info(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    if ddp:
        destroy_process_group()  