from dataclasses import dataclass
import os
import torch 
import torch.nn as nn 
import math 
from torch.nn import functional as F
from dotenv import load_dotenv
from rich import print 
import inspect
import time 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
load_dotenv() # to get hf_token 
# -----------------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed,3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed,config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0 # flag 

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B, T, C, = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed,dim=2)

        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        ## attention 
        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(x.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0,1e-6) # autoregressive mask : i.e tokens attend only before them not future 
        # y = att @ v
        # y = y.transpose(1,2).contiguous().view(B,T,C) # re-assemble 
        ## we will go with flash attention 
        y = F.scaled_dot_product_attention(q,k,v,attn_mask=self.bias[:,:,:T,:T],is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C) # re-assemble -> not i functional 
        y = self.c_proj(y)
        return y 


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed,4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embed,config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0 # flag 
    
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

@dataclass
class GPTConfig:
    block_size : int = 1024 # max sequence length
    vocab_size : int = 50304 # for even number  # number of tokens = 50000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
    n_layer : int = 12 # number of layes 
    n_head : int = 12 # number of heads 
    n_embed : int = 768 # embedding dimension 
    embd_pdrop: float = 0.1

class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config 

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embed),
            wpe = nn.Embedding(config.block_size,config.n_embed),
            drop = nn.Dropout(config.embd_pdrop), # and error : fix 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False) # final linear layer  

        # weight sharing scheme 
        self.transformer.wte.weight = self.lm_head.weight # redirecting the weight of lm_head to wte weight
        ## sharing this saves 30% of total parameters of 124 million model 

        # init param 
        self.apply(self._init_weights) # -> this iterate all the submodules and apply the function to each of them

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,"NANOGPT_SCALE_INIT"):
                std *= 2*self.config.n_layer**-0.5
            nn.init.normal_(module.weight,std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias) # not by default 
        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # [B,T,vocab_size]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss
    
    @classmethod #  it will return gpt object if we give the model type 
    def from_pretrained(cls,model_type):
        """loads pretrained GPT2 model weights from hugging face"""
        assert model_type in ("gpt2","gpt2-medium","gpt2-large","gpt2-xl")
        from transformers import GPT2LMHeadModel 
        print("loading weights from pretrained gpt:",model_type)

        config_args = {
            "gpt2": dict(n_layer=12,n_head=12,n_embed=768),
            "gpt2-medium": dict(n_layer=24,n_head=16,n_embed=1024),
            "gpt2-large": dict(n_layer=36,n_head=20,n_embed=1280),
            "gpt2-xl": dict(n_layer=48,n_head=25,n_embed=1600),
        }[model_type]
        config_args["vocab_size"] = 50304
        config_args["block_size"] = 1024

        # create a from-sratch initialized mingpt model 
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # discard this mask buffer 

        # init a huggingface/transformers model 
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all param are matched and correct 
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.maked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all candidate parameters 
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad} # filter out frozen params
        decay_params = [p for n,p in param_dict.items() if p.ndim == 2] # weight decay on 2D or higher params
        no_decay_params = [p for n,p in param_dict.items() if p.ndim < 2] # no weight decay on 1D or lower params - layernorm , scale , biases
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},  
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"setting up AdamW optimizer with {num_decay_params:,} decay and {num_no_decay_params:,} no_decay parameters")
        print(f"num non-decay params: {num_no_decay_params:,} | num decay params: {num_decay_params:,}")
        ## create adam optimzier and used fuesed version if available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters 
        use_fused = fused_available and "cuda" is device
        print(f"fused AdamW available: {fused_available}, using fused: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


## dataloader lite
import tiktoken 
class DataLoaderLite:
    def __init__(self,B,T,process_rank,num_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_process = num_process

        # load tokens from disk to memory 
        with open("input.txt","r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(tokens) // (B*T)} batches")

        # state
        self.current_positions = self.B * self.T * self.process_rank # each process will start from different position in the dataset

    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_positions:self.current_positions+B*T+1]
        x = (buf[:-1].view(B,T)) # inputs
        y = (buf[1:].view(B,T)) # output
        self.current_positions += B*T*self.num_process

        # if loading next batch is out of bound , reset
        if self.current_positions + B*T*self.num_process + 1 >= len(self.tokens):
            self.current_positions = self.B * self.T * self.process_rank # reset to start of dataset for this process
        return x,y

## Distributed Data Parallel (DDP) : https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA is required for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ.get("RANK", -1))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp_world_size = int(os.environ.get("WORLD_SIZE", 1)) 
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this is do looging ans checkpointing
else:
    # vanilla non ddp run 
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda" 
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        device = "mps" # macos metal backend
    print(f"using device: {device}")


# seed for reproducibility 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# enable tf32 precision 
torch.set_float32_matmul_precision("high") # will automatically take to tf32 precicion ( 1 sign , 8 exponent and 10 mantissa bits )

# gradient accumulation
total_batch_size = 524288 # 0.5M tokens 
B = 16
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0, "making sure total batchsize is divisible by (B*T*ddp_world_size)"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size) # if ddp is used then we need to divide by world size
if master_process:
    print(f"total desired batch size: {total_batch_size} = {B} * {T} * {grad_accum_steps}")
    print(f"==> calculated gradient accumulation steps: {grad_accum_steps}")
## here weight updated at only one grad accum step 

train_loader = DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_process=ddp_world_size)

model = GPT(GPTConfig(vocab_size=50304))
# model.eval()
model.to(device)
model = torch.compile(model) # to compile model 
if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # unwrap the model if ddp is used

## coise decay learning schedule with warmup 
max_lr = 6e-4
min_lr = 6e-4
warmup_steps = 10
max_steps = 50 

def get_lr(step):
    ## 1. linear warmup for first 1000 steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    ## 2. if step > lr_decay_steps, then return min_lr
    if step > max_steps:
        return min_lr
    ## 3. in between down to min learning rate 
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr + coeff*(max_lr - min_lr)

# lets create optimizer 
optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=max_lr,device=device) # this will return optimizer and scheduler

for step in range(max_steps):
    start_time = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device),y.to(device)
        with torch.autocast(device_type="gpu", dtype=torch.bfloat16):
            logits,loss = model(x,y)
        loss = loss / grad_accum_steps # scale the loss to account for gradient accumulation 
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # only sync gradients on last micro step
        loss.backward() # here in gradinent accumulation just loss gets added but the scaling is lost 
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # avg the loss across all processes
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0) # gradient clipping ->. clipping the gradient to between 1.0 / taking global norm 
    # optimizer 
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for all kernels to finish
    T = time.time() - start_time
    # token throughput per sec 
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / T if T > 0 else 0
    if master_process:
        print(f"step:{step+1} | loss:{loss_accum.item()} | norm:{norm:.4f} | lr:{lr:.4f} | time:{T:.2f} seconds | tokens/sec:{tokens_per_sec:.2f}")


if ddp:
    destroy_process_group()



import sys;sys.exit(0)


max_length = 30
return_number_sequences = 5

# seed for reproducibility 
torch.manual_seed(42)
torch.cuda.manual_seed(42)


while x.size(1) < max_length:
# forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0] # (8, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim =- 1)
        print(f"sum of prob",torch.sum(probs,dim=1,keepdim=True))
        # do top-k sampling of 50 (huggingface pipeline default)
        # & topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim =-1) # this take prob that not make to 1.0 at sum 
        # topk_probs = topk_probs / torch.sum(topk_probs, dim=-1, keepdim=True)
        # select a token from the top-k probabilities
        # topk_probs = topk_probs.to(torch.float32)
        print(topk_probs.shape)
        ix = torch.multinomial(topk_probs, 1) # (8, 1) # taking one from top 50 
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (8, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(return_number_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
