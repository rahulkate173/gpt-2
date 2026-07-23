from dataclasses import dataclass
import torch 
import torch.nn as nn 
import math 
from torch.nn import functional as F
from dotenv import load_dotenv
from rich import print 
load_dotenv() # to get hf_token 
# -----------------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed,3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed,config.n_embed)

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
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(x.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf')) # autoregressive mask : i.e tokens attend only before them not future 
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C) # re-assemble 
        y = self.c_proj(y)
        return y 


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed,4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embed,config.n_embed)
    
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
        self.ln_2 = nn.LayerNorm(self.n_embed)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

@dataclass
class GPTConfig:
    block_size : int = 1024 # max sequence length
    vocab_size : int = 50257 # number of tokens = 50000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
    n_layer : int = 12 # number of layes 
    n_head : int = 12 # number of heads 
    n_embed : int = 768 # embedding dimension 


class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config 

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embed),
            wpe = nn.Embedding(config.block_size,config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False) # final linear layer  

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
        return logits 
    
    @classmethod #  it will return gpt object if we give the model type 
    def from_pretrained(cls,model_type):
        """loads pretrained GPT2 model weights from hugging face"""
        assert model_type in ("gpt2","gpt2-medium","gpt2-large","gpt2-xl")
        from transformers import GPT2LMHeadModel 
        print("loading weights from pretrained gpt:",model_type)

        config_args = {
            "gpt2": dict(n_layer=12,n_head=12,n_emded=768),
            "gpt2-medium": dict(n_layer=24,n_head=16,n_emded=1024),
            "gpt2-xl": dict(n_layer=36,n_head=20,n_emded=1280),
            "gpt2": dict(n_layer=48,n_head=25,n_emded=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
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

model = GPT.from_pretrained("gpt2")
print("loading weights we didn`t crash!")
# print(model)
## gpt-2 implementation code 