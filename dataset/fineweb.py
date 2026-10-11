import os
import multiprocessing as mp
import torch
import numpy as np 
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# =========================================
local_dir = "edu_fineweb14B"
remote_name = "sample-10BT"
shard_size = int(1e8)

# create cache 
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__),local_dir)
os.makedirs(DATA_CACHE_DIR,exist_ok=True)

# download dataset 
fw = load_dataset("HuggingFaceFW/fineweb-edu",name=remote_name,split="train")

# init tokenizer 
enc = tiktoken.get_encoding("gpt2")
eot = enc.special_tokens["<|endoftext|>"]
def tokenize(doc):
    # tokenizing single document
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
