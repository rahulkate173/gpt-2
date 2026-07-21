"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""
from .base import Tokenizer, get_stats, merge

class BaseTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self,text,vocab_size,verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
    
    def encode(self,text):
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # map to int 
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats,key=lambda p : self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break # nothing else to merge
            idx = self.merges[pair]
            ids = merge(ids,pair,idx)
        return ids

    def decode(self,ids):
        text = b"".join(self.vocab[idx] for idx in ids)
        return text.decode("utf-8",errors="replace")
