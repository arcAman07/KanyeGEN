import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 8
n_layer = 4
dropout = 0.0
vocab = 101
# ------------


class Head(nn.Module):
    def __init__(self, head_size):
        super(Head,self).__init__()
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self,x):
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super(MultiHeadAttention,self).__init__()
        self.head_size = head_size
        self.n_head = n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.out = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.out(out)
        out = self.dropout(out)
        return out

class FeedForwardLayer(nn.Module):
    def __init__(self, n_embd):
        super(FeedForwardLayer, self).__init__()
        self.n_embd = n_embd
        self.fc1 = nn.Linear(n_embd, 4*n_embd)
        self.fc2 = nn.Linear(4*n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.gelu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.attn = MultiHeadAttention(n_head, n_embd // n_head)
        self.ff = FeedForwardLayer(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, n_embd, n_layer):
        super(Transformer, self).__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.token_embedding = nn.Embedding(vocab, n_embd)
        self.position_embedding = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.ffwd = nn.Linear(n_embd, vocab)
    
    def forward(self, idx, targets=None):
        B,T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.ffwd(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=0)
        return logits,loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx