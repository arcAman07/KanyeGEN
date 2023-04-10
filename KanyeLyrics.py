#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# In[4]:


with open('/Users/deepaksharma/Documents/Python/Kaggle/GenerateKanyeLyrics/Kanye West Lyrics.txt','r',encoding='utf-8') as f:
    text = f.read()


# In[6]:


len(text)


# In[9]:


chars = sorted(list(set(text)))


# In[10]:


len(chars)


# In[31]:


stoi = {ch:i for i,ch in enumerate(chars)}


# In[32]:


itos = {i:ch for i,ch in enumerate(chars)}


# In[33]:


encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])


# In[36]:


a = encode("Hi guys I am Aman")
a


# In[37]:


decode(a)


# In[44]:


data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.type)


# In[55]:


n = int(0.9*len(text))
train_data = data[:n]
val_data = data[n:]
block_size = 8
batch_size = 32
print("A single block is: ",train_data[:block_size])


# In[56]:


x = train_data[:block_size]
y = train_data[1:block_size+1]


# In[57]:


for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print("Context: ", context, " Target: ", target)

ix = torch.randint(len(train_data)-block_size,(batch_size,))


# In[58]:


x = torch.stack([train_data[i:i+8] for i in ix])
y = torch.stack([train_data[i+1:i+9] for i in ix])


# In[158]:


def get_batch(split):
    if split == 'train':
        data = train_data
    elif split == 'val':
        data = val_data
    else:
        raise ValueError("Invalid split")
  
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# In[159]:


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
vocab = len(chars)
# ------------


# In[160]:


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
        


# In[161]:


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


# In[162]:


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


# In[163]:


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
        


# In[164]:


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


# In[165]:


model = Transformer(n_embd,n_layer)

print("Total params: ", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for steps in range(100000):
    x,y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print("Step: ", steps, " Loss: ", loss.item())


# In[167]:


# generate from the model
context = torch.zeros((1, 64), dtype=torch.long, device=device)
print(decode(model.generate(context, max_tokens=1000)[0].tolist()))


# In[168]:


# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# In[169]:


# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


# In[170]:


torch.save(model.state_dict(), 'model_weights.pth')


# In[171]:


stoi


# In[176]:


lyrics = encode("Bitch I am back on my comma , sipping on my CocaCola, driving on a hangover ")
lyrics = torch.tensor(lyrics, dtype=torch.long)
lyrics = torch.stack([lyrics for _ in range(1)], dim=0)
lyrics.shape


# In[177]:


print(decode(model.generate(lyrics, max_tokens=1000)[0].tolist()))


# In[220]:


def generate_kanye_lyrics(text):
    if len(text)<64:
        initial_text = ""
        padding = 64-len(text)
        initial_list = []
        for i in range(0, padding):
            initial_list.append(0)
        context = initial_list + encode(text)
    else:
        padding = 0
        initial_text = text[0:len(text)-block_size]
        context = text[-block_size:]
        context = encode(context)
    context = torch.tensor(context, dtype=torch.long)
    lyrics = torch.stack([context for _ in range(1)], dim=0)
    return initial_text + decode(model.generate(lyrics, max_tokens=1000)[0].tolist())[padding:]


# In[221]:


final_lyric_1 = generate_kanye_lyrics("Bitch don't kill my vibe!, I can feel your energy from two places away. I got my music and my ")


# In[222]:


final_lyric


# In[223]:


final_lyric_2 = generate_kanye_lyrics("She says she wants some Marvin Gayes ")


# In[224]:


final_lyric_2


# In[225]:


get_ipython().system('pip install gradio')

