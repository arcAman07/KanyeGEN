import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import Transformer

with open('/Users/deepaksharma/Documents/Python/Kaggle/GenerateKanyeLyrics/Kanye West Lyrics.txt','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(text))
train_data = data[:n]
val_data = data[n:]

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


model = Transformer(n_embd,n_layer)

print("Total params: ", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for steps in range(20000):
    x,y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print("Step: ", steps, " Loss: ", loss.item())

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

torch.save(model.state_dict(), 'kanye_weights.pth')

lyrics = encode("Bitch I am back on my comma , sipping on my CocaCola, driving on a hangover ")
lyrics = torch.tensor(lyrics, dtype=torch.long)
lyrics = torch.stack([lyrics for _ in range(1)], dim=0)

print(decode(model.generate(lyrics, max_tokens=1000)[0].tolist()))
