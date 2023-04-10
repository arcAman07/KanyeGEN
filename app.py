import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Transformer

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

with open('/Users/deepaksharma/Documents/Python/Kaggle/GenerateKanyeLyrics/Kanye West Lyrics.txt','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])


model = Transformer(n_embd,n_layer)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

def generate_kanye_lyrics(text, max_tokens=500):
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
    return initial_text + decode(model.generate(lyrics, max_tokens=int(max_tokens))[0].tolist())[padding:]

demo = gr.Interface(fn=generate_kanye_lyrics, inputs=[gr.Textbox(lines=2, placeholder="Enter Starting lyrics ..."),gr.Number()], outputs="text")

demo.launch()