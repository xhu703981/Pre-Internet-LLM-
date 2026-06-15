from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F

#----------------------------------------------------------------------------------------------------------------------------------#
@dataclass
class Configuration:
    block_size : int = 1024
    vocab_size : int = 50304
    n_embd : int = 768 
    n_layer : int = 12
    n_head : int = 12
    n_inner : int = 3072


class ffw(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.proj1 = nn.Linear(config.n_embd, config.n_inner)
        self.gelu = nn.GELU(approximate='tanh')
        self.proj2 = nn.Linear(config.n_inner, config.n_embd)
    
    def forward(self, x):
        x = self.proj1(x)
        x = self.gelu(x)
        x = self.proj2(x)
        return x
    

class attention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.atten = nn.Linear (config.n_embd, 3*config.n_embd)  #qkv
        self.lproj = nn.Linear (config.n_embd, config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_size =  self.n_embd // self.n_head
    
    def forward (self, x):
        B, T, C= x.size()
        qkv = self.atten(x)  # B* T * (3*C)
        q, k, v = qkv.split(self.n_embd, dim =-1) # 3 :  B * T * C
        q = q.reshape(B, T, self.n_head, self.head_size)  # B * T * n-head * head_size            
        q = q.transpose(1, 2)  # B * n_head * T * head_size
        k = k.reshape(B, T, self.n_head, self.head_size)  # B * T * n-head * head_size            
        k = k.transpose(1, 2)  # B * n_head * T * head_size
        v = v.reshape(B, T, self.n_head, self.head_size)  # B * T * n-head * head_size            
        v = v.transpose(1, 2)  # B * n_head * T * head_size
        out = F.scaled_dot_product_attention(q,k,v, is_causal=True)  # flash attention
        out = out.transpose(1, 2)
        out = out.reshape(B, T ,C)
        out = self.lproj (out)
        return out
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffw = ffw(config)
    
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x  # residual connection
    

class Model(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.transformer = nn.ModuleDict({
            'wte' : nn.Embedding(config.vocab_size, config.n_embd), #token embd 
            'wpe' : nn.Embedding(config.block_size, config.n_embd), #position embd
            'h' : nn.ModuleList(Block(config) for _ in range(config.n_layer)), #hidden layers
            'ln' : nn.LayerNorm(config.n_embd), # layer normalization 
            }
        )
        self.lp = nn.Linear (config.n_embd, config.vocab_size, bias = False)  # softmax (x+c) == softmax（x） 
        self.lp.weight = self.transformer.wte.weight
    

    def forward(self, idx, target = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device = idx.device)
        tok_embd = self.transformer.wte(idx)
        pos_embd = self.transformer.wpe(pos)

        x = tok_embd + pos_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln(x)

        if target is not None:
            logits = self.lp(x)  # B * T * vocab_size
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index = -1)
        else: #inference
            logits = self.lp(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss


#----------------------------------------------------------------------------------------------------------------------------------#
@dataclass
class Data_config:
    data_dir : str = r'C:\Users\xhu70\Documents\LLM_from_scratch\bpe\books_data_final.bin'
    out_dir :str = r'C:\Users\xhu70\Documents\LLM_from_scratch\out'

# ----------------------------------------------------------------------------
@dataclass 
class Train_config:
    batch_size : int = 4
    max_iters : int = 100000
    lr : float = 3e-4

    warmup_iters: int = 2000
    decay_iters: int = 100000
    min_lr: float = 3e-5

    eval_interval: int = 2000


def get_lr(iter, config = Train_config()):
    import math
    if iter < config.warmup_iters:
        return config.lr * (iter / config.warmup_iters) # warmup: gradually increase
    if iter > config.decay_iters:
        return config.min_lr
    decay_ratio = (iter - config.warmup_iters) / (config.decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  #consine decay
    return config.min_lr + coeff * (config.lr - config.min_lr)


def get_batch(data, block_size, batch_size, device):
    import numpy as np
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def train():
    import os 
    import numpy as np

    model_config = Configuration()
    train_config = Train_config()
    data_config = Data_config()

    os.makedirs(data_config.out_dir, exist_ok=True)

    data = np.memmap(data_config.data_dir, dtype='uint16', mode='r')
    n = int(len(data) * 0.8)
    train_data = data[:n]
    val_data   = data[n:]

    model = Model(model_config).to('cuda' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    best_val_loss = float('inf')

    print('start training')
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")


    for it in range(train_config.max_iters):
        model.train()  #train mode

        x, y = get_batch(train_data, model_config.block_size, train_config.batch_size, device)

        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):  #mixed precision
            logits, loss = model(x, target=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #gradient clipping
        optimizer.step()
        
        print(f"Iter {it} | loss: {loss.item():.4f} | lr: {lr:.2e}")

        if it % train_config.eval_interval == 0 and it != 0:
            model.eval()

            val_loss = 0.0
            eval_iters = 50
            for i in range(eval_iters):
                val_x, val_y = get_batch(val_data, model_config.block_size, train_config.batch_size, device)
                with torch.no_grad():
                    _, val_loss_batch = model(val_x, target=val_y)
                val_loss += val_loss_batch.item()  # gradient accumulation
            val_loss /= eval_iters
            print(f"=====================")
            print(f"Iter {it} | Val loss: {val_loss:.4f}")
            print(f"=====================")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(data_config.out_dir, 'best_model.pt'))


if __name__ == '__main__':
    train()
