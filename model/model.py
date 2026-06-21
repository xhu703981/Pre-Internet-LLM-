from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F
import os
from dotenv import load_dotenv
load_dotenv()
#----------------------------------------------------------------------------------------------------------------------------------#

@dataclass
class Configuration:
    block_size : int = 1024
    vocab_size : int = 50304
    n_embd : int = 768 
    n_layer : int = 12
    n_head : int = 12
    #n_inner : int = 3072
    n_swi : int = 2048


class RoPE(nn.Module):

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        theta = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size)) #frequence of each subspace (head_size/2)
        pos = torch.arange(0, config.block_size).float() 
        angles = torch.outer(pos, theta)  # position m, rotate (m*theta) degrees;  T * (head_size/2)
        # save as cos and sin
        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())
        #register_buffer: used to save a part of model state dict but not learnable parameters
        #not a part of the optimization

    def rotate(self, x):
         # x(q,k): B * n_head * T * head_size
        x1 = x[:,:,:, 0::2]  # even dimensions  B * n_head * T * (head_size/2)
        x2 = x[:,:,:, 1::2]  # odd dimensions   B * n_head * T * (head_size/2)
        cos = self.cos[:x.size(2), :] #T * (head_size/2)
        sin = self.sin[:x.size(2), :] #T * (head_size/2)
        x_rot = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1) # B * n_head * T * (head_size/2) * 2
        x_rot = x_rot.flatten(-2) #B * n_head * T * head_size
        return x_rot

    def forward(self, q, k):
        return self.rotate(q), self.rotate(k)
    
    
class SwiGLU(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.w= nn.Linear(config.n_embd, config.n_swi, bias=False) # *8/3 to control the nunmber of paramters
        self.v = nn.Linear(config.n_embd, config.n_swi, bias=False)
        self.proj = nn.Linear(config.n_swi, config.n_embd, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.w(x)) 
        content = self.v(x)
        x = self.proj(gate * content)
        return x


class ffw(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.swiglu = SwiGLU(config)
    
    def forward(self, x):
        x = self.swiglu(x)
        return x
    

class attention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.atten = nn.Linear (config.n_embd, 3*config.n_embd,bias=False)  #qkv
        self.lproj = nn.Linear (config.n_embd, config.n_embd,bias=False)
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_size =  self.n_embd // self.n_head
        self.rope = RoPE(config)
    
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
        q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(q,k,v, is_causal=True)  # flash attention
        out = out.transpose(1, 2)
        out = out.reshape(B, T ,C)
        out = self.lproj (out)
        return out
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.attention = attention(config)
        self.ln2 = nn.RMSNorm(config.n_embd)
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
            'h' : nn.ModuleList(Block(config) for _ in range(config.n_layer)), #hidden layers
            'ln' : nn.RMSNorm(config.n_embd), # layer normalization 
            }
        )
        self.lp = nn.Linear (config.n_embd, config.vocab_size, bias = False)  # softmax (x+c) == softmax（x） 
        self.lp.weight = self.transformer.wte.weight
    

    def forward(self, idx, target = None):
        B, T = idx.size()
        tok_embd = self.transformer.wte(idx)

        x = tok_embd 
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
    data_dir : str = os.environ.get('DATA_DIR', './data/books_data_final.bin')
    out_dir : str = os.environ.get('OUT_DIR', './out')

# ----------------------------------------------------------------------------

@dataclass 
class Train_config:
    batch_size : int = 4
    grad_accum_step : int = 8
    max_iters : int = 100000
    lr : float = 3e-4

    warmup_iters: int = 2000
    decay_iters: int = 100000
    min_lr: float = 3e-5

    eval_interval: int = 2000
    ckpt_interval : int = 1000


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


def _raw_state_dict(model):
    """
    Strip the '_orig_mod.' prefix that torch.compile() adds to state_dict keys.
    This keeps checkpoints loadable into either a compiled or uncompiled model.
    """
    sd = model.state_dict()
    cleaned = {}
    for k, v in sd.items():
        cleaned[k.replace('_orig_mod.', '')] = v
    return cleaned


def save_ckpt(path, model, optimizer, iter_num, best_val_loss):
    ckpt = {
         'model' : _raw_state_dict(model),
         'optimizer' : optimizer.state_dict(),
         'iter_num' : iter_num,  # so the loop resumes at the right iter
         'best_val_loss' : best_val_loss,
    }
    torch.save(ckpt, path)
    print(f"saved checkpoint to {path} (iter {iter_num})")


def load_checkpoint(path, device):
    # weights_only=False because this checkpoint contains the dataclass config object
    # and optimizer state, not just tensors. Only load checkpoints you trust/created yourself.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


def train():
    import os 
    import numpy as np

    model_config = Configuration()
    train_config = Train_config()
    data_config = Data_config()

    os.makedirs(data_config.out_dir, exist_ok=True)

    last_ckpt_path = os.path.join(data_config.out_dir, 'last_model.pt')
    best_ckpt_path = os.path.join(data_config.out_dir, 'best_model1.pt')

    data = np.memmap(data_config.data_dir, dtype='uint16', mode='r')
    n = int(len(data) * 0.8)
    train_data = data[:n]
    val_data   = data[n:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    best_val_loss = float('inf')
    starting_iter = 0

    resume_path = None
    if os.path.exists(last_ckpt_path):
        resume_path = last_ckpt_path
    if resume_path is not None: 
        print(f"resuming from : {resume_path}")
        checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        starting_iter = checkpoint['iter_num'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"resumed at iter {starting_iter}, best_val_loss so far: {best_val_loss:.4f}")
    else:
        print(f'starting from zero')


    #model = torch.compile(model=model)
    device = next(model.parameters()).device



    print('start training')
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")


    for it in range(starting_iter, train_config.max_iters):
        model.train()  #train mode


        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        for micro_step in range(train_config.grad_accum_step):
            x, y = get_batch(train_data, model_config.block_size, train_config.batch_size, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):  #mixed precision
                logits, loss = model(x, target=y)
            loss = loss / train_config.grad_accum_step
            accum_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #gradient clipping
        optimizer.step()
        
        print(f"Iter {it} | loss: {accum_loss:.4f} | lr: {lr:.2e}")

        if it % train_config.ckpt_interval == 0 and it != 0:
            save_ckpt(last_ckpt_path, model, optimizer, it, best_val_loss)

        if it % train_config.eval_interval == 0 and it != 0:
            model.eval()

            val_loss = 0.0
            eval_iters = 100
            for i in range(eval_iters):
                val_x, val_y = get_batch(val_data, model_config.block_size, train_config.batch_size, device)
                with torch.no_grad():
                    _, val_loss_batch = model(val_x, target=val_y)
                val_loss += val_loss_batch.item()  
            val_loss /= eval_iters
            print(f"=====================")
            print(f"Iter {it} | Val loss: {val_loss:.4f}")
            print(f"=====================")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt(best_ckpt_path, model, optimizer, it, best_val_loss)

    save_ckpt(last_ckpt_path, model, optimizer, train_config.max_iters - 1, best_val_loss)


if __name__ == '__main__':
    train()
