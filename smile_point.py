import io
import os
import time
import json
import torch
import zipfile
import numpy as np
import torch.nn as nn
from PIL import Image,ImageOps
import torch.nn.functional as F
from vidaug import augmentors as va
from einops import rearrange, repeat
import math
from torch import einsum
from argparse import ArgumentParser
from core.models.curvenet_cls import CurveNet
from tqdm import tqdm
from torchsummary import summary
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)
np.seterr(invalid='ignore')


torch.backends.cudnn.benchmark = True # Default


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        normalized = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        return normalized
    
def get_embedder(multires = 10, i=0):
    if i == -1:
        return nn.Identity(), 1

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 1,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

embeder = get_embedder()[0]    
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# GELU -> Gaussian Error Linear Units
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class RemixerBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        causal = False,
        bias = False
    ):
        super().__init__()
        self.causal = causal
        self.proj_in = nn.Linear(dim, 2 * dim, bias = bias)
        self.mixer = nn.Parameter(torch.randn(seq_len, seq_len))
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.proj_out = nn.Linear(dim, dim, bias = bias)

    def forward(self, x):
        mixer, causal, device = self.mixer, self.causal, x.device
        x, gate = self.proj_in(x).chunk(2, dim = -1)
        x = F.gelu(gate) * x

        if self.causal:
            seq = x.shape[1]
            mask_value = -torch.finfo(x.dtype).max
            mask = torch.ones((seq, seq), device = device, dtype=torch.bool).triu(1)
            mixer = mixer[:seq, :seq]
            mixer = mixer.masked_fill(mask, mask_value)

        mixer = mixer.softmax(dim = -1)
        mixed = einsum('b n d, m n -> b m d', x, mixer)

        alpha = self.alpha.sigmoid()
        out = (x * mixed) * alpha + (x - mixed) * (1 - alpha)

        return self.proj_out(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        
        # print(f'Attention:: {dim} - {heads} - {dim_head} - {dropout}')

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.pos_embedding = PositionalEncoding(dim,0.1,128)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x += self.pos_embedding(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()

        # print('\n')
        # print(f'Transformers:: {dim} - {depth} - {heads} - {dim_head} - {mlp_dim}')

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                #PreNorm(dim, RemixerBlock(dim,17))
            ]))

    def forward(self, x, swap = False):
        if swap: # for the self.transformer(x,swap = True)
            b, t, n , c = x.size() 
        for idx, (attn, ff) in enumerate(self.layers):
            if swap: # for the self.transformer(x,swap = True)
                if idx % 2 == 0:
                    #* attention along with all timesteps(frames) for each point(landmark)
                    x = rearrange(x, "b t n c -> (b n) t c")
                else:
                    #* attention to all points(landmarks) in each timestep(frame)
                    x = rearrange(x, "b t n c -> (b t) n c")
            x = attn(x) + x  # skip connections
            x = ff(x) + x    # skip connections
            
            # Now return the input x to its original formation
            if swap: # for the self.transformer(x,swap = True)
                if idx % 2 == 0:
                    x = rearrange(x, "(b n) t c -> b t n c", b = b)
                else:
                    x = rearrange(x, "(b t) n c -> b t n c", b = b)
                
        return x


class TemporalModel(nn.Module):
    
    def __init__(self):
        super(TemporalModel,self).__init__()
                
        self.encoder  =  CurveNet() # curve aggregation, needed for Point Clouds Shape Analysis. 
        self.downsample = nn.Sequential(
                            nn.Conv1d(478, 32, kernel_size=1, bias=False),
                            nn.BatchNorm1d(32),
                            # nn.Dropout(p=0.25), #* NEW
                            #nn.ReLU(inplace=True),
                            #nn.Conv1d(128, 32, kernel_size=1, bias=False),
                            #nn.BatchNorm1d(32),
                            )
        
        self.transformer = Transformer(256, 6, 4, 256//4, 256 * 2, 0.1)
        self.time = Transformer(256, 3, 4, 256//4, 256 * 2, 0.1)
        self.dropout = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        b,t,n,c = x.size()
    
        x = rearrange(x, "b t n c -> (b t) c n")
        x = rearrange(self.dropout(self.encoder(x)), "b c n -> b n c") 
        x = self.downsample(x).view(b,t,32,-1) #b t 32 c
        x = self.transformer(x,swap = True).view(b,t,-1,256).mean(2)
        x = self.time(x).mean(1)
        x = self.mlp_head(x)
        return x
        

min_xyz = np.array([0.06372425, 0.05751023, -0.08976112]).reshape(1,1,3)
max_xyz = np.array([0.63246971, 1.01475966, 0.14436169]).reshape(1,1,3)



class DataGenerator(torch.utils.data.Dataset):
    
    def __init__(self,data,label_path,test = False):
        self.data = data
        self.label_path = label_path
        self.__dataset_information()
        self.test = test

    def __dataset_information(self):
        self.numbers_of_data = 0

        with open(self.label_path) as f:
            labels = json.load(f)

        self.index_name_dic = dict()
        for index,(k,v) in enumerate(labels.items()):
            self.index_name_dic[index] = [k,v]

        self.numbers_of_data = index + 1

        output(f"Load {self.numbers_of_data} videos")
        print(f"Load {self.numbers_of_data} videos")

    def __len__(self):
        
        return self.numbers_of_data

    def __getitem__(self,idx):
        ids = self.index_name_dic[idx]
        size = 5 if self.test else 1 
        x, y = self.__data_generation(ids, size)
        return x,y
             
    def __data_generation(self,ids, size):
        name,label = ids
        y = torch.FloatTensor([label])
        
        clips = []
        for _ in range(size):
          x = np.load(os.path.join(self.data,f"{name}.mp4.npy"))
          start = x.shape[0] - 16

          if start > 0:
            start = np.random.randint(0,start) 
            x = x[start:][:16]
          else:
            start = np.random.randint(0,1)
            x = np.array(x)[start:]
        
          x = (x - min_xyz) / (max_xyz - min_xyz)
          pad_x = np.zeros((16,478,3))
          if x.shape[0] == 16:
            pad_x = x
          else:
            pad_x[:x.shape[0]] = x
          pad_x = torch.FloatTensor(pad_x) 
          clips.append(pad_x)
        clips = torch.stack(clips,0)
        return clips,y
    
perf = ""


def train(epochs,training_generator,test_generator,file):
    
    con = []      
    net = TemporalModel()
    net.cuda()
    
    lr = 0.0005
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr,weight_decay= 0.0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[299], gamma=0.1)
    loss_func = nn.BCELoss()
    start_time = time.time()
    best_accuracy = 0

    for epoch in range(epochs):
        train_loss  = 0
        pred_label = []
        true_label = []
        number_batch = 0
        for x, y in tqdm(training_generator, desc=f"Epoch {epoch}/{epochs-1}", ncols=60):
            if torch.cuda.device_count() > 0:
                x = x.cuda()
                y = y.cuda()
                
            b,d,t,n,c = x.size()
            x = x.view(-1,t,n,c)
            pred = net(x)
            loss = loss_func(pred,y)
            pred_y = (pred >= 0.5).float()
            pred_label.append(pred_y)
            true_label.append(y)
            
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            number_batch += 1
            lr = lr_scheduler.get_last_lr()[0]

        lr_scheduler.step()
        pred_label = torch.cat(pred_label,0)
        true_label = torch.cat(true_label,0)
        train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        output('Epoch: ' + 'train' + str(epoch) + 
              '| train accuracy: ' + str(train_accuracy.item())  + 
              '| train  loss: ' + str(train_loss / number_batch))
        print('Epoch: ' + 'train' + str(epoch) + 
              '| train accuracy: ' + str(train_accuracy.item())  + 
              '| train  loss: ' + str(train_loss / number_batch))
        
        net.eval()
        pred_label = []
        pred_avg   = []
        true_label = []
        with torch.no_grad():
          for x, y in tqdm(test_generator, desc=f"Epoch {epoch}/{epochs-1}", ncols=60):

              if torch.cuda.device_count() > 0:
                  x = x.cuda()
                  y = y.cuda()
                  
              b,d,t,n,c = x.size()
              x = x.view(-1,t,n,c)
              pred_y    = net(x)
              pred_mean = (pred_y.view(b,d).mean(1,keepdim = True) >= 0.5).float().cpu().detach()
              pred_y    = ((pred_y).view(b,d).mean(1,keepdim = True) >= 0.5).float().cpu().detach()
              pred_label.append(pred_y)
              pred_avg.append(pred_mean)
              true_label.append(y.cpu())
              
          pred_label = torch.cat(pred_label,0)
          pred_avg   = torch.cat(pred_avg,0)  
          true_label = torch.cat(true_label,0)
          
          test_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
          test_avg      = torch.sum(pred_avg   == true_label).type(torch.FloatTensor) / true_label.size(0)
          con.append([epoch,test_accuracy])
          output('test accuracy: ' + str(test_accuracy.item()) + 
                '| avg accuracy: '  + str(test_avg.item()))
          print(Fore.GREEN + 'test accuracy: ' + str(test_accuracy.item()) + 
                '| avg accuracy: '  + str(test_avg.item()))

          if test_accuracy > best_accuracy:
              filepath = f"uva/{file}-{epoch:}-{loss}-{test_accuracy}.pt"
              torch.save(net.state_dict(), filepath)
            #   torch.save(net, filepath)
            #   test_frames(f'{test_accuracy}={test_f}')
              best_accuracy = test_accuracy

        net.train()
        
        output(f"ETA Per Epoch:{(time.time() - start_time) / (epoch + 1)}")
        # print(f"ETA Per Epoch:{(time.time() - start_time) / (epoch + 1)}")

    best_v = max(con,key = lambda x:x[1])
    global perf
    perf += f"best accruacy is {best_v[1]} in epoch {best_v[0]}" + "\n"
    output(perf)
    
    
image_size = 48
label_path = "labels"
data = "npy"

sometimes = lambda aug: va.Sometimes(0.5, aug)
seq = va.Sequential([
    va.RandomCrop(size=(image_size, image_size)),       
    sometimes(va.HorizontalFlip()),              
])


label_path = "labels"

def main(args):
    global output
    def output(s):
        with open(f"log_m{args.fold}a","a") as f:
            f.write(str(s) + "\n")
        
    # paths = [os.path.join(label_path,file) for file in os.listdir(label_path) if os.path.join(label_path,file)] 
    paths = [os.path.join(label_path,file) for file in sorted(os.listdir(label_path)) if os.path.join(label_path,file)] 
    for current_path in [paths[args.fold]]: 
    
        train_labels = os.path.join(current_path,"train.json")         
        params = {"label_path": train_labels,
                  "data": data} 
                
        dg = DataGenerator(**params)
        training_generator = torch.utils.data.DataLoader(dg,batch_size=16,shuffle=True,num_workers = 2, drop_last = True)
                       
        test_labels    = os.path.join(current_path,"test.json")
        params = {"label_path": test_labels,
                  "data": data,
                  "test": True}    
                
        test_generator = torch.utils.data.DataLoader(DataGenerator(**params),batch_size=16,shuffle=False, num_workers = 2)
        
        train(300,training_generator,test_generator,current_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fold", default = 0, type = int)
    args = parser.parse_args()
    main(args)
