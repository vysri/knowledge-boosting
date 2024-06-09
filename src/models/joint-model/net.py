import torch
import torch.nn as nn
import torch.nn.functional as F

class Embed(nn.Module):
    def __init__(self):
        super(Embed, self).__init__()
        self.max_pool = nn.MaxPool1d(8, stride=4)
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=4, stride=16, padding=1)

    def forward(self, input):
        #print("PRE SHAPE", input.shape)
        x = self.conv1d(input)
        #print("POST MP SHAPE", x.shape)
        return x


class Embed_Causal(nn.Module): ### The embed() is not causal because the current sample have info from future 2 samples due to kernel size = 4
    def __init__(self, D_in, D_out, freq):
        super(Embed_Causal, self).__init__()
        self.K = 3 ## kernel size
        self.conv1d = nn.Conv2d(in_channels=D_in, out_channels=D_out, kernel_size=(3, self.K)) ### also I think first step we do not need to set large stride size
        self.D_in = D_in
        self.D_out = D_out
        self.freq = freq

    def forward(self, input, current_state):
        # input - B x 2 x F X T 
        
        #print("PRE SHAPE", input.shape)
        # Pad in the time and frequency dimensions
        
        if self.K > 1:
            ctx_buf = current_state['ctx_buf']
            input = torch.concatenate([ctx_buf, input], axis=-1)
            current_state['ctx_buf'] = input[..., -ctx_buf.shape[-1]:]

        input =  nn.functional.pad(input, (0, 0, 1, 1)) ### pad zero in front of the tensor to guarantee the causality

        x = self.conv1d(input)

        #print("POST MP SHAPE", x.shape)
        
        outputs = {
            'output': x,
            'next_state': current_state
        }
        
        return outputs

    def init_buffers(self, batch, device):
        state_buffers = {
            'ctx_buf': torch.zeros(batch, self.D_in, self.freq, self.K - 1, device=device)
        }
        return state_buffers

class JointModel(nn.Module):
    def __init__(self, small_model, big_model, compression=1, delay = 0):
        super(JointModel, self).__init__()
        self.small = small_model
        self.big = big_model 
        
        self.embed_dim = self.big.tfgridnet.n_srcs * 2
        self.compression_dim = self.embed_dim//compression
        self.nfreqs = self.small.tfgridnet.n_freqs

        self.gen_embedding = Embed_Causal(self.embed_dim,
                                          self.compression_dim,
                                          self.nfreqs)
        self.delay = delay ## delay how many samples, 1 samples for 8ms

        self.small.tfgridnet.add_embedding_layers(self.compression_dim)

    def init_buffers(self, batch, device):
        state_buffers = dict()
        
        state_buffers['lm_bufs'] = self.big.init_buffers(batch, device)
        state_buffers['sm_bufs'] = self.small.init_buffers(batch, device)
        state_buffers['emb_bufs'] = self.gen_embedding.init_buffers(batch, device)
        
        return state_buffers

    def forward(self, inputs, current_state = None, pad=True, bm_poison_idx = None):
        x = inputs['mixture']

        if current_state is None:
            current_state = self.init_buffers(x.shape[0], x.device)
        
        bm_inputs = inputs

        # Poison big model at specific index. FOR DEBUGGING ONLY!!!!
        # Poison should affect small model `delay` chunks later
        if bm_poison_idx is not None:
            bm_inputs = {}
            bm_inputs['mixture'] = inputs['mixture'].clone()
            bm_inputs['mixture'][0, 0, bm_poison_idx] = torch.nan

        out_b = self.big(bm_inputs, current_state=current_state['lm_bufs'], pad=pad) # [B, 2S, F, T]
        current_state['lm_bufs'] = out_b['next_state']

        embed = out_b['embed']
        
        embed = self.gen_embedding(embed, current_state=current_state['emb_bufs']) # [B, S, F, T]
        current_state['emb_bufs'] = embed['next_state']
        embed = embed['output']
        
        # Delay embedding
        _e = torch.roll(embed, self.delay, 3)
        _e[..., 0:self.delay] = 0

        # Process with small model
        out_s = self.small(inputs, embed=_e, pad=pad, current_state=current_state['sm_bufs'])
        current_state['sm_bufs'] = out_s['next_state']
        
        return out_b, out_s