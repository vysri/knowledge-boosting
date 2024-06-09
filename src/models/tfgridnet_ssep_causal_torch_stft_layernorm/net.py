import torch
import torch.nn as nn

from .tfgridnet import TFGridNet
import torch.nn.functional as F


def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod

class Net(nn.Module):
    """
    WARNING: FOR STREAMING, use_masked_attention MUST BE FALSE, OTHERWISE PREVIOUS STATES WOULDN'T BE
    CACHED PROPERLY!!!
    """
    def __init__(self, stft_chunk_size=160, stft_pad_size = 120,
                 num_ch=2, num_src=2, D=64, B=6, I=1, J=1, L=8, H=128,
                 local_atten_len=100, use_attn=False, lookahead=True,
                 generate_embeddings=False, use_cross_attention=False,
                 use_masked_attention = True, use_alignment = False, delay = None,
                 cross_attention_ctx_len = 50, knowledge_distillation=False):
        super(Net, self).__init__()
        self.stft_chunk_size = stft_chunk_size
        self.stft_pad_size = stft_pad_size
        self.num_ch = num_ch
        self.lookahead = lookahead
        self.embed_dim = D
        self.num_src = num_src

        self.knowledge_distillation = knowledge_distillation
        
        # Input conv to convert input audio to a latent representation        
        self.nfft = stft_chunk_size + stft_pad_size

        # TF-GridNet
        self.tfgridnet = TFGridNet(None,
                                   n_srcs=num_src * num_ch,
                                   n_fft=self.nfft,
                                   stride=stft_chunk_size,
                                   emb_dim=D,
                                   emb_ks=I,
                                   emb_hs=J,
                                   n_layers=B,
                                   n_imics=num_ch,
                                   attn_n_head=L,
                                   use_attn = use_attn,
                                   lstm_hidden_units=H,
                                   attn_buf_length=local_atten_len,
                                   generate_embeddings=generate_embeddings,
                                   use_cross_attention=use_cross_attention,
                                   use_masked_attention = use_masked_attention,
                                   use_alignment=use_alignment,
                                   delay=delay,
                                   cross_attention_ctx_len=cross_attention_ctx_len,
                                   knowledge_distillation=knowledge_distillation)

        self.generate_embeddings = generate_embeddings

    def init_buffers(self, batch_size, device):
        return self.tfgridnet.init_buffers(batch_size, device)

    def predict(self, x, state_buffers, pad=True, embed=None):
        mod = 0
        if pad:
            pad_size = (0, self.stft_pad_size) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.stft_chunk_size, pad=pad_size)
        
        # Returns output, <optional> embedding, state buffers
        return_val = self.tfgridnet(x, embed=embed, input_state=state_buffers)
        
        x = return_val[0]
        x = x[..., : -self.stft_pad_size]
        
        if mod != 0:
            x = x[:, :, :-mod]

        x = x.reshape(x.shape[0], self.num_src, self.num_ch, x.shape[-1])
        t1 = x[:, 0]
        t2 = x[:, 1]

        return t1, t2, *return_val[1:]

    def forward(self, inputs, embed=None, current_state = None, pad=True):
        x = inputs['mixture']

        if current_state is None:
            current_state = self.init_buffers(x.shape[0], x.device)

        if self.knowledge_distillation:
            t1, t2, kd_lists, embed, next_state = self.predict(x, state_buffers=current_state, embed=embed, pad=pad)
            return {'output1': t1, 'output2': t2, 'gridnet_out':kd_lists, 'next_state': next_state}
        elif not self.generate_embeddings:
            t1, t2, next_state = self.predict(x, state_buffers=current_state, embed=embed, pad=pad)
            return {'output1': t1, 'output2': t2, 'next_state': next_state}
        else:
            t1, t2, embed, next_state = self.predict(x, state_buffers=current_state, embed=embed, pad=pad)
            return {'output1': t1, 'output2': t2, 'embed':embed, 'next_state': next_state}
