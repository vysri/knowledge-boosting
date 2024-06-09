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
    def __init__(self, stft_chunk_size=160, stft_pad_size = 120,
                 num_ch=2,  D=64, B=6, I=1, J=1, L=8, H=128,
                 local_atten_len=100, use_attn=False, lookahead=True,
                 generate_embeddings=False, use_cross_attention=False,
                 use_masked_attention = True, use_alignment = False, delay = None,
                 cross_attention_ctx_len = 50):
        super(Net, self).__init__()
        self.stft_chunk_size = stft_chunk_size
        self.stft_pad_size = stft_pad_size
        self.num_ch = num_ch
        self.lookahead = lookahead
        self.embed_dim = D
        
        # Input conv to convert input audio to a latent representation        
        self.nfft = stft_chunk_size + stft_pad_size

        # TF-GridNet
        self.tfgridnet = TFGridNet(None,
                                   n_srcs= num_ch,
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
                                   cross_attention_ctx_len=cross_attention_ctx_len)

        self.generate_embeddings = generate_embeddings

    def init_buffers(self, batch_size, device):
        return self.tfgridnet.init_buffers(batch_size, device)

    def predict(self, x, spk_id, state_buffers, pad=True, embed=None):
        mod = 0
        if pad:
            pad_size = (0, self.stft_pad_size) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.stft_chunk_size, pad=pad_size)

        
        # Returns output, <optional> embedding, state buffers
        spk_id = spk_id.unsqueeze(2)
        return_val = self.tfgridnet(x, spk_id, embed=embed, input_state=state_buffers)
        
        x = return_val[0]
        x = x[..., : -self.stft_pad_size]
        
        if mod != 0:
            x = x[:, :, :-mod]

        return x, *return_val[1:]

    def forward(self, inputs, embed=None, current_state = None, pad=True):
        x = inputs['mixture']
        spk_id = inputs["spk_id"]

        if current_state is None:
            current_state = self.init_buffers(x.shape[0], x.device)

        if not self.generate_embeddings:
            x,  next_state = self.predict(x, spk_id, state_buffers=current_state, embed=embed, pad=pad)
            return {'output': x, 'next_state': next_state}
        else:
            x, embed, next_state = self.predict(x, spk_id, current_state, embed=embed, pad=pad)
            return {'output': x, 'embed':embed, 'next_state': next_state}
