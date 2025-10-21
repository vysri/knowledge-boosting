import torch
import torch.nn as nn
from espnet2.torch_utils.get_layer_from_string import get_layer
import math

from src.models.common.LayerNorms import LayerNormalization4DCF_new

import torch.nn.functional as F


class FilmLayer(nn.Module):
    def __init__(self, D_in, D, freqs, use_alignment = False, delay = None):
        super().__init__()
        self.D = D
        self.F = freqs
        self.use_alignment = use_alignment

        if self.use_alignment:
            assert delay is not None, "If we use alignment, delay parameter must be provided"
            self.delay = delay
        
        self.weight = nn.Conv2d(D_in, self.D, 1)
        self.bias = nn.Conv2d(D_in, self.D, 1)

    def init_buffers(self, batch, device):
        state_buffer = dict()
        if self.use_alignment and self.delay > 0:
            input_buffer = torch.zeros(batch, self.D, self.F, self.delay, device=device)
            state_buffer['input_buffer'] = input_buffer
        
        return state_buffer

    def forward(self, x: torch.Tensor, embedding: torch.Tensor, init_state):
        """
        x: (B, D_in, F, T)
        embedding: (B, D, F, T)
        """
        # Align embedding
        if self.use_alignment and self.delay > 0:
            # embedding = F.pad(embedding, pad=(0, self.delay))
            x = torch.concatenate([init_state['input_buffer'], x], dim=-1)
            init_state['input_buffer'] = x[..., -init_state['input_buffer'].shape[-1]:]
            x = x[..., :-self.delay]

        # Apply film
        w = self.weight(embedding) # (B, D, F, T)
        b = self.bias(embedding) # (B, D, F, T)

        output = x * w + b

        # # Remove alignment
        # if self.use_alignment and self.delay > 0:
        #     output = output[..., :-self.delay]

        return output, init_state


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class CrossAtten_Layer(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, emb_dim,  n_head, n_freqs, ctx_len=1):
        super().__init__()
        activation="prelu"
        eps=1.0e-5
        approx_qk_dim = 256
        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)
        self.E = E
        self.emb_dim = emb_dim
         
        self.local_atten_len = 50 
        self.n_head = n_head
        self.n_freqs = n_freqs
        self.ctx_len = ctx_len
        assert emb_dim % n_head == 0
        self.add_module(
            "attn_conv_Q",
            nn.Sequential(
                nn.Linear(emb_dim, E * n_head), # [B, T, F, HE]
                get_layer(activation)(),
                # [B, T, F, H, E] -> [B, H, T, F, E] ->  [B * H, T, F * E]
                Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, E)\
                                    .permute(0, 3, 1, 2, 4)\
                                    .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * E)), # (BH, T, F * E)
                LayerNormalization4DCF_new((n_freqs, E), eps=eps),
            ),
        )
        self.add_module(
            "attn_conv_K",
            nn.Sequential(
                nn.Linear(emb_dim, E * n_head),
                get_layer(activation)(),
                Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, E)\
                                    .permute(0, 3, 1, 2, 4)\
                                    .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * E)),
                LayerNormalization4DCF_new((n_freqs, E), eps=eps),
            ),
        )
        self.add_module(
            "attn_conv_V",
            nn.Sequential(
                nn.Linear(emb_dim, (emb_dim // n_head) * n_head),
                get_layer(activation)(),
                Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, (emb_dim // n_head))\
                                    .permute(0, 3, 1, 2, 4)\
                                    .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * (emb_dim // n_head))),
                LayerNormalization4DCF_new((n_freqs, emb_dim // n_head), eps=eps),
            ),
        )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                get_layer(activation)(),
                Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])),
                LayerNormalization4DCF_new((n_freqs, emb_dim), eps=eps)
            ),
        )
        self.V_dim = emb_dim // n_head

    def init_buffers(self, batch, device):
            K_buf = torch.zeros((batch * self.n_head,
                                 self.ctx_len - 1,
                                 self.n_freqs * self.E), device=device)
            
            V_buf = torch.zeros((batch * self.n_head,
                                 self.ctx_len - 1,
                                 self.n_freqs * self.emb_dim // self.n_head), device=device)
            
            return dict(K_buf=K_buf, V_buf=V_buf)

    def _causal_unfold_chunk(self, x):
        """
        Unfolds the sequence into a batch of sequences
        prepended with `ctx_len` previous values.
        Args:
            x: [B, C, T, F], L is total length of signal
            ctx_len: int
        Returns:
            [B, num_chunk,  (ctx_len + chunk_size), C]
        """
        B, T, E = x.shape
        x = x.transpose(1, 2) ### B, E, T  

        B, E_n, T = x.shape

        x = x.unfold(2, self.ctx_len, 1)
        
        x = x.transpose(1, 2)

        B, num_chunk, E, L = x.shape

        assert(self.ctx_len == L) 
        x = x.reshape(B*num_chunk,  E, self.ctx_len)
        x = x.transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor, merged_context: torch.Tensor, init_state):
        """
        x: (B, D, F, T)
        embedding: (B, D_in, F, T)
        """
        x = x.permute(0, 3, 2, 1) ### (B, T, F, D)
        out = x
        merged_context = merged_context.permute(0, 3, 2, 1) ### (B, T, F, D)

        B, old_T, old_F, C  = x.shape
        
        Q = self["attn_conv_Q"](x) # [B', T, F * E] B' = B * n_head
        K = self["attn_conv_K"](merged_context) # [B', T, F * E]
        V = self["attn_conv_V"](merged_context) # [B', T, F*emb_dim // n_head]

        K_buf = init_state['K_buf']
        K = torch.cat([K_buf, K], dim = 1)
        init_state['K_buf'] = K[:, -K_buf.shape[1]:]

        V_buf = init_state['V_buf']
        V = torch.cat([V_buf, V], dim = 1)
        init_state['V_buf'] = V[:, -V_buf.shape[1]:]

        K = self._causal_unfold_chunk(K) ## # [B* n_head * num_chunk,  T, F * E]
        V = self._causal_unfold_chunk(V) ## # [B* n_head * num_chunk,  T, F * emb_dim // n_head]
        
        Q = Q.reshape(Q.shape[0]*Q.shape[1], 1, Q.shape[2]) # [B* n_head * num_chunk, 1, F * E]

        emb_dim = Q.shape[-1]
        # print("KQV", K.shape, Q.shape, V.shape)
        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B* n_head * num_chunk, 1, T]
        #print(Q.shape, V.shape, K.shape)
        attn_mat = F.softmax(attn_mat, dim=2)  # [B* n_head * num_chunk, 1, T]

        V = torch.matmul(attn_mat, V)  # [B* n_head * num_chunk, 1, F * emb_dim // n_head  ]

        V = V.reshape(-1, old_T, V.shape[-1]) # [B* n_head, num_chunk (T), F * emb_dim // n_head ]
        V = V.transpose(1, 2) # [B* n_head, F * emb_dim // n_head, T]
        batch = V.reshape(B, self.n_head, self.n_freqs, self.V_dim, old_T) # [B, n_head, F, emb_dim // n_head, T]
        batch =  batch.transpose(2, 3)  # [B, n_head,  emb_dim // n_head, F, T]

        batch = batch.reshape([B, self.n_head*self.V_dim, self.n_freqs, old_T])  # [B, emb_dim, F, T])
        batch = batch.permute(0, 3, 2, 1) # [B, T, F, emb_dim])
        batch = self["attn_concat_proj"](batch)  # [B,  T, F*emb_dim])
        batch = batch.reshape(B, old_T, self.n_freqs, C)

        out = batch + out
        out = out.permute(0, 3, 2, 1)  # [B, C, F, T])
        return out, init_state