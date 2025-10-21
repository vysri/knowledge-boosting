import torch
import torch.nn as nn
from espnet2.torch_utils.get_layer_from_string import get_layer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

import math

from src.models.common.LayerNorms import LayerNormalization4DCF


class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        local_atten_len= 100,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
        use_attn=True,
        use_masked_attention = True
    ):
        super().__init__()
        bidirectional = False
        self.use_attn = use_attn
        self.local_atten_len = local_atten_len
        self.n_freqs = n_freqs
        self.H = hidden_channels
        self.V_dim = emb_dim // n_head


        self.use_masked_attention = use_masked_attention

        in_channels = emb_dim * emb_ks
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels*2, emb_dim, emb_ks, stride=emb_hs
        )

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM( 
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=False
        )
        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels*(bidirectional + 1), emb_dim, emb_ks, stride=emb_hs
        )

        if self.use_attn:
            self.E = E = math.ceil(
                    approx_qk_dim * 1.0 / n_freqs
                )  # approx_qk_dim is only approximate
            assert emb_dim % n_head == 0
            for ii in range(n_head):
                self.add_module(
                    "attn_conv_Q_%d" % ii,
                    nn.Sequential(
                        nn.Conv2d(emb_dim, E, 1),
                        get_layer(activation)(),
                        LayerNormalization4DCF((E, n_freqs), eps=eps),
                    ),
                )
                self.add_module(
                    "attn_conv_K_%d" % ii,
                    nn.Sequential(
                        nn.Conv2d(emb_dim, E, 1),
                        get_layer(activation)(),
                        LayerNormalization4DCF((E, n_freqs), eps=eps),
                    ),
                )
                self.add_module(
                    "attn_conv_V_%d" % ii,
                    nn.Sequential(
                        nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                        get_layer(activation)(),
                        LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                    ),
                )
            self.add_module(
                "attn_concat_proj",
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
                ),
            )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def _causal_unfold_chunk(self, x):
        """
        Unfolds the sequence into a batch of sequences
        prepended with `ctx_len` previous values.

        Args:
            x: [B, T, CQ], L is total length of signal
            ctx_len: int
        Returns:
            [B * num_chunk, CQ, atten_len]
        """
        x = x.transpose(1, 2) # [B, CQ, T]
        
        if x.shape[-1] == self.local_atten_len:
            return x
        
        # print('A', x.shape)
        x = x.unfold(2, self.local_atten_len, 1) # [B, CQ, num_chunk, atten_len]
        
        B, CQ, N, L = x.shape
        x = x.transpose(1, 2).reshape(B * N, CQ, L)

        return x

    def get_lookahead_mask(self, seq_len, device):
        """Creates a binary mask for each sequence which maskes future frames.
        Arguments
        ---------
        padded_input: torch.Tensor
            Padded input tensor.
        Example
        -------
        >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
        >>> get_lookahead_mask(a)
        tensor([[0., -inf, -inf],
                [0., 0., -inf],
                [0., 0., 0.]])
        """
        if(seq_len <= self.local_atten_len):
            mask = (
                torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1
            ).transpose(0, 1)
        else:
            mask1 = torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1
            mask2 = torch.triu(torch.ones((seq_len, seq_len), device=device),  diagonal = self.local_atten_len) == 0
            mask = (
               mask1*mask2
            ).transpose(0, 1) 

        return mask.detach().to(device)
        
    def init_buffers(self, batch_size, device):
        ctx_buf = {}
        
        if self.use_attn:
            K_buf = torch.ones((batch_size * self.n_head,
                             self.E,
                             self.local_atten_len - 1,
                             self.n_freqs),
                             device=device) * torch.nan # Fix because of non-cached masked attention during training
            
            ctx_buf['K_buf'] = K_buf
            
            V_buf = torch.zeros((batch_size * self.n_head,
                                self.V_dim,
                                self.local_atten_len - 1,
                                self.n_freqs),
                                device=device)
            ctx_buf['V_buf'] = V_buf
            
        c0 = torch.zeros((1,
                          batch_size * self.n_freqs,
                          self.H), device=device)
        ctx_buf['c0'] = c0

        h0 = torch.zeros((1,
                          batch_size * self.n_freqs,
                          self.H), device=device)
        ctx_buf['h0'] = h0

        return ctx_buf

    def forward(self, x, init_state):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        )  # [BT, C, Q]
        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        self.intra_rnn.flatten_parameters()
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BF, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        self.inter_rnn.flatten_parameters()
        
        H0, C0 = init_state['h0'], init_state['c0']
        inter_rnn, (H0, C0) = self.inter_rnn(inter_rnn, (H0, C0))  # [BF, -1, H]
        init_state['h0'], init_state['c0'] = H0, C0
        
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]

        # Output is inter_rnn by default
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        out = inter_rnn

        if self.use_attn:
            # attention
            batch = inter_rnn
            
            if self.use_masked_attention:
                local_mask = self.get_lookahead_mask(inter_rnn.shape[2], inter_rnn.device)
                out = inter_rnn
                if self.use_attn:
                    # attention
                    batch = inter_rnn

                    all_Q, all_K, all_V = [], [], []
                    for ii in range(self.n_head):
                        all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
                        all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
                        all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

                    Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
                    K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
                    V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

                    Q = Q.transpose(1, 2)
                    Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
                    K = K.transpose(1, 2)
                    K = K.flatten(start_dim=2)  # [B', T, C*Q]
                    V = V.transpose(1, 2)  # [B', T, C, Q]
                    old_shape = V.shape
                    V = V.flatten(start_dim=2)  # [B', T, C*Q]

                    emb_dim = Q.shape[-1]
                    attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
                    
                    # Get causal mask
                    attn_mat.masked_fill_(local_mask==0, -float('Inf'))
                    
                    attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]                    
                    V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

                    V = V.reshape(old_shape)  # [B', T, C, Q]
                    V = V.transpose(1, 2)  # [B', C, T, Q]
                    emb_dim = V.shape[1]

                    batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [n_head, B, C, T, Q])
                    batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
                    batch = batch.contiguous().view(
                        [B, self.n_head * emb_dim, old_T, -1]
                    )  # [B, C, T, Q])
                    batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

                    # Add batch if attention is performed
                    out = out + batch
            else:               
                all_Q, all_K, all_V = [], [], []
                for ii in range(self.n_head):
                    all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
                    all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
                    all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]
                # print('K', batch.shape[0], batch[0, 0, -5:, 0])
                # Buffers
                K_buf = init_state['K_buf']
                V_buf = init_state['V_buf']
                
                # Get Q buf
                Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
                
                # Get K buf and update state buffer
                K = torch.cat(all_K, dim=0)  # [B', C, T, Q]

                K = torch.cat([K_buf, K], dim = 2)
                start = K.shape[2] - (self.local_atten_len-1)
                init_state['K_buf'] = K[:, :, start:start+self.local_atten_len - 1]

                # Get V buf and update state buffer
                V = torch.cat(all_V, dim=0)  # [B', C, T, Q]
                _C = V.shape[1]
                _T = V.shape[2]
                _Q = V.shape[3]

                V = torch.cat([V_buf, V], dim = 2)
                start = V.shape[2] - (self.local_atten_len-1)
                init_state['V_buf'] = V[:, :, start:start+self.local_atten_len - 1]

                K = K.transpose(1, 2)
                K = K.flatten(start_dim=2)  # [B', T, C*Q]
                V = V.transpose(1, 2)  # [B', T, C, Q]
                V = V.flatten(start_dim=2)  # [B', T, C*Q]

                Q = Q.transpose(1, 2) # [B', T, C, Q]
                Q = Q.reshape(Q.shape[0] * Q.shape[1], 1, Q.shape[2] * Q.shape[3])

                K = self._causal_unfold_chunk(K) # [B' * T, CQ, L]
                V = self._causal_unfold_chunk(V)

                emb_dim = Q.shape[-1]
                attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]
                
                # Fix for when key buffer is initially empty (change nan values to -inf so they become 0 in softmax)
                attn_mat = torch.nan_to_num(attn_mat, -float('Inf'))
                
                attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]

                V = torch.matmul(attn_mat, V.transpose(1, 2))  # [B', T, C*Q]

                V = V.reshape(V.shape[0]//_T, _T, _C, _Q)  # [B', T, C, Q]
                
                V = V.transpose(1, 2)  # [B', C, T, Q]
                emb_dim = V.shape[2]

                batch = V.reshape(self.n_head, B, _C, old_T, -1)  # [n_head, B, C, T, Q])
                batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
                batch = batch.contiguous().view(
                    [B, self.n_head * _C, old_T, -1]
                )  # [B, C, T, Q])
                batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

                # Add batch if attention is performed
                out = out + batch
        
        return out, init_state


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat
