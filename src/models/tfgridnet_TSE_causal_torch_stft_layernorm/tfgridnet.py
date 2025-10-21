import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from espnet2.enh.separator.abs_separator import AbsSeparator

from src.models.common.GridNetBlock import GridNetBlock
from src.models.common.merging import FilmLayer, CrossAtten_Layer


class TSELayer(nn.Module):
    def __init__(self, D, C, nF):
        super().__init__()
        self.D = D
        self.C = C
        self.nF = nF
        self.weight = nn.Conv1d(self.D, self.C * nF, 1)
        self.bias = nn.Conv1d(self.D, self.C * nF, 1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        """
        x: (B, D, F, T)
        embedding: (B, D,1)
        """
        # print(embedding.shape)
        # print("embedding", embedding.shape)
        B, D, _F, T = x.shape
        w = self.weight(embedding).reshape(B, self.C, _F, 1) # (B, C, F, 1)
        b = self.bias(embedding).reshape(B, self.C, _F, 1) # (B, C, F, 1)

        return x * w + b
'''
class TSELayer(nn.Module):
    def __init__(self, D, C, nF):
        super().__init__()
        self.D = D
        self.C = C
        self.nF = nF
        self.weight = nn.Conv1d(self.D, self.C * nF, 1, groups = 4)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        """
        x: (B, D, F, T)
        embedding: (B, D,1)
        """
        # print(embedding.shape)
        # print("embedding", embedding.shape)
        B, D, _F, T = x.shape
        w = self.weight(embedding).reshape(B, self.C, _F, 1) # (B, C, F, 1)

        return x * w
'''

class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        """
        x = x.permute(0, 2, 1) # [B, T, C]
        x = super().forward(x)
        x = x.permute(0, 2, 1) # [B, C, T]
        return x

class LayerNormPermuted2D(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted2D, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, F]
        """
        x = x.permute(0, 2, 3, 1) # [B, T, F, C]
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2) # [B, C, T, F]
        return x

class TFGridNet(AbsSeparator):
    """Offline TFGridNet

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in arXiv preprint arXiv:2211.12433, 2022.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in arXiv preprint arXiv:2209.03952, 2022.

    NOTES:
    As outlined in the Reference, this model works best when trained with variance
    normalized mixture input and target, e.g., with mixture of shape [batch, samples,
    microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
    must do the same for the target signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.

    Args:
        input_dim: placeholder, not used
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNet blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_approx_qk_dim: approximate dimention of frame-level key and value tensors
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNet model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
        self,
        input_dim,
        n_srcs=2,
        n_fft=128,
        stride=64,
        window="hann",
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        attn_buf_length=100,
        activation="prelu",
        eps=1.0e-5,
        ref_channel=-1,
        use_attn=True,
        generate_embeddings=False,
        use_cross_attention=True,
        use_masked_attention=True,
        use_alignment = False,
        delay = None,
        cross_attention_ctx_len=50
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
       
        # STFT params
        self.n_fft = n_fft
        self.stride = stride 

        self.n_freqs = n_freqs
        self.ref_channel = ref_channel
        self.emb_dim = emb_dim

        self.istft_pad = n_fft - stride
        # ISTFT overlap-add will affect this many chunks in the future
        self.istft_lookback = 1 + (self.istft_pad - 1) // self.istft_pad

        # Joint model params
        self.generate_embeddings = generate_embeddings
        self.delay = delay
        self.use_alignment = use_alignment
        self.use_cross_attention = use_cross_attention
        self.cross_attention_ctx_len = cross_attention_ctx_len

        t_ksize = 3
        self.t_ksize = t_ksize
        ks, padding = (t_ksize, 3), (0, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            
            LayerNormPermuted2D(emb_dim, eps=eps)
            #nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        self.TSE_blocks = nn.ModuleList([])
        self.film_blocks = nn.ModuleList([])
        self.cross_attention_blocks = nn.ModuleList([]) 

        for _i in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                    use_attn=use_attn,
                    local_atten_len=attn_buf_length,
                    use_masked_attention=use_masked_attention
                )
            )

            if _i < n_layers - 1:
                TSE_block = TSELayer(D=256, C = emb_dim, nF = self.n_freqs) # TSELayer LAYER
                self.TSE_blocks.append(TSE_block)
                
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=( self.t_ksize - 1, 1))        


    def add_embedding_layers(self, input_embedding_dim):
        assert not self.generate_embeddings, f"Embeddings can only be added if the model was initialized with the flag\
                                               generate_embeddings=False"

        for i in range(self.n_layers - 1):
            self.film_blocks.append(FilmLayer(D_in=input_embedding_dim,
                                              D =self.emb_dim,
                                              use_alignment=self.use_alignment,
                                              freqs=self.n_freqs,
                                              delay=self.delay))
            if self.use_cross_attention:
                self.cross_attention_blocks.append(CrossAtten_Layer(emb_dim=self.emb_dim,
                                                                    n_head=4,
                                                                    n_freqs=self.n_freqs,
                                                                    ctx_len=self.cross_attention_ctx_len))
    def init_buffers(self, batch_size, device):
        conv_buf = torch.zeros(batch_size, self.n_imics*2, self.t_ksize - 1, self.n_freqs,
                              device=device)
        deconv_buf = torch.zeros(batch_size, self.emb_dim, self.t_ksize - 1, self.n_freqs,
                              device=device)
        istft_buf = torch.zeros(batch_size, self.n_srcs, self.n_freqs, self.istft_lookback,
                                device=device)

        gridnet_buffers = {}
        for i in range(len(self.blocks)):
            gridnet_buffers[f'buf{i}'] = self.blocks[i].init_buffers(batch_size, device)

        ca_buffers = {}
        for i in range(len(self.cross_attention_blocks)):
            ca_buffers[f'buf{i}'] = self.cross_attention_blocks[i].init_buffers(batch_size, device)

        film_buffers = {}
        for i in range(len(self.film_blocks)):
            film_buffers[f'buf{i}'] = self.film_blocks[i].init_buffers(batch_size, device)

        return dict(conv_buf=conv_buf, deconv_buf=deconv_buf,
                    istft_buf=istft_buf, gridnet_bufs=gridnet_buffers,
                    ca_bufs=ca_buffers, film_bufs=film_buffers)

    def forward(
        self,
        input: torch.Tensor,
        spk_id: torch.Tensor,
        embed: torch.Tensor = None,
        input_state: dict = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor): batched multi-channel audio tensor with
                    M audio channels and N samples [B, M, N]

        Returns:
            output (List[Union(torch.Tensor)]): [B, C * S, T] of mono audio tensors with T samples.
        """
        
        if self.generate_embeddings:
            assert embed is None, 'Embedding should be None if we are generating embeddings (big model)'
        else:
            assert embed is not None, 'Embedding should not be None if we receiving embeddings (small model)'

        conv_buf = input_state['conv_buf']
        deconv_buf = input_state['deconv_buf']
        istft_buf = input_state['istft_buf']
        gridnet_buf = input_state['gridnet_bufs']
        ca_buf = input_state['ca_bufs']
        film_buf = input_state['film_bufs']
        
        # Changed to torch stft
        B, M, T = input.shape
        batch = input.reshape(B * M, T)
        batch = torch.stft(batch, n_fft = self.n_fft, hop_length = self.stride,
                           win_length = self.n_fft, window=torch.hamming_window(self.n_fft, device=input.device),
                           center=False, normalized=False, return_complex=False) # [B, M, nfft + 2, T]
        _, _F, T, C = batch.shape
        batch = batch.permute(0, 3, 1, 2) # [BM, C, F, T]
        batch = batch.reshape(B * M, C * _F, T)  # [BM, CF, T]

        # import numpy as np
        
        # print()
        # print(batch[0,0])
        # print(np.argmax(batch[0,0].isnan().numpy()))

        if self.n_imics == 1:
            assert len(batch.shape) == 3
            batch = batch.unsqueeze(1) # Unsqueeze mic dimension
        
        batch = batch.reshape(B, M, C * _F, T)  # [B, M, CF, T]

        batch = batch[..., :self.n_freqs, :] + 1j * batch[..., self.n_freqs:, :] # [B, M, F, T]
        
        batch0 = batch.transpose(2, 3) # [B, M, T, F]
        
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape # B, 2M, T, F

        batch = torch.cat((conv_buf, batch), dim=2)
        conv_buf = batch[:, :,  -(self.t_ksize - 1):, :]
        
        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch, gridnet_buf[f'buf{ii}'] = self.blocks[ii](batch, gridnet_buf[f'buf{ii}'])  # [B, -1, T, F]


            if ii < len(self.TSE_blocks):
                batch = batch.transpose(2,3) # [B, C, F, T]
                batch = self.TSE_blocks[ii](batch, spk_id)
                batch = batch.transpose(2,3) # [B, C, T, F]

            if ii < len(self.film_blocks):
                batch = batch.transpose(2,3) # [B, C, F, T]
                merged_embedding, film_buf[f'buf{ii}'] = self.film_blocks[ii](batch, embed, film_buf[f'buf{ii}'])
                if self.use_cross_attention:
                    batch, ca_buf[f'buf{ii}'] = self.cross_attention_blocks[ii](batch, merged_embedding, ca_buf[f'buf{ii}'])
                else:
                    assert not self.use_alignment, "If we are only using FiLM, aligning will remove latency"
                    batch = merged_embedding
                
                # print("FILM OUTPUT SHAPE", filmed_inputs[0].shape)
                # batch = torch.cat(filmed_inputs, dim=1)
                batch = batch.transpose(2,3) # [B, C, T, F]

        batch = torch.cat((deconv_buf, batch), dim=2)
        deconv_buf = batch[:, :,  -(self.t_ksize - 1):, :]
        
        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]batch ] 
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs]) # [B, n_srcs, 2, n_frames, n_freqs]
        
        batch = batch.transpose(3, 4) # (B, n_srcs, 2, n_fft//2 + 1, T)

        if self.generate_embeddings:
            embedding = batch.reshape(n_batch, self.n_srcs * 2, n_freqs, n_frames).clone() # [B, 2S, F, T]

        # Convert to complex
        batch = batch[:, :, 0] + 1j * batch[:, :, 1] # (B, S, n_fft//2 + 1, T)

        # Cat istft from previous chunks
        batch = torch.cat([istft_buf, batch], dim=3)
        istft_buf = batch[..., -self.istft_lookback:]

        B, S, _F, T = batch.shape
        batch = batch.reshape(B * S, _F, T)
        
        # Changed to torch ISTFT
        batch =  torch.istft(batch, n_fft = self.n_fft, hop_length = self.stride, center=False,
                           win_length = self.n_fft, window=torch.hamming_window(self.n_fft, device=batch.device),
                           normalized=False, return_complex=False) # [B, n_srcs, n_srcs, -1]
        batch = batch.reshape(B, S, -1)
        
        batch = batch[..., self.istft_lookback * self.stride:]

        # Update state buffer
        input_state['conv_buf'] = conv_buf
        input_state['deconv_buf'] = deconv_buf
        input_state['istft_buf'] = istft_buf
        input_state['gridnet_bufs'] = gridnet_buf
        input_state['ca_bufs'] = ca_buf
        input_state['film_bufs'] = film_buf
        
        if self.generate_embeddings:
            return batch, embedding, input_state
        else:
            return batch, input_state

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor



if __name__ == "__main__":
    net = CrossAtten_Layer(
        emb_dim = 16,
        n_head = 4,
        n_freqs = 81
    )    


    x = torch.rand(2, 16, 81, 1500)
    embed = torch.rand(2, 16, 81, 1500)
    y = net(x, embed)
