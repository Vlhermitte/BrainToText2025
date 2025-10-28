from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- CTC Encoder (What you need to focus on) ----------------------

class ConvSubsampler1D(nn.Module):
    """
    Simple 1D conv front-end over time to:
      - do local context mixing
      - reduce sequence length (strides)
      - project feature dim
    Input:  (T, B, F)  where F = feat_dim (e.g., 512)
    Output: (T', B, C) where C = conv_out_dim
    """
    def __init__(
        self,
        in_dim: int = 512,
        hidden: int = 256,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # We implement as Conv1d with shape (B, C_in, T), so we’ll permute in forward.
        self.net = nn.Sequential(
            # (B, in_dim, T) -> (B, hidden, T/2)
            nn.Conv1d(in_dim, hidden, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            # Depthwise separable style block
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),

            # Pointwise projection + second subsample: (B, hidden, T/2) -> (B, out_dim, T/4)
            nn.Conv1d(hidden, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # small residual pointwise to help training (optional)
        self.res_proj = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=4, bias=False)

    @staticmethod
    def _conv_out_len(L_in, kernel_size, stride, padding=0, dilation=1):
        # standard Conv1d length formula
        return torch.floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

    def compute_out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        # Track lengths through the two subsampling layers: k=5,s=2,p=2 then k=3,s=2,p=1
        L = lengths.to(torch.long)
        L = self._conv_out_len(L, 5, 2, 2)  # first conv
        L = self._conv_out_len(L, 3, 1, 1)  # depthwise (no downsample)
        L = self._conv_out_len(L, 3, 2, 1)  # second conv (downsample)
        return L

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: (T, B, F) -> (B, F, T)
        x = x.permute(1, 2, 0)
        y = self.net(x)
        # residual path (align channels/stride=4 total)
        res = self.res_proj(x)
        # crop to same T (just in case length math differs by 1)
        min_T = min(y.size(-1), res.size(-1))
        y = y[..., :min_T] + res[..., :min_T]

        # back to (T', B, C)
        y = y.permute(2, 0, 1).contiguous()
        out_lengths = self.compute_out_lengths(lengths)
        # (Potential single-step correction if residual crop changed T by 1)
        out_lengths = torch.clamp(out_lengths, max=y.size(0))
        # make out_lengths integer type
        out_lengths = out_lengths.to(torch.int)
        return y, out_lengths

class CTCEncoder(nn.Module):
    def __init__(self, input_dim=512, rnn_hidden=256, rnn_layers=2, vocab_size=129, blank_id=128, dropout=0.1, use_conv=True, use_gru=False):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.use_conv = use_conv

        # Convolutional front-end
        if self.use_conv:
            self.subsample = ConvSubsampler1D(
                in_dim=input_dim,
                hidden=256,
                out_dim=256,
                dropout=dropout,
            )

            # After conv, LSTM input dim equals conv out_dim
            input_dim = 256

        if use_gru:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0.0,
                bidirectional=True,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0.0,
                bidirectional=True,
            )

        self.proj = nn.Linear(2 * rnn_hidden, vocab_size)

        # Helpful init: bias final layer slightly against blank to reduce "all blank"
        with torch.no_grad():
            self.proj.bias.fill_(0.0)
            self.proj.bias[blank_id] = -2.0  # push down blank a bit at init

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: (T, B, 512), lengths: (B,)
        returns logits (T, B, C)
        """
        x = self.norm(x)
        if self.use_conv:
            # conv front-end (subsample in time)
            x, lengths = self.subsample(x, lengths)

        # pack for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)  # (T, B, 2*hidden)
        logits = self.proj(out)                                # (T, B, C)
        return logits, lengths

    def __str__(self):
        return f"ctc_encoder_rnn_{'gru' if isinstance(self.rnn, nn.GRU) else 'lstm'}_{self.rnn.num_layers}_layers"


# ---------------------- Conformer CTC (Just for fun) ----------------------

class Swish(nn.Module):
    def forward(self, x):  # swish = x * sigmoid(x)
        return x * torch.sigmoid(x)

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # broadcast over batch

class FeedForwardModule(nn.Module):
    """
    Conformer FFN (Macaron-compatible): LN -> Linear -> Swish -> Dropout -> Linear -> Dropout
    Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View
    Yiping Lu et al., arXiv 2019
    https://doi.org/10.48550/arXiv.1906.02762
    """
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = d_model * expansion_factor
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, inner)
        self.act = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(inner, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.act(y)
        y = self.dropout1(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        return y

class MultiHeadSelfAttentionModule(nn.Module):
    """
    MHSA with pre-LN; uses PyTorch MultiheadAttention (batch_first=True) for simplicity.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, D]
        key_padding_mask: [B, T] with True at PAD positions (will be ignored by attention)
        """
        y = self.layer_norm(x)
        y, _ = self.mha(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout(y)

class ConvModule(nn.Module):
    """
    Conformer convolution module:
    LN -> PW-Conv (expand) -> GLU -> DW-Conv (kernel) -> BN -> Swish -> PW-Conv (project) -> Dropout
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)  # for GLU
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.act = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        y = self.layer_norm(x)
        y = y.transpose(1, 2)  # [B, D, T] for conv1d
        y = self.pointwise_conv1(y)
        y = self.glu(y)  # [B, D, T]
        y = self.depthwise_conv(y)
        y = self.batch_norm(y)
        y = self.act(y)
        y = self.pointwise_conv2(y)
        y = y.transpose(1, 2)  # back to [B, T, D]
        return self.dropout(y)

class ConformerBlock(nn.Module):
    """
    One Conformer block with Macaron-style dual FFN.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        macaron_scale: float = 0.5
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.final_ln = nn.LayerNorm(d_model)
        self.mscale = macaron_scale

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.mscale * self.ffn1(x)
        x = x + self.mhsa(x, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + self.mscale * self.ffn2(x)
        return self.final_ln(x)

class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 4,
        ffn_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        max_len: int = 10000,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len) if use_positional_encoding else None
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_expansion=ffn_expansion,
                conv_kernel=conv_kernel,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [T, B, F]
        lengths: [B]
        Returns: enc [T, B, D], out_lens [B]
        """
        if x.dim() != 3:
            raise ValueError("Expected shape [T, B, F]")

        T, B, _ = x.shape

        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        # Move to batch-first for attention & conv
        x = x.permute(1, 0, 2)  # -> [B, T, F]

        key_padding_mask = self._make_key_padding_mask(lengths, T)

        x = self.input_proj(x)
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        # Return to [T, B, D]
        x = x.permute(1, 0, 2)
        return x, lengths

    @staticmethod
    def _make_key_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        # True where we want to mask (padding positions)
        range_row = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # [1, T]
        return range_row >= lengths.unsqueeze(1)  # [B, T]

class ConformerCTC(nn.Module):
    """
    End-to-end Conformer encoder with CTC projection head.
    """
    def __init__(
        self,
        vocab_size: int,
        blank_id: int,
        input_dim: int = 512,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 4,
        ffn_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        max_len: int = 10000,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            conv_kernel=conv_kernel,
            dropout=dropout,
            max_len=max_len,
            use_positional_encoding=use_positional_encoding,
        )
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # Helpful init: bias final layer slightly against blank to reduce "all blank"
        with torch.no_grad():
            self.ctc_head.bias.fill_(0.0)
            self.ctc_head.bias[blank_id] = -2.0  # push down blank a bit at init

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        log_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, 512] or [T, 512]
        lengths: [B] valid lengths (optional)
        Returns:
            probs:  [B, T, vocab] (log-probs if log_probs=True, else raw logits)
            out_lens: [B] output lengths (identical to input lengths when no subsampling)
        """
        enc, out_lens = self.encoder(x, lengths)    # enc: [T, B, D]
        logits = self.ctc_head(enc)  # [T, B, V]
        return (F.log_softmax(logits, dim=-1) if log_probs else logits), out_lens

    def __str__(self):
        return f"conformer_ctc_layers{len(self.encoder.layers)}"