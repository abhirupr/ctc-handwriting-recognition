"""Transformer CTC handwriting model (paper-aligned).

This file now contains the full implementation that previously lived in
`paper_model.py` (inverted bottleneck CNN + Transformer encoder + chunking).
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# Model: Isometric Fused Inverted Bottleneck Backbone + Transformer Encoder
# - Space-to-Depth (block size 4)
# - Conv pair (3x3 -> 128, 1x1 -> 64)
# - 10 Fused Inverted Bottleneck blocks (expansion 8x, out 64)
# - ReduceConv block (collapse H to 1, expand to 256)
# - Transformer encoder (depth configurable, hidden=256, 4 heads, rel pos bias)
# - Overlapping chunking for long width handling
################################################################################


class SpaceToDepth(nn.Module):
	def __init__(self, block_size: int = 4):
		super().__init__()
		self.block = block_size

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		b, c, h, w = x.size()
		bs = self.block
		assert h % bs == 0 and w % bs == 0, "Height/Width must be divisible by block size for SpaceToDepth"
		x = x.view(b, c, h // bs, bs, w // bs, bs)
		x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
		x = x.view(b, c * bs * bs, h // bs, w // bs)
		return x  # (B, 16, H/4, W/4)


class FusedInvertedBottleneck(nn.Module):
	def __init__(self, channels: int = 64, expansion: int = 8, kernel_size: int = 3, dropout: float = 0.0):
		super().__init__()
		inner = channels * expansion
		padding = kernel_size // 2
		self.conv1 = nn.Conv2d(channels, inner, kernel_size, padding=padding, bias=False)
		self.bn1 = nn.BatchNorm2d(inner)
		self.act = nn.GELU()
		self.conv2 = nn.Conv2d(inner, channels, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)
		self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.act(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.drop(out)
		return self.act(out + x)


class ReduceConvBlock(nn.Module):
	"""Collapse height dimension (H -> 1) while increasing channels to 256.
	Assumes input height = 10 (after space-to-depth)."""
	def __init__(self, in_ch=64, out_ch=256):
		super().__init__()
		self.reduce = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=(10, 3), padding=(0, 1), bias=False),
			nn.BatchNorm2d(out_ch),
			nn.GELU(),
		)

	def forward(self, x):
		return self.reduce(x)  # (B, out_ch, 1, W')


class RelativePositionBias(nn.Module):
	def __init__(self, num_heads: int, max_distance: int = 512):
		super().__init__()
		self.max_distance = max_distance
		self.bias = nn.Embedding(2 * max_distance + 1, num_heads)

	def forward(self, q_len: int, k_len: int) -> torch.Tensor:
		device = self.bias.weight.device
		q_pos = torch.arange(q_len, device=device)[:, None]
		k_pos = torch.arange(k_len, device=device)[None, :]
		rel = (q_pos - k_pos).clamp(-self.max_distance, self.max_distance) + self.max_distance
		bias = self.bias(rel)  # (q_len,k_len,H)
		return bias.permute(2, 0, 1)  # (H,q_len,k_len)


class TransformerEncoderLayer(nn.Module):
	def __init__(self, hidden=256, heads=4, mlp_ratio=4.0, dropout=0.1, max_rel_dist: int = 512):
		super().__init__()
		self.heads = heads
		self.hidden = hidden
		self.head_dim = hidden // heads
		assert hidden % heads == 0, "hidden must be divisible by heads"
		self.qkv = nn.Linear(hidden, hidden * 3)
		self.out_proj = nn.Linear(hidden, hidden)
		self.ln1 = nn.LayerNorm(hidden)
		self.ln2 = nn.LayerNorm(hidden)
		inner = int(hidden * mlp_ratio)
		self.mlp = nn.Sequential(
			nn.Linear(hidden, inner),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(inner, hidden),
		)
		self.dropout = nn.Dropout(dropout)
		self.rel_bias = RelativePositionBias(heads, max_rel_dist)
		self.scale = self.head_dim ** -0.5

	def forward(self, x):  # (B,T,C)
		B, T, C = x.shape
		qkv = self.qkv(x).view(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,T,D)
		attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
		bias = self.rel_bias(T, T)  # (H,T,T)
		attn_logits = attn_logits + bias.unsqueeze(0)
		attn = torch.softmax(attn_logits, dim=-1)
		attn = self.dropout(attn)
		out = torch.matmul(attn, v)
		out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
		out = self.out_proj(out)
		x = x + self.dropout(out)
		x = self.ln1(x)
		mlp_out = self.mlp(x)
		x = x + self.dropout(mlp_out)
		x = self.ln2(x)
		return x


class TransformerEncoder(nn.Module):
	def __init__(self, depth=16, hidden=256, heads=4, dropout=0.1, max_rel_dist: int = 512):
		super().__init__()
		self.layers = nn.ModuleList([
			TransformerEncoderLayer(hidden=hidden, heads=heads, dropout=dropout, max_rel_dist=max_rel_dist)
			for _ in range(depth)
		])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x


def sinusoidal_position_encoding(length: int, dim: int, device) -> torch.Tensor:
	pe = torch.zeros(length, dim, device=device)
	position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
	div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
	pe[:, 0::2] = torch.sin(position * div_term)
	pe[:, 1::2] = torch.cos(position * div_term)
	return pe


class PaperBackbone(nn.Module):
	def __init__(self, dropout: float = 0.1):
		super().__init__()
		self.s2d = SpaceToDepth(4)
		self.initial = nn.Sequential(
			nn.Conv2d(16, 128, 3, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.GELU(),
			nn.Conv2d(128, 64, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.GELU(),
		)
		blocks = []
		for _ in range(10):
			blocks.append(FusedInvertedBottleneck(64, expansion=8, dropout=dropout))
		self.blocks = nn.Sequential(*blocks)
		self.reduce = ReduceConvBlock(64, 256)

	def forward(self, x):
		x = self.s2d(x)
		x = self.initial(x)
		x = self.blocks(x)
		x = self.reduce(x)
		return x


class CTCTransformerModel(nn.Module):
	"""Paper-aligned optical model + transformer encoder with chunking & CTC logits."""
	def __init__(self, vocab_size: int, depth: int = 16, dropout: float = 0.1,
				 chunk_width: int = 320, pad: int = 32):
		super().__init__()
		self.vocab_size = vocab_size
		self.chunk_width = chunk_width
		self.pad = pad
		self.backbone = PaperBackbone(dropout=dropout)
		self.encoder = TransformerEncoder(depth=depth, hidden=256, heads=4, dropout=dropout)
		self.classifier = nn.Linear(256, vocab_size)
		self.dropout = nn.Dropout(dropout)
		self.supports_batch_ctc = True

	def _compute_chunks(self, img: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
		_, _, _, W = img.shape
		cw, pad = self.chunk_width, self.pad
		if W <= cw:
			if W < cw:
				img = F.pad(img, (0, cw - W, 0, 0))
			return [img], W
		stride = cw - 2 * pad
		chunks = []
		start = 0
		while start < W:
			end = min(start + cw, W)
			slice_ = img[:, :, :, start:end]
			if slice_.size(-1) < cw:
				slice_ = F.pad(slice_, (0, cw - slice_.size(-1), 0, 0))
			chunks.append(slice_)
			if end == W:
				break
			start += stride
		return chunks, W

	def _process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
		w = chunk.size(-1)
		pad_needed = (4 - (w % 4)) % 4
		if pad_needed:
			chunk = F.pad(chunk, (0, pad_needed, 0, 0))
		feats = self.backbone(chunk)
		feats = feats.squeeze(2).permute(0, 2, 1)
		return feats

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_seqs = []
		max_T = 0
		for img in x:
			img = img.unsqueeze(0)
			chunks, original_w = self._compute_chunks(img)
			chunk_feats = []
			for ci, ch in enumerate(chunks):
				feats = self._process_chunk(ch)[0]
				if len(chunks) == 1:
					trimmed = feats
				else:
					pad_feat = self.pad // 4
					if ci == 0:
						trimmed = feats[:-pad_feat] if pad_feat > 0 else feats
					elif ci == len(chunks) - 1:
						trimmed = feats[pad_feat:] if pad_feat > 0 else feats
					else:
						trimmed = feats[pad_feat:-pad_feat] if pad_feat > 0 else feats
				chunk_feats.append(trimmed)
			seq = torch.cat(chunk_feats, dim=0)
			target_len = math.ceil(original_w / 4)
			if seq.size(0) > target_len:
				seq = seq[:target_len]
			batch_seqs.append(seq)
			max_T = max(max_T, seq.size(0))
		padded = []
		for seq in batch_seqs:
			if seq.size(0) < max_T:
				seq = F.pad(seq, (0, 0, 0, max_T - seq.size(0)))
			padded.append(seq.unsqueeze(0))
		seq_batch = torch.cat(padded, dim=0)
		pe = sinusoidal_position_encoding(seq_batch.size(1), seq_batch.size(2), seq_batch.device)
		seq_batch = seq_batch + pe.unsqueeze(0)
		enc = self.encoder(seq_batch)
		enc = self.dropout(enc)
		logits = self.classifier(enc)
		return logits

__all__ = ["CTCTransformerModel"]
