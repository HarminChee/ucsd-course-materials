import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def make_local_window_mask(t, window_size, device):
    if window_size is None:
        return None
    i = torch.arange(t, device=device).view(t, 1)
    j = torch.arange(t, device=device).view(1, t)
    return (j >= (i - window_size + 1))


class FeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_embd)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, causal, window_size=None):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.causal = causal
        self.window_size = window_size
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, pad_mask=None):
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.causal:
            m = self.tril[:t, :t].bool()
            lm = make_local_window_mask(t, self.window_size, att.device)
            if lm is not None:
                m = m & lm
            att = att.masked_fill(m == 0, float("-inf"))

        if pad_mask is not None:
            km = pad_mask.view(b, 1, 1, t)
            att = att.masked_fill(km == 0, float("-inf"))

        w = F.softmax(att, dim=-1)
        y = w @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.proj(y)
        w_avg = w.mean(dim=1)
        return y, w_avg


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, ffn_hidden, causal, window_size=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, block_size, causal, window_size=window_size)
        self.ffn = FeedForward(n_embd, ffn_hidden)

    def forward(self, x, pad_mask=None):
        a, w = self.attn(self.ln1(x), pad_mask=pad_mask)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, w


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, ffn_hidden):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, block_size, ffn_hidden, causal=False, window_size=None) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx):
        b, t = idx.shape
        t = min(t, self.block_size)
        idx = idx[:, :t]
        pos = torch.arange(t, device=idx.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        pad_mask = (idx != 0).to(x.dtype)

        attn_maps = []
        for blk in self.blocks:
            x, w = blk(x, pad_mask=pad_mask)
            attn_maps.append(w)

        x = self.ln_f(x)

        if t < self.block_size:
            pad_len = self.block_size - t
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
            padded_attn = []
            for w in attn_maps:
                w = F.pad(w, (0, pad_len, 0, pad_len), value=0.0)
                padded_attn.append(w)
            attn_maps = padded_attn

        return x, attn_maps


class EncoderClassifier(nn.Module):
    def __init__(self, encoder, n_embd, n_hidden, n_output):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(n_embd, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, idx):
        x, _ = self.encoder(idx)
        mask = (idx != 0).to(x.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (x * mask).sum(dim=1) / denom
        h = F.relu(self.fc1(pooled))
        return self.fc2(h)


class TransformerDecoderLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, ffn_hidden, window_size=None):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, block_size, ffn_hidden, causal=True, window_size=window_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        if t > self.block_size:
            idx = idx[:, -self.block_size:]
            if targets is not None:
                targets = targets[:, -self.block_size:]
            t = self.block_size

        pos = torch.arange(t, device=idx.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        attn_maps = []
        for blk in self.blocks:
            x, w = blk(x, pad_mask=None)
            attn_maps.append(w)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, attn_maps

        loss = F.cross_entropy(logits.reshape(b * t, -1), targets.reshape(b * t))
        return loss
