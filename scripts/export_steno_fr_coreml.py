#!/usr/bin/env python3
"""Export Sténo French EmbeddingGemma 2 + CTC Reader (guigssss/steno-fr-v1-large or base)
to a self-contained Core ML package (StenoFr.mlpackage) + metadata (steno_fr_meta.json).

Architecture:
  audio (1, 480000) float32 + n_frames (1,) int32
    -> Gemma4 LogMel (1, 3000, 128) [kept in float32]
    -> EmbeddingGemma 2 Audio Tower (12 layers, top layers fine-tuned for French, 13 states mixed) (1, 750, 1024)
    -> Sténo CTC Reader (causal conv k=15 + 10 or 6 bidirectional Transformer blocks + 30,002 Gemma French word-piece head)
    -> Top-8 log-probabilities `logp` (1, 750, 8) float32 + token IDs `ids` (1, 750, 8) int32
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoFeatureExtractor, AutoModel

SR = 16000
WINDOW_SAMPLES = 30 * SR  # 480,000 samples (30 s)
MEL_FRAMES = 3000
ENC_FRAMES = 750
TOPK = 8

# Operations kept in float32 inside Core ML to prevent float16 overflow in RMSNorm/LayerNorm and DFT
WIDE_OPS = {
    "pow",
    "square",
    "reduce_mean",
    "reduce_sum",
    "reduce_sum_square",
    "sqrt",
    "rsqrt",
    "real_div",
    "topk",
    "reduce_log_sum_exp",
    "log",
}

MODELS = {
    "base": "guigssss/steno-fr-v1-base",
    "large": "guigssss/steno-fr-v1-large",
}


# ---------------------------------------------------------------------------
# 1. Reference Sténo PyTorch Model (exact training/inference equivalence)
# ---------------------------------------------------------------------------

class RefLogMel(nn.Module):
    def __init__(self, n_mels: int = 128, sr: int = 16000, frame: int = 320, hop: int = 160, n_fft: int = 512):
        super().__init__()
        from transformers.audio_utils import mel_filter_bank, window_function
        import warnings
        self.frame, self.hop, self.n_fft = frame, hop, n_fft
        self.register_buffer("window", torch.tensor(window_function(frame), dtype=torch.float32), persistent=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mel = mel_filter_bank(
                num_frequency_bins=n_fft // 2 + 1,
                num_mel_filters=n_mels,
                min_frequency=0.0,
                max_frequency=sr / 2,
                sampling_rate=sr,
                norm=None,
                mel_scale="htk",
            )
        self.register_buffer("mel", torch.tensor(mel, dtype=torch.float32), persistent=False)

    def n_frames(self, n_samples: torch.Tensor) -> torch.Tensor:
        return ((n_samples - (self.frame + 1) + self.frame // 2) // self.hop + 1).clamp_min(1)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        w = F.pad(wave.float(), (self.frame // 2, 0))
        if w.shape[-1] < self.frame + 1:
            w = F.pad(w, (0, self.frame + 1 - w.shape[-1]))
        frames = w.unfold(-1, self.frame + 1, self.hop)[..., :-1] * self.window
        mag = torch.fft.rfft(frames, n=self.n_fft, dim=-1).abs()
        return torch.log(mag @ self.mel + 1e-3)


class HFTower(nn.Module):
    def __init__(self, audio_tower: nn.Module):
        super().__init__()
        self.at = audio_tower
        self.hidden = audio_tower.config.hidden_size
        self._mod = sys.modules[type(audio_tower).__module__]

    @property
    def layers(self):
        return self.at.layers

    def embed(self, mel: torch.Tensor):
        at = self.at
        dtype = next(at.parameters()).dtype
        h, _ = at.subsample_conv_projection(mel.to(dtype), None)
        pos = at.rel_pos_enc(h)
        full = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        c = at.config
        mask = self._mod.create_bidirectional_mask(
            config=c,
            inputs_embeds=h,
            attention_mask=full,
            and_mask_function=self._mod.sliding_window_mask_function(
                (c.attention_context_left - 1, c.attention_context_right)
            ),
        )
        if mask is not None:
            mask = at._convert_4d_mask_to_blocked_5d(mask)
        return h, (mask, pos)

    def layer(self, i: int, h: torch.Tensor, ctx) -> torch.Tensor:
        return self.layers[i](h, attention_mask=ctx[0], position_embeddings=ctx[1])


class RefReaderBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_mult: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.attn_norm = nn.LayerNorm(width)
        self.attn_in = nn.Linear(width, 3 * width)
        self.attn_out = nn.Linear(width, width)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp_up = nn.Linear(width, mlp_mult * width)
        self.mlp_down = nn.Linear(mlp_mult * width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, t, w = x.shape
        qkv = self.attn_in(self.attn_norm(x)).reshape(b, t, 3, self.heads, w // self.heads)
        q, k, v = (z.transpose(1, 2) for z in qkv.unbind(2))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.attn_out(a.transpose(1, 2).reshape(b, t, w))
        return x + self.mlp_down(self.drop(F.gelu(self.mlp_up(self.mlp_norm(x)))))


class RefStenoModel(nn.Module):
    def __init__(
        self,
        tower: nn.Module,
        n_units: int,
        enc_dim: int = 1024,
        d_model: int = 768,
        n_layers: int = 10,
        n_heads: int = 12,
        ff_mult: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.0,
        text_dim: int = 512,
        n_states: int = 13,
        unfreeze_layers=(9, 10, 11),
    ):
        super().__init__()
        self.logmel = RefLogMel()
        self.tower = tower
        n_tower = len(tower.layers)
        self.top = sorted(i % n_tower for i in unfreeze_layers)
        self.first_top = self.top[0] if self.top else n_tower
        self.mix_logits = nn.Parameter(torch.zeros(n_states))
        self.norm_in = nn.LayerNorm(enc_dim)
        self.proj_in = nn.Linear(enc_dim, d_model)
        self.conv_kernel = conv_kernel
        self.causal_conv = nn.Conv1d(d_model, d_model, conv_kernel, groups=d_model)
        self.blocks = nn.ModuleList(RefReaderBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers))
        self.norm_out = nn.LayerNorm(d_model)
        self.proj_text = nn.Linear(d_model, text_dim)
        self.head = nn.Linear(text_dim, n_units)

    def out_lengths(self, n_samples: torch.Tensor) -> torch.Tensor:
        mel = self.logmel.n_frames(n_samples)
        return ((mel + 1) // 2 + 1) // 2

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.mix_logits.float(), 0)
        h, ctx = self.tower.embed(mel)
        acc = w[0] * h.float()
        for i in range(len(self.tower.layers)):
            h = self.tower.layer(i, h, ctx)
            acc = acc + w[i + 1] * h.float()
        return acc

    def forward(self, wave: torch.Tensor, n_samples: torch.Tensor):
        mel = self.logmel(wave)
        x = self.encode(mel)
        b, t, _ = x.shape
        lens = self.out_lengths(n_samples).clamp_max(t)
        x = self.proj_in(self.norm_in(x))
        pad = torch.arange(t, device=x.device)[None, :] < lens[:, None]
        c = F.pad((x * pad[..., None].to(x.dtype)).transpose(1, 2), (self.conv_kernel - 1, 0))
        x = x + F.gelu(self.causal_conv(c).transpose(1, 2))
        mask = pad[:, None, None, :] if bool((lens < t).any()) else None
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.head(self.proj_text(self.norm_out(x)))
        return F.log_softmax(logits.float(), dim=-1), lens


# ---------------------------------------------------------------------------
# 2. Traceable Core ML Graph (Fixed 30 s window + dynamic n_frames mask)
# ---------------------------------------------------------------------------

def blocked_mask(seq_len: int, chunk: int, left: int, right: int) -> torch.Tensor:
    past = left - 1
    ctx = chunk + past + right
    nb = (seq_len + chunk - 1) // chunk
    q = torch.arange(nb)[:, None, None] * chunk + torch.arange(chunk)[None, :, None]
    k = torch.arange(nb)[:, None, None] * chunk - past + torch.arange(ctx)[None, None, :]
    ok = (k >= 0) & (k < seq_len) & (q < seq_len) & (k > q - past) & (k <= q + right)
    return ok[None, None]


class StaticAttention(nn.Module):
    """Gemma4AudioAttention for fixed sequence length T=750."""

    def __init__(self, a: nn.Module, seq_len: int, pos: torch.Tensor):
        super().__init__()
        self.a = a
        self.T = seq_len
        self.H, self.D = a.num_heads, a.head_dim
        self.C, self.P, self.F = a.chunk_size, a.max_past_horizon, a.max_future_horizon
        self.ctx = a.context_size
        self.nb = (seq_len + self.C - 1) // self.C
        self.cap = float(a.attention_logits_soft_cap)
        idx = torch.arange(self.nb)[:, None] * self.C + torch.arange(self.ctx)[None, :]
        self.register_buffer("kv_idx", idx.reshape(-1), persistent=False)
        with torch.no_grad():
            rk = a.relative_k_proj(pos).view(-1, self.H, self.D).permute(1, 2, 0).contiguous()
        self.register_buffer("rel_k", rk, persistent=False)
        self.n_pos = int(rk.shape[-1])
        self.register_buffer("qmul", (a.q_scale * F.softplus(a.per_dim_scale.detach())).float(), persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_embeddings=None, attention_mask=None):
        x, mask = hidden_states, attention_mask
        a, B, T, H, D, C, nb, ctx = self.a, x.shape[0], self.T, self.H, self.D, self.C, self.nb, self.ctx
        q = a.q_proj(x).view(B, T, H, D) * self.qmul
        k = a.k_proj(x).view(B, T, H, D) * a.k_scale
        v = a.v_proj(x).view(B, T, H, D)
        q = F.pad(q, (0, 0, 0, 0, 0, nb * C - T)).view(B, nb, C, H, D)
        pad_k = (0, 0, 0, 0, self.P, self.F + C - 1)
        k = F.pad(k, pad_k).index_select(1, self.kv_idx).view(B, nb, ctx, H, D)
        v = F.pad(v, pad_k).index_select(1, self.kv_idx).view(B, nb, ctx, H, D)
        qh = q.permute(0, 3, 1, 2, 4)
        ac = qh @ k.permute(0, 3, 1, 4, 2)
        bd = (qh.reshape(B, H, nb * C, D) @ self.rel_k).reshape(B, H, nb, C, -1)
        bd = F.pad(bd, (0, ctx + 1 - self.n_pos)).reshape(B, H, nb, C * (ctx + 1))[..., : C * ctx].reshape(B, H, nb, C, ctx)
        w = torch.tanh((ac + bd) / self.cap) * self.cap
        w = w.masked_fill(~mask, -1e4)
        w = torch.softmax(w, dim=-1)
        o = w @ v.permute(0, 3, 1, 2, 4)
        o = o.permute(0, 2, 3, 1, 4).reshape(B, nb * C, H * D)[:, :T]
        return a.post(o), None


class TraceableLogMel(nn.Module):
    """Gemma4AudioFeatureExtractor as a 1-D conv DFT so Core ML executes it natively in float32."""

    def __init__(self, fe, frames: int = MEL_FRAMES):
        super().__init__()
        n, hop, nfft = fe.frame_length, fe.hop_length, fe.fft_length
        self.pad_left, self.hop, self.n, self.frames = n // 2, hop, n, frames
        t = np.arange(n)[:, None]
        k = np.arange(nfft // 2 + 1)[None, :]
        ang = 2 * np.pi * t * k / nfft
        win = fe.window.astype(np.float64)[:, None]
        basis = np.concatenate([np.cos(ang) * win, -np.sin(ang) * win], axis=1).T
        self.scale = 64.0
        self.register_buffer("basis", torch.tensor(basis / self.scale, dtype=torch.float32)[:, None, :])
        self.register_buffer("mel", torch.tensor(fe.mel_filters, dtype=torch.float32))
        self.floor = float(fe.mel_floor)
        self.bins = nfft // 2 + 1

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = F.pad(wav, (self.pad_left, 0))[:, None, :]
        spec = F.conv1d(x, self.basis, stride=self.hop)
        spec = spec[:, :, : self.frames]
        re, im = spec[:, : self.bins], spec[:, self.bins :]
        mag = torch.sqrt(re * re + im * im)
        return torch.log((mag.transpose(1, 2) @ self.mel) * self.scale + self.floor)


class TraceableStenoEncoder(nn.Module):
    """Runs all 12 EmbeddingGemma 2 audio tower layers and returns the softmax-weighted sum of all 13 states."""

    def __init__(self, tower: nn.Module, mix_logits: torch.Tensor, mel_frames: int = MEL_FRAMES):
        super().__init__()
        self.t = tower
        c = tower.config
        self.n_layers = c.num_hidden_layers
        T = (mel_frames + 3) // 4
        self.register_buffer(
            "mask",
            blocked_mask(T, c.attention_chunk_size, c.attention_context_left, c.attention_context_right),
            persistent=False,
        )
        w = torch.softmax(mix_logits.detach().float(), dim=0)
        self.w = [float(x) for x in w.tolist()]
        pos = tower.rel_pos_enc(torch.zeros(1, T, c.hidden_size))
        for layer in tower.layers[: self.n_layers]:
            layer.self_attn = StaticAttention(layer.self_attn, T, pos)
        for layer in tower.layers:
            for m in (layer, layer.feed_forward1, layer.feed_forward2, layer.lconv1d):
                m.gradient_clipping = 1e10

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h, _ = self.t.subsample_conv_projection(mel, None)
        acc = h * self.w[0]
        for i, layer in enumerate(self.t.layers[: self.n_layers]):
            h = layer(h, attention_mask=self.mask, position_embeddings=None)
            acc = acc + h * self.w[i + 1]
        return acc


class TraceableReaderBlock(nn.Module):
    def __init__(self, blk: RefReaderBlock, seq_len: int = ENC_FRAMES):
        super().__init__()
        self.heads = blk.heads
        self.attn_norm = blk.attn_norm
        self.attn_in = blk.attn_in
        self.attn_out = blk.attn_out
        self.mlp_norm = blk.mlp_norm
        self.mlp_up = blk.mlp_up
        self.mlp_down = blk.mlp_down
        self.T = seq_len
        self.W = blk.attn_out.out_features
        self.D = self.W // self.heads
        self.scale = 1.0 / math.sqrt(self.D)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        qkv = self.attn_in(self.attn_norm(x)).view(1, self.T, 3, self.heads, self.D)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (1, H, T, D)
        k = qkv[:, :, 1].permute(0, 2, 3, 1)  # (1, H, D, T)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)  # (1, H, T, D)
        scores = (q @ k) * self.scale + attn_bias
        probs = torch.softmax(scores, dim=-1)
        a = (probs @ v).permute(0, 2, 1, 3).reshape(1, self.T, self.W)
        x = x + self.attn_out(a)
        return x + self.mlp_down(F.gelu(self.mlp_up(self.mlp_norm(x))))


class TraceableStenoTranscriber(nn.Module):
    def __init__(self, ref: RefStenoModel, fe):
        super().__init__()
        self.mel = TraceableLogMel(fe, frames=MEL_FRAMES)
        tower_copy = copy.deepcopy(ref.tower.at).eval()
        self.enc = TraceableStenoEncoder(tower_copy, ref.mix_logits, mel_frames=MEL_FRAMES)
        self.norm_in = copy.deepcopy(ref.norm_in).eval()
        self.proj_in = copy.deepcopy(ref.proj_in).eval()
        self.conv_kernel = ref.conv_kernel
        self.causal_conv = copy.deepcopy(ref.causal_conv).eval()
        self.blocks = nn.ModuleList(TraceableReaderBlock(copy.deepcopy(b).eval(), ENC_FRAMES) for b in ref.blocks)
        self.norm_out = copy.deepcopy(ref.norm_out).eval()
        self.proj_text = copy.deepcopy(ref.proj_text).eval()
        self.head = copy.deepcopy(ref.head).eval()
        self.register_buffer("pos", torch.arange(ENC_FRAMES, dtype=torch.float32).view(1, ENC_FRAMES), persistent=False)

    def forward(self, wav: torch.Tensor, n_frames: torch.Tensor):
        mel = self.mel(wav)
        acc = self.enc(mel)
        x = self.proj_in(self.norm_in(acc))
        nf = n_frames.to(torch.float32).view(1, 1)
        valid_mask = (self.pos < nf).to(x.dtype)  # (1, 750)
        c = F.pad((x * valid_mask.unsqueeze(-1)).transpose(1, 2), (self.conv_kernel - 1, 0))
        x = x + F.gelu(self.causal_conv(c).transpose(1, 2))
        attn_bias = ((valid_mask - 1.0) * 10000.0).view(1, 1, 1, ENC_FRAMES)
        for blk in self.blocks:
            x = blk(x, attn_bias)
        logits = self.head(self.proj_text(self.norm_out(x)))
        lp = F.log_softmax(logits.float(), dim=-1)
        v, i = lp.topk(TOPK, dim=-1)
        return v, i.to(torch.int32)


# ---------------------------------------------------------------------------
# 3. Greedy CTC Word-Piece Decoder (for verification)
# ---------------------------------------------------------------------------

def greedy_decode(ids: list[int], symbols: list[str], blank: int = 0, unk: int = 1) -> str:
    words: list[str] = []
    cur: str | None = None
    prev = blank
    for i in ids:
        if i == prev and i != blank:
            continue
        prev = i
        if i in (blank, unk) or i >= len(symbols) or symbols[i].startswith("<0x"):
            continue
        piece = symbols[i]
        if piece.startswith("▁") or cur is None:
            if cur:
                words.append(cur)
            cur = piece.lstrip("▁")
        else:
            cur += piece
    if cur:
        words.append(cur)
    return " ".join(words)


# ---------------------------------------------------------------------------
# 4. Loading, Verification & Core ML Export
# ---------------------------------------------------------------------------

def load_steno(model_key: str = "large"):
    repo_id = MODELS.get(model_key, model_key)
    print(f"[1/5] Downloading Sténo French weights from {repo_id}...")
    folder = Path(snapshot_download(repo_id, allow_patterns=["config.json", "units.json", "model.safetensors"]))
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    units_data = json.loads((folder / "units.json").read_text(encoding="utf-8"))
    symbols = units_data["symbols"]

    base_model = cfg["base_model"]
    base_rev = cfg["base_revision"]
    print(f"[2/5] Loading base EmbeddingGemma 2 audio tower from {base_model} @ {base_rev[:8]}...")
    full = AutoModel.from_pretrained(
        base_model,
        revision=base_rev,
        dtype=torch.float32,
    )
    tower = HFTower(full.audio_tower.eval())
    del full

    fe = AutoFeatureExtractor.from_pretrained(base_model, revision=base_rev)

    ref = RefStenoModel(
        tower=tower,
        n_units=len(symbols),
        enc_dim=cfg["enc_dim"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        ff_mult=cfg["ff_mult"],
        conv_kernel=cfg["conv_kernel"],
        dropout=0.0,
        text_dim=cfg["text_dim"],
        n_states=cfg["tower_states"],
        unfreeze_layers=cfg["unfrozen_tower_layers"],
    )
    sd = {
        ("tower.at.layers." + k[len("audio_tower.layers.") :] if k.startswith("audio_tower.layers.") else k): v
        for k, v in load_file(str(folder / "model.safetensors")).items()
    }
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    if unexpected or any(not k.startswith("tower.") for k in missing):
        raise RuntimeError(f"State dict mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    ref.eval()

    print("[3/5] Building TraceableStenoTranscriber for Core ML export...")
    traceable = TraceableStenoTranscriber(ref, fe).eval()
    return ref, traceable, cfg, symbols, repo_id


def synthesize_french_test_audio() -> np.ndarray:
    """Uses macOS `say` + `afconvert` if available to generate a real spoken French test clip."""
    sentence = "Bonjour, je m'appelle Jacques. Ceci est un test de transcription automatique en français sur macOS."
    try:
        with tempfile.TemporaryDirectory() as tmp:
            aiff_path = os.path.join(tmp, "test.aiff")
            raw_path = os.path.join(tmp, "test.f32")
            subprocess.run(["say", "-v", "Thomas", "-o", aiff_path, sentence], check=True, timeout=15)
            subprocess.run(
                ["afconvert", "-f", "caff", "-d", "LEF32@16000", "-c", "1", aiff_path, raw_path],
                check=False,
                timeout=15,
            )
            # Use ffmpeg if available for clean raw f32le extraction
            out = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", aiff_path, "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                capture_output=True,
                timeout=15,
            )
            if out.returncode == 0 and len(out.stdout) > 32000:
                wav = np.frombuffer(out.stdout, dtype=np.float32).copy()
                print(f"Synthesized French speech clip via macOS `say -v Thomas`: {len(wav)/SR:.2f}s")
                return wav
    except Exception as exc:
        print(f"Note: macOS `say` synthesis skipped ({exc}), using deterministic synthetic signal.")
    rng = np.random.default_rng(42)
    return (rng.standard_normal(4 * SR) * 0.05).astype(np.float32)


def compute_encoder_frames(n_samples: int) -> int:
    mel_frames = max(1, (n_samples - 321 + 160) // 160 + 1)
    enc_frames = ((mel_frames + 1) // 2 + 1) // 2
    return min(ENC_FRAMES, max(1, enc_frames))


def verify_pytorch_equivalence(ref: RefStenoModel, traceable: TraceableStenoTranscriber, symbols: list[str], wav: np.ndarray):
    n_samples = min(len(wav), WINDOW_SAMPLES)
    wav_clip = wav[:n_samples]
    n_frames = compute_encoder_frames(n_samples)

    wav_padded = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    wav_padded[0, :n_samples] = wav_clip

    with torch.no_grad():
        ref_lp, ref_lens = ref(torch.from_numpy(wav_clip)[None, :], torch.tensor([n_samples]))
        ref_top_ids = ref_lp[0, : int(ref_lens[0])].argmax(dim=-1).tolist()

        tr_lp, tr_ids = traceable(torch.from_numpy(wav_padded), torch.tensor([n_frames], dtype=torch.int32))
        tr_top_ids = tr_ids[0, :n_frames, 0].tolist()

    agree = np.mean(np.array(ref_top_ids[:n_frames]) == np.array(tr_top_ids[:n_frames]))
    ref_text = greedy_decode(ref_top_ids, symbols)
    tr_text = greedy_decode(tr_top_ids, symbols)
    print(f"PyTorch Ref vs Traceable ({n_frames} frames): slot agreement = {agree * 100:.2f}%")
    print(f"  Ref text      : {ref_text}")
    print(f"  Traceable text: {tr_text}")
    if agree < 0.98:
        raise RuntimeError(f"Traceable graph diverged from reference model (agreement {agree*100:.2f}% < 98%)")
    return wav_padded, n_frames, tr_top_ids, tr_text


def export_coreml(traceable: TraceableStenoTranscriber, out_dir: Path, bits: int = 8) -> Path:
    import coremltools as ct
    import coremltools.optimize as cto

    print("[4/5] Tracing and converting to Core ML (.mlpackage)...")
    wav_dummy = torch.zeros(1, WINDOW_SAMPLES, dtype=torch.float32)
    nf_dummy = torch.tensor([ENC_FRAMES], dtype=torch.int32)
    with torch.no_grad():
        ts = torch.jit.trace(traceable, (wav_dummy, nf_dummy))

    inputs = [
        ct.TensorType(name="audio", shape=(1, WINDOW_SAMPLES), dtype=np.float32),
        ct.TensorType(name="n_frames", shape=(1,), dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="logp", dtype=np.float32),
        ct.TensorType(name="ids", dtype=np.int32),
    ]

    probe = ct.convert(
        ts,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,
        convert_to="milinternal",
    )
    front = set()
    for op in probe.functions["main"].operations:
        if op.op_type == "conv" and op.inputs["x"].rank == 4:
            break
        front.add(op.name)

    mlmodel = ct.convert(
        ts,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.transform.FP16ComputePrecision(
            op_selector=lambda op: op.name not in front and op.op_type not in WIDE_OPS
        ),
    )

    if bits == 8:
        print("Applying 8-bit linear symmetric weight quantization...")
        q = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=4096)
        mlmodel = cto.coreml.linear_quantize_weights(
            mlmodel,
            cto.coreml.OptimizationConfig(global_config=q),
        )

    pkg_path = out_dir / "StenoFr.mlpackage"
    mlmodel.save(str(pkg_path))
    print(f"Saved Core ML package to {pkg_path}")
    return pkg_path


def verify_coreml_package(pkg_path: Path, wav_padded: np.ndarray, n_frames: int, ref_ids: list[int], symbols: list[str]):
    import coremltools as ct

    print("[5/5] Verifying Core ML execution on Apple Silicon...")
    cm = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    t0 = time.time()
    out = cm.predict({"audio": wav_padded, "n_frames": np.array([n_frames], dtype=np.int32)})
    dt = (time.time() - t0) * 1000.0
    cm_ids = out["ids"][0, :n_frames, 0].tolist()
    agree = np.mean(np.array(ref_ids) == np.array(cm_ids))
    cm_text = greedy_decode(cm_ids, symbols)
    print(f"Core ML execution time: {dt:.0f} ms | slot agreement vs PyTorch: {agree * 100:.2f}%")
    print(f"  Core ML French transcript: {cm_text}")
    if agree < 0.94:
        raise RuntimeError(f"Core ML model accuracy check failed (slot agreement {agree*100:.2f}% < 94%)")


def write_metadata(out_dir: Path, repo_id: str, cfg: dict, symbols: list[str]) -> Path:
    meta = {
        "version": 1,
        "model_id": repo_id,
        "sample_rate": SR,
        "window_samples": WINDOW_SAMPLES,
        "encoder_frames": ENC_FRAMES,
        "frame_ms": cfg.get("frame_ms", 40),
        "topk": TOPK,
        "blank_id": cfg.get("blank_id", 0),
        "unk_id": cfg.get("unk_id", 1),
        "symbols": symbols,
    }
    meta_path = out_dir / "steno_fr_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote metadata to {meta_path} ({meta_path.stat().st_size:,} bytes)")
    return meta_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large", help="'large' (default), 'base', or HF repo id")
    ap.add_argument("--out", required=True, help="Output directory for StenoFr.mlpackage and steno_fr_meta.json")
    ap.add_argument("--bits", type=int, default=8, choices=[8, 16], help="Weight quantization bits (default: 8)")
    ap.add_argument("--skip-coreml-predict", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref, traceable, cfg, symbols, repo_id = load_steno(args.model)
    wav = synthesize_french_test_audio()
    wav_padded, n_frames, tr_ids, _ = verify_pytorch_equivalence(ref, traceable, symbols, wav)
    del ref  # Free memory before Core ML conversion

    pkg_path = export_coreml(traceable, out_dir, bits=args.bits)
    write_metadata(out_dir, repo_id, cfg, symbols)

    if sys.platform == "darwin" and not args.skip_coreml_predict:
        verify_coreml_package(pkg_path, wav_padded, n_frames, tr_ids, symbols)


if __name__ == "__main__":
    main()
