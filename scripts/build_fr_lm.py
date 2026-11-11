#!/usr/bin/env python3
"""Build a compact, memory-mapped 4-gram Interpolated Kneser-Ney French language model (`lm_fr.bin`)
in the `JNLM` binary format read by `NGramLM` in `EG2Transcribe.app`.

Corpus:
  1. Vocabulary seeded from the top 64,000 normalized words of `guigssss/steno-fr-v1-large` (`lm/unigrams.txt`).
  2. Spoken French transcripts (`manu/french_librispeech_text_only`, weighted 3x).
  3. French Wikipedia (`wikimedia/wikipedia` `20231101.fr/train-00000-of-00017.parquet`).
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import time
from pathlib import Path

import numpy as np

BOS = "<s>"
UNK = "<unk>"
SEP = "\x1f"
D = 0.75
STRIP_CHARS = ".,;:!?\"'()[]{}«»—–-… \t\r\n"


def _fr_below_100(n: int) -> str:
    units = [
        "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
        "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf",
    ]
    tens = ["", "", "vingt", "trente", "quarante", "cinquante", "soixante"]
    if n < 20:
        return units[n]
    if n < 70:
        t, u = divmod(n, 10)
        if u == 0:
            return tens[t]
        if u == 1:
            return f"{tens[t]}-et-un"
        return f"{tens[t]}-{units[u]}"
    if n < 80:
        u = n - 60
        if u == 11:
            return "soixante-et-onze"
        return f"soixante-{units[u]}"
    if n == 80:
        return "quatre-vingts"
    u = n - 80
    return f"quatre-vingt-{units[u]}"


def int_to_fr(n: int) -> str:
    if n < 100:
        return _fr_below_100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        prefix = "cent" if h == 1 else f"{_fr_below_100(h)} cent"
        if r == 0:
            return prefix if h == 1 else f"{prefix}s"
        return f"{prefix} {_fr_below_100(r)}"
    th, r = divmod(n, 1000)
    prefix = "mille" if th == 1 else f"{int_to_fr(th)} mille"
    if r == 0:
        return prefix
    return f"{prefix} {int_to_fr(r)}"


FR_NUM_WORDS: dict[str, list[str]] = {str(i): int_to_fr(i).split() for i in range(10000)}


def clean_word(w: str) -> str:
    """Normalizes a surface token to match `FrenchTranscriber.cleanWordForLM` in Swift."""
    w = w.replace("\u2019", "'").replace("\u2018", "'").strip(STRIP_CHARS).lower()
    if not w or not any(c.isalpha() for c in w):
        return ""
    return w


def fnv64(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def load_vocab_from_unigrams(unigrams_path: Path, max_vocab: int = 64000) -> tuple[dict[str, int], list[str]]:
    stoi = {BOS: 0, UNK: 1}
    itos = [BOS, UNK]
    # Ensure all standard French number words (0..99) are in the vocabulary
    for i in range(100):
        w = _fr_below_100(i)
        if w not in stoi:
            stoi[w] = len(itos)
            itos.append(w)
    for extra in ["cent", "cents", "mille", "million", "millions", "milliard", "milliards", "virgule", "point"]:
        if extra not in stoi:
            stoi[extra] = len(itos)
            itos.append(extra)
    with open(unigrams_path, encoding="utf-8") as f:
        for line in f:
            w = clean_word(line)
            if w and w not in stoi:
                stoi[w] = len(itos)
                itos.append(w)
                if len(itos) >= max_vocab:
                    break
    return stoi, itos


def tokenize_sentence(text: str, stoi: dict[str, int]) -> list[int]:
    out: list[int] = []
    for raw in text.replace("\n", " ").split():
        stripped = raw.strip(STRIP_CHARS)
        if stripped in FR_NUM_WORDS:
            for nw in FR_NUM_WORDS[stripped]:
                out.append(stoi.get(nw, 1))
            continue
        w = clean_word(raw)
        if w:
            out.append(stoi.get(w, 1))
    return out


def generate_spoken_number_corpus():
    """Yields natural French spoken number expressions so all number transitions (e.g. 'mille neuf cent soixante-trois')
    have strong 2-gram, 3-gram, and 4-gram probabilities."""
    # 1. All numbers 0..9999 in both hyphenated and space-separated forms
    for n in range(10000):
        words = int_to_fr(n)
        yield words
        if "-" in words:
            yield words.replace("-", " ")

    # 2. Years 1100..2099 in common spoken phrases + "dix-neuf cent ..." form
    prefixes = ["en", "depuis", "vers", "année", "né en", "en l'an", "numéro", "ticket", "page", "ligne"]
    for yr in range(1100, 2100):
        std = int_to_fr(yr)
        p = prefixes[yr % len(prefixes)]
        yield f"{p} {std}"
        yield f"{p} {std.replace('-', ' ')}"
        if 1100 <= yr < 2000:
            h, r = divmod(yr, 100)
            alt = f"{_fr_below_100(h)} cents" if r == 0 else f"{_fr_below_100(h)} cent {_fr_below_100(r)}"
            yield f"{p} {alt}"
            yield f"{p} {alt.replace('-', ' ')}"

    # 3. Percentages, decimals, and technical dictation triggers
    for n in range(1, 200):
        w = int_to_fr(n)
        yield f"{w} pour cent"
        yield f"{w} euros"
        yield f"ticket {w}"
        yield f"version {_fr_below_100(n % 20)} point {_fr_below_100((n * 3) % 20)}"
        yield f"{_fr_below_100(n % 50)} virgule {_fr_below_100((n * 7) % 100)}"


def build_stream(sentence_iter, stoi: dict[str, int], order: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    buf: list[int] = []
    bos_prefix = [0] * (order - 1)
    for sent in sentence_iter:
        ids = tokenize_sentence(sent, stoi)
        if len(ids) < 2:
            continue
        buf.extend(bos_prefix)
        buf.extend(ids)
        if len(buf) >= 2_000_000:
            chunks.append(np.array(buf, dtype=np.uint16))
            buf.clear()
    if buf:
        chunks.append(np.array(buf, dtype=np.uint16))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)


def ngrams(stream: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Unique k-grams (rows, sorted) and their counts; excludes any k-gram ending in BOS (0) or containing UNK (1)."""
    n = len(stream) - k + 1
    rows = np.stack([stream[j : j + n] for j in range(k)], axis=1)
    valid = (rows[:, -1] != 0) & np.all(rows != 1, axis=1)
    rows = rows[valid]
    uniq, counts = np.unique(rows, axis=0, return_counts=True)
    return uniq, counts.astype(np.int64)


def _void(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def find_rows(table: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if rows.shape[1] == 0 or len(table) == 0:
        return np.zeros(len(rows), dtype=np.int64)
    tv = _void(table.astype(">u2"))
    rv = _void(rows.astype(">u2"))
    idx = np.searchsorted(tv, rv)
    idx = np.minimum(idx, len(tv) - 1)
    return np.where(tv[idx] == rv, idx, -1)


def thresholds(counts: dict[int, np.ndarray], order: int, budget: int) -> tuple[int, ...]:
    out = [1]
    share = max(1, (budget - len(counts[1])) // (order - 1))
    for k in range(2, order + 1):
        c = np.sort(counts[k])[::-1]
        t = int(c[share]) + 1 if len(c) > share else 1
        out.append(max(t, 2 if k >= 3 else 1))
    return tuple(out)


def build_kneser_ney(stream: np.ndarray, itos: list[str], order: int = 4, budget: int = 4_000_000):
    t0 = time.time()
    print(f"Building {order}-gram Kneser-Ney model from {len(stream):,} tokens ({len(itos):,} vocab)...", flush=True)
    grams: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    for k in range(1, order + 1):
        grams[k], counts[k] = ngrams(stream, k)
        print(f"  {k}-grams: {len(grams[k]):,}", flush=True)

    cont = np.bincount(grams[2][:, 1], minlength=len(itos)).astype(np.float64)
    cont_total = float(max(1, len(grams[2])))
    p: dict[int, np.ndarray] = {1: (cont[grams[1][:, 0]] + 0.5) / (cont_total + 0.5 * 10000)}
    bow: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for k in range(2, order + 1):
        g, c = grams[k], counts[k]
        hist, inv = np.unique(g[:, :-1], axis=0, return_inverse=True)
        inv = inv.ravel()
        c_h = np.bincount(inv, weights=c).astype(np.float64)
        types_h = np.bincount(inv).astype(np.float64)
        lower = find_rows(grams[k - 1], g[:, 1:])
        p_lower = np.where(lower >= 0, p[k - 1][np.maximum(lower, 0)], 0.5 / (cont_total + 5000))
        p[k] = np.maximum(c - D, 0) / c_h[inv] + D * types_h[inv] / c_h[inv] * p_lower
        bow[k] = (hist, np.log10(D * types_h / c_h))

    min_count = thresholds(counts, order, budget) if budget else (1, 1, 2, 2)
    print(f"  Pruning thresholds for budget ~{budget:,}: {min_count}", flush=True)

    entries: dict[tuple[int, ...], list[float]] = {}
    for k in range(1, order + 1):
        keep = counts[k] >= min_count[k - 1]
        for row, lp in zip(grams[k][keep], np.log10(p[k][keep])):
            entries[tuple(int(x) for x in row)] = [float(lp), 0.0]

    for k in range(2, order + 1):
        hist, lb = bow[k]
        for row, b in zip(hist, lb):
            key = tuple(int(x) for x in row)
            if key in entries:
                entries[key][1] = float(b)
            elif all(x == 0 for x in key):
                entries[key] = [0.0, float(b)]

    unk = math.log10(0.05 / (cont_total + 5000))
    print(f"  Retained {len(entries):,} entries ({time.time() - t0:.1f}s), unk={unk:.3f}", flush=True)
    return entries, unk


def write_jnlm(entries: dict[tuple[int, ...], list[float]], itos: list[str], unk: float, order: int, out_path: Path):
    keys = np.array([fnv64(SEP.join(itos[i] for i in g)) for g in entries], dtype=np.uint64)
    vals = np.array(list(entries.values()), dtype=np.float16)
    o = np.argsort(keys)
    rec = np.zeros(len(keys), dtype=[("k", "<u8"), ("p", "<f2"), ("b", "<f2")])
    rec["k"], rec["p"], rec["b"] = keys[o], vals[o, 0], vals[o, 1]
    with open(out_path, "wb") as f:
        f.write(b"JNLM" + struct.pack("<IIIf", 1, order, len(keys), unk))
        f.write(rec.tobytes())
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(keys):,} entries)", flush=True)


class VerifyBackoffLM:
    def __init__(self, path: Path):
        data = path.read_bytes()
        assert data[:4] == b"JNLM"
        _, self.n, count, self.unk = struct.unpack("<IIIf", data[4:20])
        rec = np.frombuffer(data[20:], dtype=[("k", "<u8"), ("p", "<f2"), ("b", "<f2")], count=count)
        self.keys = rec["k"]
        self.p = rec["p"].astype(np.float32)
        self.b = rec["b"].astype(np.float32)

    def _find(self, g: tuple[str, ...]) -> int:
        k = np.uint64(fnv64(SEP.join(g)))
        i = int(np.searchsorted(self.keys, k))
        return i if i < len(self.keys) and self.keys[i] == k else -1

    def logp(self, hist: list[str], w: str) -> float:
        h = tuple(hist[-(self.n - 1) :])
        if len(h) < self.n - 1:
            h = (BOS,) * (self.n - 1 - len(h)) + h
        bo = 0.0
        for k in range(len(h), -1, -1):
            hh = h[len(h) - k :]
            i = self._find(hh + (w,))
            if i >= 0:
                return bo + float(self.p[i])
            if k > 0:
                j = self._find(hh)
                if j >= 0:
                    bo += float(self.b[j])
        return bo + self.unk

    def sentence_score(self, words: list[str]) -> float:
        hist: list[str] = []
        total = 0.0
        for w in words:
            cw = clean_word(w)
            if cw:
                total += self.logp(hist, cw)
                hist.append(cw)
        return total


def stream_hf_corpora(wiki_chars: int = 45_000_000, spoken_weight: int = 3):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    print("Generating spoken French numbers & years corpus...", flush=True)
    for _ in range(3):
        for line in generate_spoken_number_corpus():
            yield line

    print("Downloading spoken French transcripts (manu/french_librispeech_text_only)...", flush=True)
    mls_parquet = hf_hub_download(
        repo_id="manu/french_librispeech_text_only",
        repo_type="dataset",
        filename="data/train-00000-of-00001-494661a0baeb14c0.parquet",
    )
    spoken_lines: list[str] = []
    pf_mls = pq.ParquetFile(mls_parquet)
    for batch in pf_mls.iter_batches(columns=["text"], batch_size=4000):
        for text in batch.column(0).to_pylist():
            if text:
                spoken_lines.append(text)
    print(f"  Loaded {len(spoken_lines):,} spoken French utterances (weighted {spoken_weight}x)", flush=True)
    for _ in range(spoken_weight):
        for line in spoken_lines:
            yield line

    print("Downloading French Wikipedia (wikimedia/wikipedia 20231101.fr)...", flush=True)
    wiki_parquet = hf_hub_download(
        repo_id="wikimedia/wikipedia",
        repo_type="dataset",
        filename="20231101.fr/train-00000-of-00017.parquet",
    )
    pf_wiki = pq.ParquetFile(wiki_parquet)
    total_chars = 0
    for batch in pf_wiki.iter_batches(columns=["text"], batch_size=1000):
        for text in batch.column(0).to_pylist():
            if not text:
                continue
            for para in text.split("\n"):
                para = para.strip()
                if len(para) < 25:
                    continue
                yield para
                total_chars += len(para)
                if total_chars >= wiki_chars:
                    print(f"  Streamed {total_chars:,} chars of French Wikipedia", flush=True)
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output path for lm_fr.bin")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--budget", type=int, default=4_000_000, help="Target n-gram count (~12 bytes/entry)")
    ap.add_argument("--wiki-chars", type=int, default=45_000_000)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    unigrams_path = Path(
        hf_hub_download(repo_id="guigssss/steno-fr-v1-large", filename="lm/unigrams.txt")
    )
    stoi, itos = load_vocab_from_unigrams(unigrams_path, max_vocab=64000)
    stream = build_stream(stream_hf_corpora(wiki_chars=args.wiki_chars), stoi, args.order)
    entries, unk = build_kneser_ney(stream, itos, order=args.order, budget=args.budget)
    del stream

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jnlm(entries, itos, unk, args.order, out_path)

    # Verify homophone disambiguation & French spoken numbers on standard French test pairs
    lm = VerifyBackoffLM(out_path)
    pairs = [
        (["il", "a", "pris", "le", "train"], ["il", "à", "pris", "le", "train"]),
        (["c'est", "un", "très", "bon", "projet"], ["s'est", "un", "très", "bon", "projet"]),
        (["il", "s'est", "passé", "quelque", "chose"], ["il", "c'est", "passé", "quelque", "chose"]),
        (["la", "ville", "où", "j'habite"], ["la", "ville", "ou", "j'habite"]),
        (["ils", "ont", "terminé", "le", "travail"], ["ils", "on", "terminer", "le", "travail"]),
        (["mille", "neuf", "cent", "soixante-trois"], ["mille", "neuf", "cent", "soixantetrois"]),
        (["ticket", "quatre", "cent", "douze"], ["ticket", "quatre", "sans", "douze"]),
    ]
    print("Verifying French homophone & number disambiguation with lm_fr.bin:")
    for good, bad in pairs:
        s_good = lm.sentence_score(good)
        s_bad = lm.sentence_score(bad)
        print(f"  {' '.join(good):35s} ({s_good:6.2f}) vs {' '.join(bad):35s} ({s_bad:6.2f}) -> Δ = {s_good - s_bad:+.2f}")
        if s_good <= s_bad:
            raise RuntimeError(f"LM failed disambiguation check: {good} <= {bad}")


if __name__ == "__main__":
    main()
