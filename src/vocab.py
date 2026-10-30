# ABOUTME: Loads a Hugging Face byte-level BPE vocabulary file and decodes each
# ABOUTME: token id into the literal text it represents, for constrained
# ABOUTME: decoding.

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _byte_to_unicode() -> dict[int, str]:
    """Return the GPT-2 style byte -> unicode-character mapping.

    This is the mapping used by byte-level BPE tokenizers (GPT-2, RoBERTa,
    Qwen, ...) to represent arbitrary bytes as printable unicode characters
    inside the vocabulary.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@lru_cache(maxsize=1)
def _unicode_to_byte() -> dict[str, int]:
    return {v: k for k, v in _byte_to_unicode().items()}


def decode_bpe_token(token: str) -> str:
    """Decode a single byte-level BPE vocabulary entry into real text.

    Tokens that are not valid byte-level BPE tokens (e.g. special tokens
    such as ``<|endoftext|>``) are returned unchanged; they will simply
    fail to match any
    of our grammars, which is the desired behaviour.
    """
    mapping = _unicode_to_byte()
    try:
        raw_bytes = bytes(mapping[ch] for ch in token)
    except KeyError:
        return token
    return raw_bytes.decode("utf-8", errors="replace")


def load_vocab(vocab_file: str | Path) -> dict[int, str]:
    """Load a ``vocab.json`` file and return ``{token_id: decoded_text}``.

    Args:
        vocab_file: path to a Hugging Face ``vocab.json`` file mapping token
            strings to their integer id.

    Returns:
        A dictionary mapping token id to the literal text that token produces
        when decoded.
    """
    path = Path(vocab_file)
    with path.open("r", encoding="utf-8") as handle:
        raw_vocab: dict[str, int] = json.load(handle)

    return {
        token_id: decode_bpe_token(token)
        for token, token_id in raw_vocab.items()
    }
