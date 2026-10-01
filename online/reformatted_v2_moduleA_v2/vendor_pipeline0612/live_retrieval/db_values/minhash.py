from __future__ import annotations

import logging

try:
    from datasketch import MinHash
except ImportError:  # pragma: no cover - depends on local experiment env
    MinHash = None


def _create_minhash(signature_size: int, string: str, n_gram: int):
    """Create a MinHash signature using the same n-gram logic as pipeline_0612."""
    if MinHash is None:
        raise ImportError("datasketch is not installed; MinHash value retrieval is unavailable")

    m = MinHash(num_perm=signature_size)
    text = str(string)
    if len(text) < n_gram:
        m.update(text.encode("utf8"))
        return m

    for gram in (text[i : i + n_gram] for i in range(len(text) - n_gram + 1)):
        m.update(gram.encode("utf8"))
    return m
