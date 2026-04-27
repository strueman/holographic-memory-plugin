"""Fact splitting: decompose long composite facts into atomic statements.

Splits facts that contain multiple independent claims into separate atomic facts,
improving signal-to-noise ratio for FTS5 retrieval.

Example:
    Input: "Simon uses Debian Trixie stable. His setup: AMD RX 7900 XTX 24GB via Aoostar OCuLink dock. 64GB LPDDR5 shared pool."
    Output:
        - "Simon uses Debian Trixie stable."
        - "His setup includes an AMD RX 7900 XTX 24GB GPU via Aoostar OCuLink dock."
        - "64GB LPDDR5 shared pool."

Usage:
    from fact_splitter import split_fact
    result = split_fact("long composite fact content here")
"""

from __future__ import annotations

import re
from typing import List, Tuple


# Split patterns
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')
_COMMA_CLAUSE = re.compile(r',\s+(?=[A-Z])')  # comma before capitalized word
_SEMICOLON = re.compile(r';\s+')
_CONJUNCTION_SPLIT = re.compile(r'(?<!\s)(?:\s+,|\s+and\s+|\s+or\s+)\s+(?=[A-Z])')
_HARDCODED_THRESHOLD = 80  # chars — split facts longer than this


def split_fact(content: str, max_tokens: int = 40) -> List[str]:
    """Split a long fact into atomic statements.

    Returns a list of atomic statements. If the fact is short enough or
    cannot be further split, returns a single-element list with the original.

    Args:
        content: The fact content to split.
        max_tokens: Maximum tokens per split to prevent overly granular splits.

    Returns:
        List of atomic fact strings.
    """
    content = content.strip()

    # Don't split if already short
    if len(content) < _HARDCODED_THRESHOLD:
        return [content]

    # Try sentence-level splits first (recursive)
    sentences = _SENTENCE_SPLIT.split(content)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) >= 2:
        # Recursively split any parts that are still long
        result = []
        for s in sentences:
            if len(s) >= _HARDCODED_THRESHOLD:
                result.extend(split_fact(s, max_tokens))
            else:
                tokens = s.split()
                if 2 <= len(tokens) <= max_tokens and len(s) >= 8:
                    result.append(s)
                elif len(tokens) < 2 and len(result) > 0:
                    result[-1] = f"{result[-1]}, {s.strip()}"
                else:
                    result.append(s)
        if result:
            return result

    # Try comma-clause splits
    clauses = _COMMA_CLAUSE.split(content)
    if len(clauses) >= 2:
        valid = []
        for c in clauses:
            c = c.strip()
            if len(c) >= _HARDCODED_THRESHOLD:
                valid.extend(split_fact(c, max_tokens))
            elif 3 <= len(c.split()) <= max_tokens and len(c) >= 10:
                valid.append(c)
        if valid:
            return valid

    # Try conjunction-based splits
    parts = _CONJUNCTION_SPLIT.split(content)
    if len(parts) >= 2:
        valid = []
        for p in parts:
            p = p.strip()
            if len(p) >= _HARDCODED_THRESHOLD:
                valid.extend(split_fact(p, max_tokens))
            elif 3 <= len(p.split()) <= max_tokens and len(p) >= 10:
                valid.append(p)
        if valid:
            return valid

    # No meaningful split found — return original
    return [content]


def split_batch(facts: List[Tuple[int, str, dict]]) -> List[Tuple[int, str, dict]]:
    """Split a batch of facts into atomic statements.

    Args:
        facts: List of (fact_id, content, metadata_dict) tuples.

    Returns:
        List of (original_fact_id, atomic_content, metadata_dict) tuples.
        Split facts carry the original fact_id for parent tracking.
    """
    results = []
    for fid, content, meta in facts:
        splits = split_fact(content)
        for split in splits:
            results.append((fid, split, dict(meta)))  # shallow copy of metadata
    return results
