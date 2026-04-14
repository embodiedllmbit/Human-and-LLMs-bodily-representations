#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for extracting stable final answers from model outputs."""

from __future__ import annotations

import re
from typing import Dict, Optional


def remove_think_sections(text: str) -> str:
    """Remove common thinking sections to avoid matching option letters in reasoning."""
    cleaned = text or ""
    patterns = [
        r"<think>.*?</think>",
        r"<\|begin_of_think\|>.*?<\|end_of_think\|>",
        r"\[thinking\].*?\[/thinking\]",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned


def strip_input_echo(decoded_output: str, decoded_input: str) -> str:
    """Remove echoed input prompt from decoded full output when present."""
    output_text = (decoded_output or "").strip()
    input_text = (decoded_input or "").strip()
    if not output_text:
        return ""
    if input_text:
        index = output_text.find(input_text)
        if index != -1:
            output_text = output_text[index + len(input_text):]
    return output_text.strip()


def clean_response_text(text: str) -> str:
    """Normalize leading assistant prefixes and trim whitespace."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"^\s*(assistant|助手|assistant_response)\s*[:：]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _extract_letter_from_fragment(fragment: str) -> Optional[str]:
    text = re.sub(r"<\|[^>]+\|>", " ", fragment or "")
    text = text.strip()
    if not text:
        return None

    line_matches = list(re.finditer(r"(?im)^\s*([A-Da-d])\s*[\.\)]?\s*$", text))
    if line_matches:
        return line_matches[-1].group(1).upper()

    tail_match = re.search(r"(?:^|[\s:：>\]\)\}])([A-Da-d])\s*[\.\)]?\s*$", text)
    if tail_match:
        return tail_match.group(1).upper()

    return None


def extract_final_answer(text: str) -> tuple[Optional[str], str]:
    """
    Extract final answer letter from text.
    Returns: (answer, method)
    """
    raw = clean_response_text(text or "")
    if not raw:
        return None, "empty"

    normalized = remove_think_sections(raw)

    tag_matches = list(
        re.finditer(
            r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if tag_matches:
        letter = _extract_letter_from_fragment(tag_matches[-1].group(1))
        if letter is not None:
            return letter, "answer_tag"

    boxed_matches = list(
        re.finditer(
            r"<\|begin_of_box\|>\s*([A-Da-d])\s*<\|end_of_box\|>",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if boxed_matches:
        return boxed_matches[-1].group(1).upper(), "boxed_token"

    latex_boxed = list(
        re.finditer(
            r"\\boxed\s*\{\s*([A-Da-d])\s*\}",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if latex_boxed:
        return latex_boxed[-1].group(1).upper(), "latex_boxed"

    parse_text = re.sub(r"<\|[^>]+\|>", " ", normalized)

    trailing_line_matches = list(re.finditer(r"(?im)^\s*([A-Da-d])\s*[\.\)]?\s*$", parse_text))
    if trailing_line_matches:
        return trailing_line_matches[-1].group(1).upper(), "trailing_line"

    if len(parse_text) <= 500:
        option_prefix_matches = list(
            re.finditer(r"(?im)^\s*([A-Da-d])\s*[:：\-]\s*\S+", parse_text)
        )
        if option_prefix_matches:
            return option_prefix_matches[-1].group(1).upper(), "option_prefix"

    phrase_matches = list(
        re.finditer(
            (
                r"(?:final\s+answer|answer\s+is|my\s+answer\s+is|"
                r"i\s+choose|i\s+pick|i\s+select)\s*[:：]?\s*([A-Da-d])\b"
            ),
            parse_text[-260:],
            flags=re.IGNORECASE,
        )
    )
    if phrase_matches:
        return phrase_matches[-1].group(1).upper(), "answer_phrase"

    trailing_match = re.search(r"(?:^|[\s:：>\]\)\}])([A-Da-d])\s*[\.\)]?\s*$", parse_text)
    if trailing_match:
        return trailing_match.group(1).upper(), "trailing_char"

    return None, "not_found"


def build_answer_payload(primary_text: str, *fallback_texts: str) -> Dict[str, Optional[str]]:
    """
    Build stable answer fields for JSON output.

    Returns keys:
    - response
    - final_answer
    - answer_parse_method
    - answer_parse_status
    """
    candidates = [primary_text, *fallback_texts]
    for idx, candidate in enumerate(candidates):
        answer, method = extract_final_answer(candidate or "")
        if answer is not None:
            if idx > 0:
                method = f"{method}_fallback_{idx}"
            return {
                "response": answer,
                "final_answer": answer,
                "answer_parse_method": method,
                "answer_parse_status": "ok",
            }

    return {
        "response": "?",
        "final_answer": None,
        "answer_parse_method": "not_found",
        "answer_parse_status": "not_found",
    }
