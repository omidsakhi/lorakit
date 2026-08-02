"""Load and sample prompts from a JSON prompt dataset.

Each entry is ``{"id": str, "pos": str, "neg": str, "seed": int}``.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

# Class nouns used to filter large prompt corpora by subject gender.
GENDER_CLASS_WORDS: dict[str, tuple[str, ...]] = {
    "female": ("woman", "girl", "bride"),
    "male": ("man", "boy", "groom"),
}


@dataclass(frozen=True)
class Prompt:
    id: str
    pos: str
    neg: str
    seed: int


def load_prompts(path: str | Path) -> list[Prompt]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"prompt_file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"prompt_file {path} must be a JSON array")
    prompts: list[Prompt] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id", i))
        pos = item.get("pos")
        neg = item.get("neg", "")
        raw_seed = item.get("seed")
        if not pos:
            continue
        if raw_seed is None:
            raise ValueError(f"Prompt {pid!r} is missing required field 'seed'")
        prompts.append(Prompt(id=pid, pos=pos, neg=neg or "", seed=int(raw_seed)))
    if not prompts:
        raise ValueError(f"No usable prompts found in {path}")
    return prompts


def sample_prompts(prompts: list[Prompt], num: int | None, seed: int = 0) -> list[Prompt]:
    """Deterministically sample ``num`` prompts, preserving each entry's pos/neg pair."""
    if num is None or num >= len(prompts):
        return list(prompts)
    if num <= 0:
        raise ValueError("num_prompts must be positive")
    rng = random.Random(seed)
    return rng.sample(prompts, num)


def normalize_gender(gender: str) -> str:
    key = gender.strip().lower()
    if key in {"female", "f", "woman", "women"}:
        return "female"
    if key in {"male", "m", "man", "men"}:
        return "male"
    raise ValueError(f"gender must be 'male' or 'female', got {gender!r}")


def gender_class_words(gender: str) -> tuple[str, ...]:
    return GENDER_CLASS_WORDS[normalize_gender(gender)]


def _gender_word_pattern(words: tuple[str, ...]) -> re.Pattern[str]:
    alt = "|".join(re.escape(word) for word in words)
    return re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)


def filter_prompts_by_gender(prompts: list[Prompt], gender: str) -> list[Prompt]:
    """Keep prompts whose positive text mentions a class noun for ``gender``."""
    pattern = _gender_word_pattern(gender_class_words(gender))
    return [prompt for prompt in prompts if pattern.search(prompt.pos)]


def inject_trigger(pos: str, trigger: str, class_word: str) -> str:
    """Insert a DreamBooth trigger before the class word (``of man`` -> ``of sks man``)."""
    if re.search(rf"\b{re.escape(trigger)}\s+{re.escape(class_word)}\b", pos):
        return pos
    return re.sub(rf"\bof {re.escape(class_word)}\b", f"of {trigger} {class_word}", pos, count=1)


def inject_subject_trigger(
    pos: str,
    *,
    trigger: str,
    class_word: str,
    gender: str,
) -> str:
    """Normalize gendered subject nouns and inject the DreamBooth trigger.

    Matches ``of [young|old] {woman|girl|bride|...}`` (depending on gender) and
    rewrites the first hit to ``of {trigger} {class_word}`` so a LoRA trained as
    ``sks woman`` still fires on girl/bride prompts (and likewise for male).
    """
    if re.search(rf"\b{re.escape(trigger)}\s+{re.escape(class_word)}\b", pos):
        return pos

    words = gender_class_words(gender)
    alt = "|".join(re.escape(word) for word in words)
    pattern = re.compile(
        rf"\bof\s+(?:(?:young|old)\s+)?(?:{alt})\b",
        re.IGNORECASE,
    )
    replaced, count = pattern.subn(f"of {trigger} {class_word}", pos, count=1)
    if count:
        return replaced

    # Fallback: bare class word with no ``of`` (rare), then classic inject.
    bare = _gender_word_pattern(words)
    replaced, count = bare.subn(f"{trigger} {class_word}", pos, count=1)
    if count:
        return replaced
    return inject_trigger(pos, trigger, class_word)


def load_training_sample_prompts(
    path: str | Path,
    *,
    trigger: str | None = None,
    class_word: str | None = None,
) -> tuple[list[str], list[str], list[int]]:
    """Load pos/neg/seed lists for lorakit training-time sampling."""
    prompts = load_prompts(path)
    pos: list[str] = []
    neg: list[str] = []
    seeds: list[int] = []
    for item in prompts:
        text = item.pos
        if trigger and class_word:
            text = inject_trigger(text, trigger, class_word)
        pos.append(text)
        neg.append(item.neg)
        seeds.append(item.seed)
    return pos, neg, seeds
