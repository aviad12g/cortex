"""
Synthetic long-context curricula for Stage A1 memory evaluations.

This module constructs four task families:
    1. Key-value binding with distractors.
    2. Copy-and-reverse with intervening noise.
    3. N-back probes on symbol streams.
    4. Long addition with carry propagation.

Each task exposes deterministic generation given a RNG seed and gap length G,
ensuring reproducibility across training and evaluation.
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

VOCAB_SYMBOLS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VOCAB_DIGITS = list("0123456789")
NOISE_TOKENS = list("abcdefg hijklmnop qrstuv wxyz ,.;:!?".split())


def _random_word(rng: random.Random, length: int = 4) -> str:
    return "".join(rng.choices(VOCAB_SYMBOLS, k=length))


def _random_digits(rng: random.Random, length: int) -> str:
    return "".join(rng.choices(VOCAB_DIGITS, k=length))


def _noise_block(rng: random.Random, length: int) -> str:
    words = rng.choices(NOISE_TOKENS, k=max(1, length // 3))
    return " ".join(words)


@dataclass
class TaskSample:
    sample_id: str
    text: str
    answer: str
    task: str
    gap: int
    meta: Dict[str, str]

    def to_json(self) -> str:
        payload = dataclasses.asdict(self)
        payload["meta"] = json.dumps(self.meta)
        return json.dumps(payload)


class BaseGenerator:
    """Base generator providing helper utilities."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)

    def _format(self, segments: Iterable[str]) -> str:
        return " ".join(segments)

    def generate(self, gap: int, sample_id: str) -> TaskSample:
        raise NotImplementedError


class KeyValueBindingGenerator(BaseGenerator):
    """Generate key/value associative retrieval sequences."""

    def __init__(self, seed: int, num_pairs: int = 6, key_len: int = 3, value_len: int = 4):
        super().__init__(seed)
        self.num_pairs = num_pairs
        self.key_len = key_len
        self.value_len = value_len
        self.key_pool = [f"K{idx}" for idx in range(1, 51)]

    def generate(self, gap: int, sample_id: str) -> TaskSample:
        pairs = []
        keys = self.rng.sample(self.key_pool, k=self.num_pairs)
        for key in keys:
            value = _random_word(self.rng, length=self.value_len)
            pairs.append((key, value))

        target_key, target_value = self.rng.choice(pairs)
        header_segments = [f"{k}->{v};" for k, v in pairs]

        noise = _noise_block(self.rng, gap)
        query = f"ASK {target_key}?"
        answer = target_value

        text = self._format(["HEADER:"] + header_segments + ["DISTRACTOR:", noise, "QUERY:", query, "ANSWER:"])
        meta = {"target_key": target_key, "target_value": target_value, "pairs": json.dumps(dict(pairs))}
        return TaskSample(sample_id=sample_id, text=text, answer=answer, task="kv", gap=gap, meta=meta)


class CopyReverseGenerator(BaseGenerator):
    """Generate sequences requiring copy+reverse retrieval."""

    def __init__(self, seed: int, token_len: int = 12):
        super().__init__(seed)
        self.token_len = token_len

    def generate(self, gap: int, sample_id: str) -> TaskSample:
        digits = _random_digits(self.rng, self.token_len)
        start_marker = "<SEQ>"
        end_marker = "</SEQ>"
        reverse_request = f"REVERSE {start_marker}"
        reversed_digits = digits[::-1]
        noise = _noise_block(self.rng, gap)
        text = self._format(
            [
                "PROMPT:",
                start_marker,
                " ".join(digits),
                end_marker,
                "DISTRACTOR:",
                noise,
                "QUERY:",
                reverse_request,
                "ANSWER:",
            ]
        )
        meta = {"sequence": digits}
        return TaskSample(
            sample_id=sample_id,
            text=text,
            answer=" ".join(reversed_digits),
            task="copy_reverse",
            gap=gap,
            meta=meta,
        )


class NBackGenerator(BaseGenerator):
    """N-back challenge using limited alphabet tokens."""

    def __init__(self, seed: int, alphabet: Optional[List[str]] = None, stream_len: int = 256):
        super().__init__(seed)
        self.alphabet = alphabet or list("ABCDE")
        self.stream_len = stream_len

    def generate(self, gap: int, sample_id: str) -> TaskSample:
        n = max(1, gap // 512)
        stream = self.rng.choices(self.alphabet, k=self.stream_len)
        probe_idx = self.rng.randrange(n, self.stream_len)
        probe_token = stream[probe_idx]
        answer = "YES" if stream[probe_idx - n] == probe_token else "NO"

        tokens = []
        for idx, token in enumerate(stream):
            if idx == probe_idx:
                tokens.append(f"[{token}]")
            else:
                tokens.append(token)
        noise = _noise_block(self.rng, gap)
        text = self._format(
            [
                "STREAM:",
                " ".join(tokens),
                "DISTRACTOR:",
                noise,
                "QUERY:",
                f"NBACK {n} {probe_token}",
                "ANSWER:",
            ]
        )
        meta = {"n": n, "probe_index": probe_idx}
        return TaskSample(sample_id=sample_id, text=text, answer=answer, task="nback", gap=gap, meta=meta)


class LongAdditionGenerator(BaseGenerator):
    """Long addition with carries through noise."""

    def __init__(self, seed: int, num_digits: int = 32):
        super().__init__(seed)
        self.num_digits = num_digits

    def generate(self, gap: int, sample_id: str) -> TaskSample:
        a = _random_digits(self.rng, self.num_digits)
        b = _random_digits(self.rng, self.num_digits)
        sum_int = int(a) + int(b)
        answer = str(sum_int)
        noise = _noise_block(self.rng, gap)
        text = self._format(
            [
                "CALC:",
                "+",
                a,
                b,
                "DISTRACTOR:",
                noise,
                "QUERY:",
                "SUM?",
                "ANSWER:",
            ]
        )
        meta = {"left": a, "right": b}
        return TaskSample(sample_id=sample_id, text=text, answer=answer, task="addition", gap=gap, meta=meta)


GENERATOR_REGISTRY = {
    "kv": KeyValueBindingGenerator,
    "copy_reverse": CopyReverseGenerator,
    "nback": NBackGenerator,
    "addition": LongAdditionGenerator,
}


def build_dataset(task: str, gap: int, num_samples: int, seed: int) -> List[TaskSample]:
    """Construct a deterministic dataset for the given task and gap."""
    if task not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown task {task}")
    generator_cls = GENERATOR_REGISTRY[task]
    generator = generator_cls(seed=seed)
    samples = []
    for idx in range(num_samples):
        sample_id = f"{task}_{gap}_{seed}_{idx:08d}"
        samples.append(generator.generate(gap=gap, sample_id=sample_id))
    return samples


def sample_to_training_pair(sample: TaskSample, tokenizer, max_length: int) -> Dict[str, List[int]]:
    """
    Convert a TaskSample into model inputs using a tokenizer.

    Returns:
        dict with input_ids, labels, metadata.
    """
    prompt = f"{sample.text} "
    target = sample.answer
    encoded_prompt = tokenizer(prompt, add_special_tokens=False)
    encoded_target = tokenizer(target, add_special_tokens=False)
    input_ids = encoded_prompt["input_ids"] + encoded_target["input_ids"]
    labels = [-100] * len(encoded_prompt["input_ids"]) + encoded_target["input_ids"]
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "task": sample.task,
        "gap": sample.gap,
        "sample_id": sample.sample_id,
        "answer_text": sample.answer,
        "prompt_text": sample.text,
        "meta": sample.meta,
    }


def generate_seed_splits(task: str, gaps: Iterable[int], seeds: Iterable[int], num_train: int, num_eval: int) -> Dict[str, Dict[int, Dict[int, List[TaskSample]]]]:
    """
    Build datasets partitioned by split -> gap -> seed -> samples.
    """
    splits: Dict[str, Dict[int, Dict[int, List[TaskSample]]]] = {"train": {}, "dev": {}, "test": {}}
    for split, count in (("train", num_train), ("dev", num_eval), ("test", num_eval)):
        for gap in gaps:
            gap_table: Dict[int, List[TaskSample]] = {}
            for seed in seeds:
                generator_seed = seed + hash((task, gap, split)) % 1_000_000
                gap_table[seed] = build_dataset(task, gap, count, generator_seed)
            splits[split][gap] = gap_table
    return splits


def iter_jsonl(samples: Iterable[TaskSample]) -> Iterator[str]:
    for sample in samples:
        record = {
            "task": sample.task,
            "gap": sample.gap,
            "text": sample.text,
            "answer": sample.answer,
            "meta": sample.meta,
        }
        yield json.dumps(record)
