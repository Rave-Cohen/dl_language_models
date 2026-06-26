import json
import re


from itertools import pairwise


def get_stats(
    ids: list[int], counts: dict[tuple[int, int], int] | None = None, freq: int = 1
) -> dict[tuple[int, int], int]:
    counts = {} if counts is None else counts
    for pair in pairwise(ids):
        counts[pair] = counts.get(pair, 0) + freq
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    i = 0
    newids: list[int] = []
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(new_id)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class BPETokenizer:
    def __init__(
        self,
        split_pattern: str | None = None,
    ) -> None:
        self.compiled_pattern = re.compile(split_pattern) if split_pattern else None
        self.merges = {}
        self.vocab = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        text_chunks = re.findall(self.compiled_pattern, text) if self.compiled_pattern else [text]
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text: str) -> list[int]:
        # To make sure we don't incorrectly encode special tokens we fisrt
        # chunk the text into parts that make sure special tokens are encoded correctly
        if not self.special_tokens:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for chunk in special_chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx])
            else:
                raise ValueError(f"Invalid ID: {idx}")
        return (b"".join(part_bytes)).decode("utf-8", errors="replace")

    def load(self, input_file: str) -> None:
        with open(input_file, "r") as fd:
            bpe_data = json.load(fd)

        if "split pattern" in bpe_data:
            self.compiled_pattern = re.compile(bpe_data["split pattern"])
        self.special_tokens = bpe_data["special tokens"]
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        json_merges = bpe_data["merges"]
        for pair, idx in json_merges.items():
            p0, p1 = pair.split(" ")
            self.merges[(int(p0), int(p1))] = idx

        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # We build a mapping from the merges indicies to string/byte representations of texts
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # We finally add the special tokens
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
