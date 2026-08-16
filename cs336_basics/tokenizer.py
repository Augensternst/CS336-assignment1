import pickle
from typing import Iterable, Iterator

import regex as re

# 和训练阶段完全一样的预分词正则
PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = merges

        # 反查表：字节内容 -> token id
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        # 特殊 token 按长度从长到短排序，防止短的抢先匹配把长的切碎
        self.special_tokens = sorted(set(special_tokens or []), key=len, reverse=True)

        # 特殊 token 如果还不在词表里，补进去
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id

        # 用来切分特殊 token 的正则（没有特殊 token 就是 None）
        self.special_pattern = None
        if self.special_tokens:
            pattern_str = "|".join(re.escape(token) for token in self.special_tokens)
            self.special_pattern = re.compile(pattern_str)

        # pretoken -> 合并后的 token id 列表，避免同一个 pretoken 重复计算
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """把文本编码成 token id 列表。"""
        if self.special_pattern is None:
            return self._encode_without_special_tokens(text)

        ids: list[int] = []
        pos = 0
        for match in self.special_pattern.finditer(text):
            # 特殊 token 前面的普通文本，正常预分词+合并
            normal_text = text[pos:match.start()]
            if normal_text:
                ids.extend(self._encode_without_special_tokens(normal_text))

            # 特殊 token 本身，直接查表变成一个 id
            special_token_bytes = match.group().encode("utf-8")
            ids.append(self.token_to_id[special_token_bytes])

            pos = match.end()

        # 最后一个特殊 token 之后剩下的文本
        remaining_text = text[pos:]
        if remaining_text:
            ids.extend(self._encode_without_special_tokens(remaining_text))

        return ids

    def _encode_without_special_tokens(self, text: str) -> list[int]:
        """对不含特殊 token 的文本做预分词，再逐个 pretoken 合并。"""
        ids: list[int] = []
        for match in PRETOKENIZE_PATTERN.finditer(text):
            pretoken = match.group()
            ids.extend(self._encode_one_pretoken(pretoken))
        return ids

    def _encode_one_pretoken(self, pretoken: str) -> list[int]:
        """把一个 pretoken（比如 "the"）合并成 token id 列表。"""
        if pretoken in self._cache:
            return self._cache[pretoken]

        # 第一步：拆成单字节，每个字节先当成一个独立的 token
        parts: list[bytes] = [bytes([b]) for b in pretoken.encode("utf-8")]

        # 第二步：按训练时的合并顺序，一条一条尝试应用 merge 规则
        for merge_pair in self.merges:
            if len(parts) == 1:
                break  # 已经合并成一个 token 了，没必要再试后面的规则
            parts = self._apply_one_merge(parts, merge_pair)

        # 第三步：把每个字节片段查表转成 token id
        token_ids = [self.token_to_id[part] for part in parts]

        self._cache[pretoken] = token_ids
        return token_ids

    @staticmethod
    def _apply_one_merge(parts: list[bytes], merge_pair: tuple[bytes, bytes]) -> list[bytes]:
        """在 parts 里找到所有相邻出现的 merge_pair，合并成一个整体。"""
        new_parts: list[bytes] = []
        i = 0
        while i < len(parts):
            is_match = i < len(parts) - 1 and (parts[i], parts[i + 1]) == merge_pair
            if is_match:
                new_parts.append(parts[i] + parts[i + 1])
                i += 2
            else:
                new_parts.append(parts[i])
                i += 1
        return new_parts

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        逐块编码一个可迭代对象（比如打开的文件对象，按行迭代）。
        每次只处理一小块文本，不会把整个文件读进内存。
        """
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """把 token id 列表解码成文本。"""
        all_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        # errors="replace"：遇到无法解码的字节，用 U+FFFD 替代，不报错
        return all_bytes.decode("utf-8", errors="replace")