import os
from collections import defaultdict
from typing import BinaryIO

import regex as re

# GPT-2 使用的预分词正则表达式：把文本切成单词、数字、标点、空白等片段
GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PATTERN = re.compile(GPT2_PRETOKENIZE_PATTERN)


class TokenizerTrainer:
    """训练一个字节级别的 BPE（Byte Pair Encoding）分词器。"""

    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 1):
        """
        初始化参数：
        input_path：str，指向包含 BPE 分词器训练数据的文本文件的路径。
        vocab_size：int，一个正整数，用于定义最终词汇表的最大尺寸（包括初始字节词汇表、通过合并产生的词汇项以及任何特殊标记）。
        special_tokens：list[str]，要添加到词汇表中的字符串列表。这些特殊标记在 BPE 训练过程中不会参与合并。
        num_processes：预留的并行进程数（当前实现未使用并行，仅保存供以后扩展）。

        训练完成后：
        self.vocab：dict[int, bytes]，分词器的词汇表，从 token id 映射到 token 字节。
        self.merges：list[tuple[bytes, bytes]]，按创建顺序排列的 BPE 合并记录。
        """
        self.input_path = input_path
        self.vocab_size = vocab_size if vocab_size else 512
        self.special_tokens = special_tokens if special_tokens else []
        self.num_processes = num_processes

        # 初始词汇表：256 个单字节 token，再加上特殊 token
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for token in self.special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")
        self.merges: list[tuple[bytes, bytes]] = []

    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """运行 BPE 训练，返回 (vocab, merges)。"""
        with open(self.input_path, "rb") as f:
            raw_bytes = f.read()

        # pretokens[i] 是第 i 个预分词单元的字节 id 序列，例如 b"the" -> [116, 104, 101]
        pretokens: list[list[int]] = self._pretokenize(raw_bytes)
        pair_to_pretoken_indices, pair_counts = self._count_pairs(pretokens)

        num_merges_needed = self.vocab_size - len(self.vocab)
        for _ in range(num_merges_needed):
            if not pair_counts:
                break

            best_pair = self._pick_best_pair(pair_counts)
            new_token_id = len(self.vocab)
            new_token_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_token_id] = new_token_bytes
            self.merges.append((self.vocab[best_pair[0]], self.vocab[best_pair[1]]))

            # 只需要更新包含 best_pair 的那些预分词单元，避免每次都全量重新扫描
            affected_indices = pair_to_pretoken_indices[best_pair].copy()
            for idx in affected_indices:
                token_ids = pretokens[idx]
                if len(token_ids) < 2:
                    continue

                # 1. 先把这个预分词单元旧的相邻 pair 计数减掉
                for pair in zip(token_ids, token_ids[1:]):
                    pair_counts[pair] -= 1
                    pair_to_pretoken_indices[pair].discard(idx)
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]
                        del pair_to_pretoken_indices[pair]

                # 2. 把 best_pair 合并成 new_token_id，得到新的 token id 序列
                merged_token_ids = _merge_pair(token_ids, best_pair, new_token_id)

                # 3. 再把合并后新的相邻 pair 计数加回去
                for pair in zip(merged_token_ids, merged_token_ids[1:]):
                    pair_counts[pair] += 1
                    pair_to_pretoken_indices[pair].add(idx)

                pretokens[idx] = merged_token_ids

        return self.vocab, self.merges

    def _pick_best_pair(self, pair_counts: dict[tuple[int, int], int]) -> tuple[int, int]:
        """选出出现频率最高的 pair；频率相同时取字节字典序更大的一个（与 GPT-2 的 tie-break 规则一致）。"""

        def rank(pair: tuple[int, int]) -> tuple[int, tuple[bytes, bytes]]:
            return pair_counts[pair], (self.vocab[pair[0]], self.vocab[pair[1]])

        return max(pair_counts, key=rank)

    def _pretokenize(self, raw_bytes: bytes) -> list[list[int]]:
        """先按特殊 token 切分文本，再用 GPT-2 正则做预分词，返回每个预分词单元的字节 id 列表。"""
        text = raw_bytes.decode("utf-8", errors="ignore")
        if self.special_tokens:
            split_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            segments = re.split(split_pattern, text)
        else:
            segments = [text]

        pretokens: list[list[int]] = []
        for segment in segments:
            for match in _COMPILED_PATTERN.finditer(segment):
                pretokens.append(list(match.group(0).encode("utf-8")))
        return pretokens

    @staticmethod
    def _count_pairs(
        pretokens: list[list[int]],
    ) -> tuple[defaultdict[tuple[int, int], set], defaultdict[tuple[int, int], int]]:
        """统计所有相邻 pair 出现的次数，以及每个 pair 出现在哪些预分词单元（按下标）里。"""
        pair_to_pretoken_indices: defaultdict[tuple[int, int], set] = defaultdict(set)
        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        for idx, token_ids in enumerate(pretokens):
            for pair in zip(token_ids, token_ids[1:]):
                pair_to_pretoken_indices[pair].add(idx)
                pair_counts[pair] += 1
        return pair_to_pretoken_indices, pair_counts


def _merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把 token_ids 中所有相邻出现的 pair 替换成 new_id。"""
    merged_ids: list[int] = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(token_ids[i])
            i += 1
    return merged_ids


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
