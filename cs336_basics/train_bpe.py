import gc
import os
from collections import defaultdict
from multiprocessing import get_context
from typing import BinaryIO

import regex as re

# GPT-2 使用的预分词正则表达式：把文本切成单词、数字、标点、空白等片段
GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PATTERN = re.compile(GPT2_PRETOKENIZE_PATTERN)


class TokenizerTrainer:
    """训练一个字节级别的 BPE（Byte Pair Encoding）分词器（多进程优化版）。"""

    def __init__(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 1,
        chunk_multiplier: int = 8,
    ):
        """
        input_path：训练数据文本文件路径。
        vocab_size：最终词表大小上限（含 256 字节 + 特殊 token + 合并产生的 token）。
        special_tokens：不参与合并的特殊 token 列表。
        num_processes：预分词阶段使用的进程数。
        chunk_multiplier：把文件切成 num_processes * chunk_multiplier 份，
            份数比进程数多是为了让每块更小、内存峰值更低，
            Pool 通过 imap_unordered 自动调度，同一时间只有 num_processes 个任务在跑。
        """
        self.input_path = input_path
        self.vocab_size = vocab_size if vocab_size else 512
        self.special_tokens = special_tokens if special_tokens else []
        self.num_processes = max(1, num_processes)
        self.chunk_multiplier = max(1, chunk_multiplier)

        # 初始词汇表：256 个单字节 token，再加上特殊 token
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for token in self.special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")
        self.merges: list[tuple[bytes, bytes]] = []

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """运行 BPE 训练，返回 (vocab, merges)。"""
        pretoken_counts = self._pretokenize_parallel()

        # 拆成两个并行数组：pretokens[i] 是 token id 列表，counts[i] 是它出现的次数
        pretokens: list[list[int]] = []
        counts: list[int] = []
        for token_ids, cnt in pretoken_counts.items():
            pretokens.append(list(token_ids))
            counts.append(cnt)
        del pretoken_counts
        gc.collect()

        pair_to_pretoken_indices, pair_counts = self._count_pairs(pretokens, counts)

        num_merges_needed = self.vocab_size - len(self.vocab)

        gc.disable()
        try:
            for _ in range(num_merges_needed):
                if not pair_counts:
                    break

                best_pair = self._pick_best_pair(pair_counts)
                new_token_id = len(self.vocab)
                new_token_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
                self.vocab[new_token_id] = new_token_bytes
                self.merges.append((self.vocab[best_pair[0]], self.vocab[best_pair[1]]))

                # 只需要更新包含 best_pair 的那些预分词单元
                affected_indices = pair_to_pretoken_indices[best_pair].copy()
                for idx in affected_indices:
                    token_ids = pretokens[idx]
                    if len(token_ids) < 2:
                        continue
                    pre_count = counts[idx]

                    # 1. 先把这个预分词单元旧的相邻 pair 计数按权重减掉
                    for pair in zip(token_ids, token_ids[1:]):
                        pair_counts[pair] -= pre_count
                        if pair_counts[pair] <= 0:
                            del pair_counts[pair]
                            pair_to_pretoken_indices[pair].discard(idx)
                            if not pair_to_pretoken_indices[pair]:
                                del pair_to_pretoken_indices[pair]
                        else:
                            pair_to_pretoken_indices[pair].discard(idx)

                    # 2. 合并 best_pair
                    merged_token_ids = _merge_pair(token_ids, best_pair, new_token_id)

                    # 3. 把合并后新的相邻 pair 计数按权重加回去
                    for pair in zip(merged_token_ids, merged_token_ids[1:]):
                        pair_counts[pair] += pre_count
                        pair_to_pretoken_indices[pair].add(idx)

                    pretokens[idx] = merged_token_ids
        finally:
            gc.enable()
            gc.collect()

        return self.vocab, self.merges

    # ------------------------------------------------------------------ #
    # 并行预分词
    # ------------------------------------------------------------------ #
    def _pretokenize_parallel(self) -> dict[tuple[int, ...], int]:
        """多进程预分词，返回 {pretoken 的字节id元组: 出现次数}。"""
        num_chunks = self.num_processes * self.chunk_multiplier
        split_bytes = b"<|endoftext|>"
        if self.special_tokens:
            # 优先用 <|endoftext|> 做切块对齐（最常见的文档分隔符）；
            # 没有的话退化为用列表里第一个特殊 token
            if "<|endoftext|>" in self.special_tokens:
                split_bytes = "<|endoftext|>".encode("utf-8")
            else:
                split_bytes = self.special_tokens[0].encode("utf-8")
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, split_bytes)

        tasks = [
            (self.input_path, start, end, self.special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
            if end > start
        ]

        merged_counts: dict[tuple[int, ...], int] = defaultdict(int)

        ctx = get_context("forkserver")
        with ctx.Pool(processes=self.num_processes) as pool:
            # imap_unordered：谁先算完就先合并谁的结果，及时释放内存，
            # 不会像 pool.map 那样把全部 worker 结果一次性攒在内存里
            for worker_result in pool.imap_unordered(_pretokenize_chunk, tasks):
                for token_ids, cnt in worker_result.items():
                    merged_counts[token_ids] += cnt

        return merged_counts

    # ------------------------------------------------------------------ #
    # 合并相关工具方法
    # ------------------------------------------------------------------ #
    def _pick_best_pair(self, pair_counts: dict[tuple[int, int], int]) -> tuple[int, int]:
        """选出频率最高的 pair；频率相同时取字节字典序更大的一个。"""

        def rank(pair: tuple[int, int]) -> tuple[int, tuple[bytes, bytes]]:
            return pair_counts[pair], (self.vocab[pair[0]], self.vocab[pair[1]])

        return max(pair_counts, key=rank)

    @staticmethod
    def _count_pairs(
        pretokens: list[list[int]],
        counts: list[int],
    ) -> tuple[defaultdict[tuple[int, int], set], defaultdict[tuple[int, int], int]]:
        """统计所有相邻 pair 的加权出现次数，以及每个 pair 出现在哪些预分词单元（按下标）里。"""
        pair_to_pretoken_indices: defaultdict[tuple[int, int], set] = defaultdict(set)
        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        for idx, token_ids in enumerate(pretokens):
            cnt = counts[idx]
            for pair in zip(token_ids, token_ids[1:]):
                pair_to_pretoken_indices[pair].add(idx)
                pair_counts[pair] += cnt
        return pair_to_pretoken_indices, pair_counts


# ---------------------------------------------------------------------- #
# 顶层函数（供 multiprocessing pickling 使用，必须是模块级函数，不能是实例方法）
# ---------------------------------------------------------------------- #
def _pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> dict[tuple[int, ...], int]:
    """子进程内部：读取自己负责的文件区间，预分词并计数，返回 {pretoken字节id元组: 次数}。"""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    text = chunk_bytes.decode("utf-8", errors="ignore")

    if special_tokens:
        split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(split_pattern, text)
    else:
        segments = [text]

    counts: dict[tuple[int, ...], int] = defaultdict(int)
    for segment in segments:
        for match in _COMPILED_PATTERN.finditer(segment):
            token_ids = tuple(match.group(0).encode("utf-8"))
            counts[token_ids] += 1
    return counts


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

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = max(1, file_size // desired_num_chunks)

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


if __name__ == "__main__":
    # 简单用法示例
    import sys
    import time

    if len(sys.argv) < 2:
        print("用法: python bpe_trainer.py <input_path> [vocab_size] [num_processes]")
        sys.exit(1)

    path = sys.argv[1]
    vsize = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count() or 1

    trainer = TokenizerTrainer(
        input_path=path,
        vocab_size=vsize,
        special_tokens=["<|endoftext|>"],
        num_processes=nproc,
    )
    t0 = time.time()
    vocab, merges = trainer.train_bpe()
    print(f"训练完成，用时 {time.time() - t0:.1f}s，词表大小 {len(vocab)}，merges 数 {len(merges)}")