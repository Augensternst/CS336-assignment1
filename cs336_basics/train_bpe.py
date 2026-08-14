import os
from collections import defaultdict

import regex as re
from typing import BinaryIO
gpt2_pat= r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(gpt2_pat)

class TokenizerTrainer:
    def __init__(self,input_path:str,vocab_size:int,special_tokens:list[str],num_processes:int):
        """
        初始化参数：
        input_path ：str，指向包含 BPE 分词器训练数据的文本文件的路径。
        vocab_size：int，一个正整数，用于定义最终词汇表的最大尺寸（包括初始字节词汇表、通过合并产生的词汇项以及任何特殊标记）。
        special_tokens：list[str]，要添加到词汇表中的字符串列表。这些特殊标记在 BPE 训练过程中不会产生其他影响。
        num_processes:进程数量
        预计输出：
        vocab：dict[int, bytes] 分词器的词汇表，一个从整数（词汇表中的token ID）到字节（token字节）的映射。
        merges：list[tuple[bytes， bytes]] 由训练产生的 BPE 合并项。每个列表项是一个字节元组 (<token1>，<token2>）
        ，表示 <token1> 已与 <token2> 合并。这些合并项应按照创建顺序排序。
        """
        self.input_path=input_path
        self.vocab_size=vocab_size if vocab_size else 512
        self.special_tokens=special_tokens if special_tokens else []
        self.num_processes=num_processes
        # 初始化输出
        self.vocab={i:bytes([i])for i in range(256)}
        for tok in special_tokens:
            self.vocab[len(self.vocab)]=tok.encode("utf-8")
        self.merge=[]
    def train_bpe(self):
        """
        Train byte pair encoding
        """
        #pre-tokenization
        with open(self.input_path,'rb') as f:
            data=f.read()
        chunk_ids=self.process_chunk(data)
        ids:list[list[int]]=chunk_ids
        pair_to_indices,counts=self.get_pair_counts(ids)



    def process_chunk(self,data:bytes):
        text=data.decode("utf-8",errors="ignore")
        if self.special_tokens:
            pattern="|".join(re.escape(tok) for tok in self.special_tokens)
            documents=re.split(pattern,text)
        else:
            documents=[text]
        chunk_ids:list[list[int]]=[]
        for doc in documents:
            tokens=[match.group(0).encode("utf-8") for match in _COMPILED_PAT.finditer(doc)]
            chunk_ids.extend([list(token) for token in tokens])
        return chunk_ids
    @staticmethod
    def get_pair_counts(ids:list[list[int]]
    )->tuple[
    defaultdict[tuple[int, int], set],
    defaultdict[tuple[int, int], int]
    ]:
        pairs=defaultdict(set)
        counts=defaultdict(int)
        for i,token_ids in enumerate(ids):
            for pair in zip(token_ids,token_ids[1:]):
                pairs[pair].add(i)
                counts[pair]+=1
        return pairs,counts
















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

