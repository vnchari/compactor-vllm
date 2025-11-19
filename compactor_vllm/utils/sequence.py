from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import List

from compactor_vllm.compression.compression_config import SequenceCompressionParams
from compactor_vllm.config.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    """
    Represents a single user request / sequence being generated.
    """

    _counter = count()

    prompt_token_ids: List[int]
    completion_token_ids: List[int] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    compression_params: SequenceCompressionParams = field(
        default_factory=SequenceCompressionParams
    )
    status: SequenceStatus = SequenceStatus.WAITING

    seq_id: int = field(default_factory=lambda: next(Sequence._counter), init=False)
    num_tokens_processed: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.completion_token_ids)

    def add_new_token(self, token_id: int) -> None:
        if len(self.completion_token_ids) == 0:
            self.num_tokens_processed += self.num_prompt_tokens
        self.completion_token_ids.append(token_id)
        self.num_tokens_processed += 1

    def tokens_to_retain_per_layer(self, num_kv_heads: int) -> int:
        n = int(
            self.compression_params.compression_ratio
            * self.num_prompt_tokens
            * num_kv_heads
        )
        return max(1, n)

    def __getstate__(self):
        return dict(
            prompt_token_ids=list(self.prompt_token_ids),
            completion_token_ids=list(self.completion_token_ids),
            sampling_params=self.sampling_params,
            compression_params=self.compression_params,
            status=self.status,
            seq_id=self.seq_id,
            num_tokens_processed=self.num_tokens_processed,
        )

    def __setstate__(self, state):
        self.prompt_token_ids = list(state["prompt_token_ids"])
        self.completion_token_ids = list(state["completion_token_ids"])
        self.sampling_params = state["sampling_params"]
        self.compression_params = state["compression_params"]
        self.status = state["status"]
        self.seq_id = state["seq_id"]
        self.num_tokens_processed = state["num_tokens_processed"]

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def completion_len(self) -> int:
        return len(self.completion_token_ids)
