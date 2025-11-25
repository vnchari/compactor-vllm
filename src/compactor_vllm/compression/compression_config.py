import logging
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    COMPACTOR = auto()
    SNAPKV = auto()
    NONE = auto()


# class CachingPolicy(Enum):
#     CACHE_PROMPT = auto()
#     DONT_CACHE = auto()


# class CompressionType(Enum):
#     QUERY_AWARE = auto()
#     QUERY_AGNOSTIC = auto()


@dataclass
class SequenceCompressionParams:
    compression_ratio: float = 1.0
    protected_first_tokens: int = 16
    protected_last_tokens: int = 64


@dataclass
class BatchCompressionParams:
    # compression_type: CompressionType = CompressionType.QUERY_AGNOSTIC
    compression_method: CompressionMethod = CompressionMethod.COMPACTOR

    do_chunked_compression: bool = True
    chunk_size: int = 512

    def __post_init__(self):
        if self.compression_method == CompressionMethod.SNAPKV:
            self.do_chunked_compression = False
            logger.warning(
                "CompressionMethod.SNAPKV is not compatible with chunked compression. Disabling it."
            )
