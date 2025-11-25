from compactor_vllm.compression.common import (
    BaseCompressionMethod,
    NoCompression,
)
from compactor_vllm.compression.compactor import CompactorCompression
from compactor_vllm.compression.compression_config import (
    BatchCompressionParams,
    CompressionMethod,
    SequenceCompressionParams,
)
from compactor_vllm.compression.snapkv import SnapKVCompression

COMPRESSION_REGISTRY: dict[CompressionMethod, type[BaseCompressionMethod]] = {
    CompressionMethod.COMPACTOR: CompactorCompression,
    CompressionMethod.SNAPKV: SnapKVCompression,
    CompressionMethod.NONE: NoCompression,
}


def apply_prerope_compression(q, k, v, context):
    method = context.compression_context.compression_method
    return COMPRESSION_REGISTRY[method].pre_rope_scoring(q, k, v, context=context)


def apply_postrope_compression(q, k, v, prerope_scores, context):
    method = context.compression_context.compression_method
    return COMPRESSION_REGISTRY[method].post_rope_scoring(
        q, k, v, prerope_scores, context=context
    )


__all__ = [
    "apply_prerope_compression",
    "apply_postrope_compression",
    "CompressionMethod",
    "BatchCompressionParams",
    "SequenceCompressionParams",
    "COMPRESSION_REGISTRY"
]
