import triton.language as tl

RESERVED_BATCH = 0
TRITON_RESERVED_BATCH = tl.constexpr(RESERVED_BATCH)
