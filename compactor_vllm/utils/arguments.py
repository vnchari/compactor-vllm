import itertools
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
from compactor_vllm.compression import CompressionMethod
from compactor_vllm.compression.compression_config import BatchCompressionParams
from compactor_vllm.config.engine_config import LLMConfig
from compactor_vllm.utils.sequence import Sequence


@dataclass
class PrefillBatchArguments:
    B: int
    N: int
    do_compression: bool
    compression_method: CompressionMethod
    compression_chunk_size: int

    seq_ids: torch.Tensor

    input_ids: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int

    batch_tokens_to_retain: Optional[torch.Tensor]
    max_tokens_to_retain: Optional[int]
    protected_first: Optional[List[int]]
    protected_last: Optional[List[int]]

    PHI: Optional[torch.Tensor]

    # args needed for memory reservation
    context_lens: torch.Tensor
    max_new_tokens: torch.Tensor


class PackedTensorArguments:
    def __init__(
        self, rank: int, max_batched_tokens: int, config: LLMConfig, seed: int = 42
    ) -> None:
        hf_config = config.hf_config
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.max_num_batches = config.max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.num_kv_heads = hf_config.num_key_value_heads // dist.get_world_size()
        self.world_size = config.tensor_parallel_size
        self.head_dim = getattr(hf_config, "head_dim", None)
        self.sketch_dim = config.leverage_sketch_size
        self.model_dtype = hf_config.torch_dtype

        # i64 pack = [seq_ids (BMAX)] || [input_ids (NMAX)] || [positions (NMAX)] || max_new_tok (BMAX)
        self.i64_len_max = (
            self.max_num_batches + 2 * self.max_batched_tokens + self.max_num_batches
        )
        self.packed_context_i64 = torch.empty(
            self.i64_len_max, dtype=torch.int64, device=self.device
        )

        # i32 pack = [header (5)] || [cu_q (BMAX+1)] || [cu_k (BMAX+1)] || [retain (BMAX)] || [context_lens (BMAX)]
        #   || [protected_first_tokens (BMAX)] || [protected_last_tokens (BMAX)]
        self.i32_len_max = (
            5
            + (self.max_num_batches + 1)
            + (self.max_num_batches + 1)
            + self.max_num_batches
            + self.max_num_batches
            + self.max_num_batches
            + self.max_num_batches
        )
        self.packed_context_i32 = torch.empty(
            self.i32_len_max, dtype=torch.int32, device=self.device
        )

        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.PHI = torch.randn(
            (self.head_dim, self.sketch_dim),
            device=self.packed_context_i32.device,
            generator=self.generator,
        ).to(self.model_dtype) * (1 / math.sqrt(self.sketch_dim))

    def _master_build_prefill(
        self, seqs: List[Sequence], batch_compression_params: BatchCompressionParams
    ) -> PrefillBatchArguments:
        B = len(seqs)
        Ls = [x.prompt_len for x in seqs]

        N = sum(Ls)
        do_compression = any(x.compression_params.compression_ratio < 1.0 for x in seqs)
        do_compression = (
            do_compression
            and batch_compression_params.compression_method != CompressionMethod.NONE
        )
        pack_slices_64 = self.packed_i64_slices(B, N)
        pack_slices_32 = self.packed_i32_slices(B)

        # max_retain = max(retain)
        protected_first_list = [
            x.compression_params.protected_first_tokens for x in seqs
        ]
        protected_last_list = [x.compression_params.protected_last_tokens for x in seqs]
        retain = [
            max(
                int(
                    round(
                        x.compression_params.compression_ratio
                        * (L - s - e)
                        * self.num_kv_heads
                    )
                ),
                1,
            )
            for s, e, L, x in zip(protected_first_list, protected_last_list, Ls, seqs)
        ]
        retain = torch.tensor(retain, dtype=torch.int32, device="cpu", pin_memory=True)
        protected_first = torch.tensor(
            protected_first_list, dtype=torch.int32, device="cpu", pin_memory=True
        )
        protected_last = torch.tensor(
            protected_last_list, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.packed_context_i32[pack_slices_32["protected_first"]].copy_(
            protected_first, non_blocking=True
        )
        self.packed_context_i32[pack_slices_32["protected_last"]].copy_(
            protected_last, non_blocking=True
        )
        compression_chunk_size = (
            batch_compression_params.chunk_size
            if batch_compression_params.do_chunked_compression
            else -1
        )
        header_host = torch.tensor(
            [
                B,
                N,
                1 if do_compression else 0,
                batch_compression_params.compression_method.value,
                compression_chunk_size,
            ],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        self.packed_context_i32[pack_slices_32["retain"]].copy_(
            retain, non_blocking=True
        )
        self.packed_context_i32[pack_slices_32["header"]].copy_(
            header_host, non_blocking=True
        )
        max_seq_qk = max(Ls)

        cu = torch.tensor(
            list(itertools.accumulate(Ls, initial=0)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.packed_context_i32[pack_slices_32["cu_q"]].copy_(cu, non_blocking=True)
        self.packed_context_i32[pack_slices_32["cu_k"]].copy_(cu, non_blocking=True)
        self.packed_context_i32[pack_slices_32["context_lens"]].copy_(
            cu.diff(), non_blocking=True
        )

        seq_ids = torch.tensor(
            [x.seq_id for x in seqs], dtype=torch.int64, device="cpu", pin_memory=True
        )
        input_ids = torch.tensor(
            [tid for x in seqs for tid in x.prompt_token_ids],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self.packed_context_i64[pack_slices_64["seq_ids"]].copy_(
            seq_ids, non_blocking=True
        )
        self.packed_context_i64[pack_slices_64["input_ids"]].copy_(
            input_ids, non_blocking=True
        )

        positions = torch.cat(
            [
                torch.arange(L, dtype=torch.int64, device="cpu", pin_memory=True)
                for L in Ls
            ]
        )
        self.packed_context_i64[pack_slices_64["positions"]].copy_(
            positions, non_blocking=True
        )

        max_new_tokens = torch.tensor(
            [seq.sampling_params.max_new_tokens for seq in seqs],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self.packed_context_i64[pack_slices_64["max_new_tokens"]].copy_(
            max_new_tokens, non_blocking=True
        )
        max_retain = int(
            self.packed_context_i32[pack_slices_32["context_lens"]].max().item()
            * self.num_kv_heads
        )
        dist.broadcast(self.packed_context_i64, src=0)
        dist.broadcast(self.packed_context_i32, src=0)
        prefill_args = PrefillBatchArguments(
            B=B,
            N=N,
            do_compression=do_compression,
            compression_method=batch_compression_params.compression_method,
            compression_chunk_size=compression_chunk_size,
            seq_ids=self.packed_context_i64[pack_slices_64["seq_ids"]],
            input_ids=self.packed_context_i64[pack_slices_64["input_ids"]],
            positions=self.packed_context_i64[pack_slices_64["positions"]],
            cu_seqlens_q=self.packed_context_i32[pack_slices_32["cu_q"]],
            cu_seqlens_k=self.packed_context_i32[pack_slices_32["cu_k"]],
            max_seqlen_q=max_seq_qk,
            max_seqlen_k=max_seq_qk,
            batch_tokens_to_retain=self.packed_context_i32[pack_slices_32["retain"]],
            max_tokens_to_retain=max_retain,
            PHI=self.PHI,
            context_lens=self.packed_context_i32[pack_slices_32["context_lens"]],
            max_new_tokens=self.packed_context_i64[pack_slices_64["max_new_tokens"]],
            protected_first=protected_first_list,
            protected_last=protected_last_list,
        )
        return prefill_args

    def _peer_receive_prefill(self) -> PrefillBatchArguments:
        dist.broadcast(self.packed_context_i64, src=0)
        dist.broadcast(self.packed_context_i32, src=0)
        header = self.packed_context_i32[:8].tolist()
        B, N = int(header[0]), int(header[1])
        do_compression = bool(int(header[2]))
        compression_method = CompressionMethod(int(header[3]))
        compression_chunk_size = int(header[4])

        pack_slices_64 = self.packed_i64_slices(B, N)
        pack_slices_32 = self.packed_i32_slices(B)
        max_retain = int(
            self.packed_context_i32[pack_slices_32["context_lens"]].max().item()
            * self.num_kv_heads
        )
        prefill_args = PrefillBatchArguments(
            B=B,
            N=N,
            do_compression=do_compression,
            compression_method=compression_method,
            compression_chunk_size=compression_chunk_size,
            seq_ids=self.packed_context_i64[pack_slices_64["seq_ids"]],
            input_ids=self.packed_context_i64[pack_slices_64["input_ids"]],
            positions=self.packed_context_i64[pack_slices_64["positions"]],
            cu_seqlens_q=self.packed_context_i32[pack_slices_32["cu_q"]],
            cu_seqlens_k=self.packed_context_i32[pack_slices_32["cu_k"]],
            max_seqlen_q=int(self.packed_context_i32[pack_slices_32["cu_q"]].max()),
            max_seqlen_k=int(self.packed_context_i32[pack_slices_32["cu_k"]].max()),
            batch_tokens_to_retain=self.packed_context_i32[pack_slices_32["retain"]],
            max_tokens_to_retain=max_retain,
            PHI=self.PHI,
            context_lens=self.packed_context_i32[pack_slices_32["context_lens"]],
            max_new_tokens=self.packed_context_i64[pack_slices_64["max_new_tokens"]],
            protected_first=self.packed_context_i32[
                pack_slices_32["protected_first"]
            ].tolist(),
            protected_last=self.packed_context_i32[
                pack_slices_32["protected_last"]
            ].tolist(),
        )
        return prefill_args

    @torch.inference_mode()
    def build_prefill_args(
        self,
        seqs: Optional[List[Sequence]] = None,
        batch_compression_params: Optional[BatchCompressionParams] = None,
    ) -> PrefillBatchArguments:
        if self.rank == 0:
            return self._master_build_prefill(seqs, batch_compression_params)
        return self._peer_receive_prefill()

    def broadcast(self):
        if self.world_size > 1:
            return dist.broadcast(self.packed_context_i64, src=0)
        return None

    @staticmethod
    def packed_i64_slices(B: int, N: int):
        return {
            "seq_ids": slice(0, B),
            "input_ids": slice(B, B + N),
            "positions": slice(B + N, B + 2 * N),
            "max_new_tokens": slice(B + 2 * N, 2 * B + 2 * N),
        }

    @staticmethod
    def packed_i32_slices(B: int):
        h0, h1 = 0, 5
        q0 = h1
        q1 = q0 + (B + 1)
        k0 = q1
        k1 = k0 + (B + 1)
        r0 = k1
        r1 = r0 + B
        c0 = r1
        c1 = r1 + B

        pf0 = c1
        pf1 = c1 + B
        pl0 = pf1
        pl1 = pf1 + B
        return {
            "header": slice(h0, h1),
            "cu_q": slice(q0, q1),
            "cu_k": slice(k0, k1),
            "retain": slice(r0, r1),
            "context_lens": slice(c0, c1),
            "protected_first": slice(pf0, pf1),
            "protected_last": slice(pl0, pl1),
        }


@dataclass
class DecodeBatchOutput:
    output_tokens: Optional[torch.Tensor]
    output_seq_ids: Optional[torch.Tensor]


@dataclass
class DecodeBatchArguments:
    batch_mapping: Optional[torch.Tensor] = None
    token_ids: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    max_ctx_lens: Optional[torch.Tensor] = None
    seq_ids: Optional[torch.Tensor] = None
    temps: Optional[torch.Tensor] = None
    desired_batch_occupancy: int = -1

    def update(
        self,
        batch_mapping,
        token_ids,
        positions,
        max_ctx_lens,
        seq_ids,
        temps=None,
        desired_batch_occupancy: int = None,
    ):
        if self.batch_mapping is not None:
            self.batch_mapping = torch.cat([self.batch_mapping, batch_mapping], dim=0)
        else:
            self.batch_mapping = batch_mapping.clone()
        if self.token_ids is not None:
            self.token_ids = torch.cat([self.token_ids, token_ids], dim=0)
        else:
            self.token_ids = token_ids.clone()
        if self.positions is not None:
            self.positions = torch.cat([self.positions, positions], dim=0)
        else:
            self.positions = positions.clone()
        if self.max_ctx_lens is not None:
            self.max_ctx_lens = torch.cat([self.max_ctx_lens, max_ctx_lens], dim=0)
        else:
            self.max_ctx_lens = max_ctx_lens.clone()
        if self.seq_ids is not None:
            self.seq_ids = torch.cat([self.seq_ids, seq_ids], dim=0)
        else:
            self.seq_ids = seq_ids.clone()

        if self.temps is not None and temps is not None:
            self.temps = torch.cat([self.temps, temps], dim=0)
        elif temps is not None:
            self.temps = temps.clone()

        if desired_batch_occupancy is not None:
            self.desired_batch_occupancy = desired_batch_occupancy

        return self

    @property
    def empty(self):
        return self.seq_ids is None
