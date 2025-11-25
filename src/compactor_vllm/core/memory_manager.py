import logging
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist
from compactor_vllm.config.engine_config import LLMConfig
from compactor_vllm.kv_cache.page_table import KVAllocationStatus, PagedKVCache
from torch import nn

logger = logging.getLogger(__name__)


class KVCacheManager:
    def __init__(self, rank: int, config: LLMConfig):
        super().__init__()
        hf_config = config.hf_config
        self.rank = rank
        self.gpu_frac = config.gpu_memory_utilization
        self.page_size = config.kvcache_page_size
        self.world_size = config.tensor_parallel_size
        self.max_num_batches = config.max_num_seqs
        self.max_model_len = config.max_model_len
        self.num_layers = hf_config.num_hidden_layers
        self.model_dtype = hf_config.torch_dtype
        self.head_dim = getattr(hf_config, "head_dim", None)
        self.max_pages_per_batch = (
            self.max_model_len + self.page_size - 1
        ) // self.page_size
        self.num_kv_heads = hf_config.num_key_value_heads // dist.get_world_size()
        assert hf_config.num_key_value_heads % dist.get_world_size() == 0, (
            "world size needs to divide num_kv_heads"
        )

        self.num_pages = None
        self.paged_cache: Optional[PagedKVCache] = None
        self.max_batched_tokens = None

        self.seq_id_to_batch = {}

    def allocate_sequences(
        self, seq_ids: List[int], max_positions: List[int]
    ) -> (bool, Optional[torch.Tensor]):
        batch_mapping = []
        for seq_id, len_to_alloc in zip(seq_ids, max_positions):
            if seq_id not in self.seq_id_to_batch:
                batch_id = self.paged_cache.new_batch()
                if batch_id is None:
                    logger.warning("Failed to allocate batch!")
                    return False, None
                self.seq_id_to_batch[seq_id] = int(batch_id)
            batch_mapping.append(self.seq_id_to_batch[seq_id])
            if (
                alloc_status := self.paged_cache.reserve_tokens(
                    self.seq_id_to_batch[seq_id], len_to_alloc
                )
            ) != KVAllocationStatus.SUCCESS:
                logger.warning(f"Failed to allocate pages ({alloc_status})!")
                return False, None
        batch_mapping = torch.as_tensor(batch_mapping, dtype=torch.int32, device="cuda")
        return True, batch_mapping

    def free_sequences(self, seq_ids: Iterable[int]):
        for seq_id in seq_ids:
            global_batch_id = self.seq_id_to_batch.pop(seq_id, None)
            self.paged_cache.free_batch(global_batch_id)

    def init_cache(self, model: nn.Module):
        self.num_pages = self.get_num_pages(self.gpu_frac, self.max_pages_per_batch)
        self.paged_cache = PagedKVCache(
            num_layers=self.num_layers,
            H_kv=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            num_pages=int(self.num_pages),
            max_num_batches=self.max_num_batches,
            device=f"cuda:{self.rank}",
            dtype=self.model_dtype,
            max_logical_pages_per_head=int(self.max_pages_per_batch),
        )
        self._assign_cache_to_layers(model)

    def _assign_cache_to_layers(self, model) -> None:
        for layer_index, layer in enumerate(model.model.layers):
            attn = layer.self_attn.attn
            k, v, pt, bh = self.paged_cache.layer_slices(layer_index)
            attn.k_cache = k
            attn.v_cache = v
            attn.page_table = pt
            attn.bh_seq_lens = bh
            attn.page_size = self.page_size

    def get_num_pages(self, frac: float, n_logical_pages_max: int):
        free, total = torch.cuda.mem_get_info()
        used = total - free
        stats = torch.cuda.memory_stats()
        peak = int(stats["allocated_bytes.all.peak"])
        current = int(stats["allocated_bytes.all.current"])
        bytes_for_kv_budget = int(total * frac * 0.9) - used - peak + current

        if bytes_for_kv_budget <= 0:
            raise RuntimeError(
                f"Insufficient memory for KV cache."
                f"Try increasing gpu_memory_utilization (currently {frac:.2f})."
            )
        # page_table[L, B, H_kv, N_LOGICAL_PAGES_MAX] + bh_seq_lens[L, B, H_kv]
        int32_sz = torch.empty((), dtype=torch.int32).element_size()  # 4
        page_table_bytes_per_layer = (
            self.max_num_batches
            * self.num_kv_heads
            * n_logical_pages_max
            * int32_sz  # page_table
            + self.max_num_batches * self.num_kv_heads * int32_sz
        )
        total_page_table_bytes = self.num_layers * page_table_bytes_per_layer
        kv_bytes_net = bytes_for_kv_budget - total_page_table_bytes
        if kv_bytes_net <= 0:
            raise RuntimeError(
                "page-table footprint exceeds KV cache budget. "
                f"reduce max_num_seqs ({self.max_num_batches}) "
                f"or increase kv_cache_mem_fraction (currently {frac:.2f})."
            )
        dtype_sz = torch.empty((), dtype=self.model_dtype).element_size()
        bytes_per_page_across_layers = self.num_layers * (
            2 * self.page_size * self.head_dim * dtype_sz
        )
        return max(1, kv_bytes_net // bytes_per_page_across_layers)

    def estimate_max_batched_tokens(
        self,
        warmup_tokens: int,
        bytes_used_before_warmup: int,
        bytes_peak_after_warmup: int,
    ) -> int:
        """
        Estimate the max total number of tokens that can be processed concurrently
        without OOM.
        """
        assert warmup_tokens > 0, "warmup_tokens must be > 0"
        # activation bytes per token
        warmup_delta = max(
            0, int(bytes_peak_after_warmup) - int(bytes_used_before_warmup)
        )
        bytes_per_token = max(1, (warmup_delta + warmup_tokens - 1) // warmup_tokens)

        free, total = torch.cuda.mem_get_info()
        target = int(total * self.gpu_frac)
        used_now = int(total - free)
        # reserve headroom equal to the gap between peak and current allocations seen so far
        stats = torch.cuda.memory_stats()
        peak_cur = int(stats.get("allocated_bytes.all.peak", 0))
        cur_now = int(stats.get("allocated_bytes.all.current", 0))
        cushion = max(0, peak_cur - cur_now)

        activation_budget = int(max(0, target - used_now - cushion) * 0.95)
        max_tokens_per_batch = activation_budget // bytes_per_token
        max_tokens_in_cache = (self.num_pages * self.page_size) // self.num_kv_heads
        # round to lower multiple of page size
        max_tokens_per_batch = (max_tokens_per_batch // self.page_size) * self.page_size
        max_tokens_in_cache = (max_tokens_in_cache // self.page_size) * self.page_size
        self.max_batched_tokens = min(max_tokens_in_cache, max_tokens_per_batch)
        return self.max_batched_tokens

    @property
    def num_free_batches(self) -> int:
        return len(self.paged_cache.free_batches)

    @property
    def num_free_pages(self) -> int:
        return min(len(fp) for fp in self.paged_cache.free_pages)

    def reclaim_pages(
        self,
        seq_ids_to_reclaim: Iterable[int],
        future_reserved_buffer: List[int] | torch.Tensor,
    ) -> int:
        approximate_bytes_freed = 0
        for i, seq_id in enumerate(seq_ids_to_reclaim):
            batch_idx = self.seq_id_to_batch[seq_id]
            approximate_bytes_freed += self.paged_cache.reclaim_pages(
                batch_idx, future_reserved_buffer[i]
            )
        return approximate_bytes_freed
