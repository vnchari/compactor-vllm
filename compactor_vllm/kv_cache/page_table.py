import heapq
import logging
from enum import Enum, auto
from typing import List, Optional, Union

import torch
from compactor_vllm.config.constants import RESERVED_BATCH
from compactor_vllm.kv_cache.write_page_table import scatter_to_page_table

logger = logging.getLogger(__name__)


def cdiv(a, b):
    return (a + b - 1) // b


def next_multiple(a, b):
    return cdiv(a, b) * b


class KVAllocationStatus(Enum):
    EXCEEDS_MAX_SEQUENCE_LENGTH = auto()
    EXCEEDS_CURRENTLY_AVAILABLE_PAGES = auto()
    EXCEEDS_MAX_NUM_BATCHES = auto()
    SUCCESS = auto()


class PagedKVCache(torch.nn.Module):
    """
    Global paged KV cache.
    This module manages:
      * A global K/V backing buffer for all layers:
          ``kv_cache[2, num_layers, n_pages * page_size, head_dim]``,
        where the first dimension indexes K vs V.
      * A per-layer page table:
          ``page_table[num_layers, max_num_seqs, H_kv, max_pages_per_head]``,
        mapping logical (batch, kv-head, logical_page) to a physical page ID
        in the global K/V buffer.
      * Per-layer, per-(batch, kv-head) logical sequence lengths
        ``bh_seq_lens[num_layers, max_num_seqs, H_kv]`` (in tokens), and
        the number of allocated pages ``bh_num_pages`` for each (layer, batch,
        head).
      * A page allocator implemented as a min-heap of free physical pages
        per layer, plus free batch indices.
    Pages are of fixed size ``page_size`` tokens.
    Args:
        :param num_layers:
            Number of transformer layers that will use this cache.
        :param max_logical_pages_per_head:
            Maximum number of logical pages that can be assigned to a single
            (batch, kv-head) pair.
        :param num_pages:
            Total number of physical pages available in the global cache per
            layer. The global K/V buffers are of length
            ``num_pages * page_size`` along the token dimension.
        :param  page_size:
            Number of tokens stored per page.
        :param H_kv:
            Number of KV heads per layer.
        :param  head_dim:
            Head dimension for K/V.
        :param max_num_batches:
            Maximum number of concurrent batches / sequences supported. One
            batch index is reserved for internal use (``RESERVED_BATCH``).
        :param dtype:
            Data type of K/V entries (e.g. ``torch.float16`` or ``torch.bfloat16``).
        :param device:
            Device on which to allocate the cache (string, torch.device, or
            int; defaults to ``"cuda"``).
    """

    def __init__(
        self,
        num_layers: int,
        max_logical_pages_per_head: int,
        num_pages: int,
        page_size: int,  # tokens per page
        H_kv: int,
        head_dim: int,
        max_num_batches: int,
        dtype: torch.dtype,
        device: Union[str, torch.device, int] = "cuda",
    ):
        super().__init__()
        self.n_pages = num_pages
        self.num_layers = num_layers
        self.page_size: int = int(page_size)
        self.H_kv = int(H_kv)
        self.max_pages_per_head = max_logical_pages_per_head
        max_num_batches += 1
        self.max_num_batches = max_num_batches
        self.head_dim = head_dim
        cache_shape = (2, num_layers, num_pages * page_size, head_dim)
        self.kv_cache = torch.empty(cache_shape, dtype=dtype, device=device)

        self.page_table = torch.empty(
            (num_layers, max_num_batches, H_kv, self.max_pages_per_head),
            device=device,
            dtype=torch.int32,
        )

        # Per-(batch, head) logical seq length (tokens)
        self.bh_seq_lens = torch.zeros(
            (num_layers, max_num_batches, H_kv), device=device, dtype=torch.int32
        )
        # self._bh_seq_lens_cpu_buffer = torch.zeros((num_layers, H_kv), device="cpu", dtype=torch.int32)
        self.bh_num_pages = torch.zeros(
            (num_layers, max_num_batches, H_kv), device=device, dtype=torch.int32
        )

        # Page allocator (min-heap of free physical pages)
        self.free_pages: List[List[int]] = [
            list(range(num_pages)) for _ in range(num_layers)
        ]
        for free_pages in self.free_pages:
            heapq.heapify(free_pages)
        # batch zero is reserved
        self.free_batches: List[int] = list(reversed(range(max_num_batches)))
        self.free_batches.remove(RESERVED_BATCH)
        # Record of physical page ids owned by a batch (for freeing)
        self.pages_indices_per_batch: List[List[set[int]]] = [
            [set() for _ in range(num_layers)] for _ in range(max_num_batches)
        ]

    def new_batch(self) -> Optional[int]:
        """
        Reserve a new batch slot.
        A batch slot corresponds to a row in ``bh_seq_lens`` /
        ``bh_num_pages`` and a slice in ``page_table`` for all layers and KV
        heads. This method checks whether a free batch index is available, and
        whether each layer has at least ``H_kv`` free pages remaining.
        If both checks pass, it returns a batch index and removes it from
        ``free_batches``. Otherwise, it returns ``None``.

        Returns:
            :return Optional[int]:
                Newly reserved batch index, or ``None`` if no capacity is
                available.
        """
        if self.free_batches and all([self.H_kv <= len(fp) for fp in self.free_pages]):
            return self.free_batches.pop()
        return None

    def reserve_tokens(self, batch_index: int, add_tokens: int) -> KVAllocationStatus:
        """
        Ensure enough pages are allocated to handle ``add_tokens`` new tokens.
        Args:
            :param batch_index:
                Batch index to reserve space for.
            :param  add_tokens:
                Number of additional tokens to reserve capacity for.
                All heads in this batch and all layers reserve
                the same number of extra tokens.
        Returns:
            :return bool:
                ``True`` if the reservation succeeds; ``False`` otherwise .
        """
        cur_bh_lens = self.bh_seq_lens[:, batch_index]  # [L, H]
        curr_pages = self.bh_num_pages[:, batch_index]  # [L, H]
        curr_cap_tokens = curr_pages * self.page_size  # [L, H]
        need_tokens = cur_bh_lens + add_tokens  # [L, H]
        if (need_tokens <= curr_cap_tokens).all():
            return KVAllocationStatus.SUCCESS
        missing_tokens = need_tokens - curr_cap_tokens
        add_pages = cdiv(missing_tokens, self.page_size)
        new_total_pages = curr_pages + add_pages
        if (new_total_pages > self.max_pages_per_head).any():
            return KVAllocationStatus.EXCEEDS_MAX_SEQUENCE_LENGTH
        # CPU work
        pages_per_layer_cpu = add_pages.sum(dim=-1).tolist()
        new_phys_pages = []
        for layer_index in range(self.num_layers):
            if pages_per_layer_cpu[layer_index] > len(self.free_pages[layer_index]):
                return KVAllocationStatus.EXCEEDS_CURRENTLY_AVAILABLE_PAGES
        for layer_index in range(self.num_layers):
            this_layer_pages = [
                heapq.heappop(self.free_pages[layer_index])
                for _ in range(pages_per_layer_cpu[layer_index])
            ]
            self.pages_indices_per_batch[batch_index][layer_index] |= set(
                this_layer_pages
            )
            new_phys_pages.extend(this_layer_pages)

        new_phys_pages = torch.tensor(new_phys_pages, dtype=torch.int32, device="cuda")

        scatter_to_page_table(
            add_pages=add_pages,
            new_phys_pages=new_phys_pages,
            curr_pages=curr_pages,
            page_table=self.page_table[:, batch_index],
            max_pages_per_head=self.max_pages_per_head,
        )

        self.bh_num_pages[:, batch_index, :] = new_total_pages.to(
            self.bh_num_pages.dtype
        )
        return KVAllocationStatus.SUCCESS

    def reclaim_pages(
        self,
        batch_index: int,
        future_reserve_tokens: int = 0,
    ):
        """
        Reclaim unused pages for a single batch index. This shrinks the KV
        allocation for the batch down to the minimum number of pages needed
        to hold the current (plus optional future) sequence length.

        Args:
            :param batch_index:
                Batch index whose pages should be compacted.
            :param future_reserve_tokens:
                Optional number of extra tokens to keep capacity for, beyond
                the current sequence length. This can reduce churn when
                sequences are expected to grow slightly in the near future.

        Returns:
            :return int:
                Approximate number of bytes freed across both K and V.
        """
        device = self.bh_seq_lens.device
        L, B, H = self.bh_seq_lens.shape
        assert 0 <= batch_index < B

        seq = self.bh_seq_lens[:, batch_index, :] + future_reserve_tokens  # [L, H]
        alloc = self.bh_num_pages[:, batch_index, :]  # [L, H]
        pt = self.page_table[:, batch_index, :, :].reshape(-1)  # [L, H, P]

        # Compute used pages: ceil_div(seq, page_size), clamped into [0, alloc]
        used_pages = cdiv(seq, self.page_size)
        used_pages = torch.minimum(used_pages, alloc)

        # page indices [0..P-1], broadcasted over [L, H, P]
        p = torch.arange(
            self.max_pages_per_head, device=device, dtype=torch.int32
        ).view(1, 1, self.max_pages_per_head)

        # allocated: p < alloc
        alloc_mask = p < alloc.unsqueeze(-1)  # [L, H, P]
        # to free: allocated and p in [used_pages, alloc)
        free_mask = alloc_mask & (p >= used_pages.unsqueeze(-1))
        free_mask_flat = free_mask.view(-1)  # [L*H*P]
        if not free_mask_flat.any():
            return 0

        idx = free_mask_flat.nonzero(as_tuple=False).squeeze(
            -1
        )  # indices of freed slots

        # Freed physical page ids
        freed_pages = pt[idx]
        # Compute layer index for each freed slot:
        # layout is [L, H, P] → flat index = ((l * H) + h) * P + p
        freed_layers = (idx // (H * self.max_pages_per_head)).to(torch.int32)
        freed_pages = freed_pages.tolist()
        layer_mapping = freed_layers.tolist()
        self.bh_num_pages[:, batch_index, :] = used_pages
        for page, layer in zip(freed_pages, layer_mapping):
            self.pages_indices_per_batch[batch_index][layer].remove(page)
            heapq.heappush(self.free_pages[layer], page)
        approximate_bytes_freed = (
            len(freed_pages)
            * (self.page_size * self.head_dim * self.kv_cache.element_size())
            * 2
        )  # multiply for two for K + V
        return approximate_bytes_freed

    def _free_batch_layer(self, layer_index: int, batch_index: int) -> None:
        """
        Free all pages belonging to batch_index and reset its metadata.
        """
        # Return pages to the global heap
        for phys in self.pages_indices_per_batch[batch_index][layer_index]:
            heapq.heappush(self.free_pages[layer_index], int(phys))

        self.pages_indices_per_batch[batch_index][layer_index] = set()

    def free_batch(self, batch_index: int) -> None:
        """
        Free all resources associated with a batch index.
        Args:
            :param batch_index:
                Batch index to release. Must have been previously allocated
                via :meth:`new_batch`.
        """
        for layer in range(self.num_layers):
            self._free_batch_layer(layer, batch_index)
        self.bh_seq_lens[:, batch_index].zero_()
        self.bh_num_pages[:, batch_index].zero_()
        self.free_batches.append(batch_index)

    def layer_slices(self, layer: int):
        """
        Return layer-local views needed by the attention module.

        For a given ``layer`` index, this method returns the slices of the
        global K/V cache, page table, and per-(batch, head) sequence lengths
        corresponding to that layer.
        Args:
            :param layer:
                Layer index ``l`` in ``[0, num_layers)``.

        Returns:
            :return Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                ``(k, v, pt, bh)`` as described above.
        """
        assert 0 <= layer < self.num_layers
        k = self.kv_cache[0, layer]
        v = self.kv_cache[1, layer]
        pt = self.page_table[layer]
        bh = self.bh_seq_lens[layer]
        return k, v, pt, bh
