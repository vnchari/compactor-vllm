import time
from typing import Iterable, List

from compactor_vllm.core.memory_manager import KVCacheManager
from compactor_vllm.utils.sequence import Sequence, SequenceStatus
from tqdm import tqdm


def cdiv(a, b):
    """ceiling division"""
    return (a + b - 1) // b


class Scheduler:
    """
    Simple sequence scheduler for prefill + decode with a paged KV cache.
    The scheduler tracks three disjoint sets of sequence IDs:

      * ``pending_sequence_ids`` – sequences that have not yet been started.
      * ``active_sequence_ids`` – sequences currently running.
      * ``finished_sequence_ids`` – sequences that have generated all tokens.

    At prefill time, :meth:`get_prefill_batch` selects a subset of pending
    sequences that can fit into the available KV cache and per-step token
    budget, given the constraints from the associated :class:`KVCacheManager`.

    The class also handles basic bookkeeping of sequence statuses.

    Args:
        :param all_sequences:
            Iterable of :class:`Sequence` objects to be scheduled. Each
            sequence must have a unique ``seq_id``.
        :param kv_manager:
            A :class:`KVCacheManager` instance that this scheduler will use
            to determine whether additional batches can be scheduled.
        :param use_tqdm:
            If True, two progress bars are created:
              * "Started Batches" – increments when a sequence moves from
                pending to running.
              * "Finished Batches" – increments when a sequence finishes.
    """

    def __init__(
        self,
        all_sequences: Iterable[Sequence],
        kv_manager: KVCacheManager,
        *,
        use_tqdm=False,
    ):
        self.allseq_mapping: dict[int, Sequence] = {s.seq_id: s for s in all_sequences}
        self.pending_sequence_ids: set[int] = set([s.seq_id for s in all_sequences])
        self.active_sequence_ids: set[int] = set()
        self.finished_sequence_ids: set[int] = set()
        self.manager = kv_manager
        self.use_tqdm = use_tqdm
        self.start_time = time.perf_counter()
        self.total_tokens_generated = 0
        self.total_tokens_input = 0
        if use_tqdm:
            self.pbar = tqdm(
                total=len(self.pending_sequence_ids),
                desc="Completed Batches",
            )

    def get_prefill_batch(self) -> List[Sequence]:
        """
        Select a batch of pending sequences to prefill under KV/memory constraints.

        The selection is greedy over ``pending_sequence_ids`` in iteration order.
        A sequence is added to the batch if:

          * The sum of its prompt length and the total prompt tokens selected so
            far does not exceed ``manager.max_batched_tokens``, and
          * There is at least one free KV "batch slot" left
            (``manager.num_free_batches``), and
          * The total number of KV pages required by the sequence's prompt +
            max_new_tokens does not exceed the remaining free pages.
        Returns:
            :return List[Sequence]:
                The list of :class:`Sequence` objects chosen for prefill in
                this step. The caller is responsible for marking them as
                active via :meth:`add_running_sequence_ids`.
        """
        total_tok, sequences = 0, []
        num_free_batches, num_free_pages = (
            self.manager.num_free_batches,
            self.manager.num_free_pages,
        )
        for seq_id in self.pending_sequence_ids:
            seq = self.allseq_mapping[seq_id]
            prompt_length = seq.prompt_len
            pages_needed = (
                cdiv(
                    prompt_length + seq.sampling_params.max_new_tokens,
                    self.manager.page_size,
                )
                * self.manager.num_kv_heads
            )
            if (
                prompt_length + total_tok <= self.manager.max_batched_tokens
                and num_free_batches > 0
                and pages_needed < num_free_pages
            ):
                sequences.append(seq)
                total_tok += prompt_length
                num_free_pages -= pages_needed
                num_free_batches -= 1
        return sequences

    def is_finished(self) -> bool:
        """
        Check whether all sequences have completed.
        """
        return (
            len(self.pending_sequence_ids) == 0 and len(self.active_sequence_ids) == 0
        )

    def any_pending_sequences(self) -> bool:
        """
        Check whether any sequences are still pending (not yet started).
        """
        return len(self.pending_sequence_ids) != 0

    def add_running_sequence_ids(
        self, active_sequence_ids: Iterable[int], *, update_status: bool = False
    ):
        """
        Mark a set of sequences as active / running. This moves sequence IDs
        from ``pending_sequence_ids`` into ``active_sequence_ids``. Optionally,
        it also updates the per-sequence status and progress bar.

        Args:
            :param active_sequence_ids:
                Iterable of sequence IDs that have been scheduled for prefill
                or decode and should now be considered running.
            :param update_status:
                If True, set each corresponding :class:`Sequence`'s
                ``status = SequenceStatus.RUNNING`` and increment the
                "Started Batches" progress bar if ``use_tqdm`` is enabled.
        """
        self.active_sequence_ids.update(active_sequence_ids)
        self.pending_sequence_ids.difference_update(self.active_sequence_ids)
        if update_status:
            for seq_id in active_sequence_ids:
                self.allseq_mapping[seq_id].status = SequenceStatus.RUNNING
                self.total_tokens_input += self.allseq_mapping[seq_id].prompt_len

    def get_finished_sequence_ids_from_unfinished(
        self, unfinished_sequence_ids: Iterable[int]
    ) -> set[int]:
        """
        Infer which active sequences have finished given the
        unfinished set (for decode steps where the caller knows
        which sequences are still generating but not necessarily
        which have just completed).
        Args:
            :param unfinished_sequence_ids:
                Iterable of sequence IDs that are still running
        Returns:
            :return set[int]:
                The inferred set of sequence IDs that transitioned from active
                to finished.
        """
        return self.active_sequence_ids.difference(unfinished_sequence_ids)

    def record_finished_sequence_ids(
        self, finished_sequence_ids: Iterable[int], *, update_status: bool = False
    ):
        """
        Record that a set of sequences has finished generation.

        This moves IDs from ``active_sequence_ids`` into
        ``finished_sequence_ids``.

        Args:
            :param finished_sequence_ids:
                Iterable of sequence IDs that have completed generation and
                no longer require KV cache.
            :param update_status:
                If True, set each corresponding :class:`Sequence`'s
                ``status = SequenceStatus.FINISHED``
        """
        self.active_sequence_ids.difference_update(finished_sequence_ids)
        self.finished_sequence_ids.update(finished_sequence_ids)
        if update_status:
            for seq_id in finished_sequence_ids:
                self.allseq_mapping[seq_id].status = SequenceStatus.FINISHED
                self.pbar.update(1)

    def update_sequences(self, tokens: Iterable[int], seq_ids: Iterable[int]):
        """
        Append newly generated tokens to their corresponding sequences.
        Args:
            :param tokens:
                Iterable of generated token IDs, one per sequence.
            :param seq_ids:
                Iterable of sequence IDs aligned with ``tokens``.
        """
        cur_time = time.perf_counter()
        for tok, seq_id in zip(tokens, seq_ids):
            self.allseq_mapping[seq_id].add_new_token(tok)
            self.total_tokens_generated += 1
        self.pbar.set_description(
            f"Throughput: {(self.total_tokens_generated + self.total_tokens_input) / (cur_time - self.start_time):.2f} tok/s"
        )

    def close(self):
        if self.use_tqdm:
            self.pbar.close()
