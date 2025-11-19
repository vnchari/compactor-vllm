import atexit
import logging
from typing import List, Optional

import torch
import torch.distributed as dist
from compactor_vllm.attention.sparse_decode_kernel import num_splits_heuristic
from compactor_vllm.compression.compression_config import BatchCompressionParams
from compactor_vllm.config.constants import RESERVED_BATCH
from compactor_vllm.config.engine_config import AttentionBackend, LLMConfig
from compactor_vllm.core.memory_manager import KVCacheManager
from compactor_vllm.core.scheduler import Scheduler
from compactor_vllm.layers.sampler import Sampler
from compactor_vllm.models import MODEL_REGISTRY
from compactor_vllm.utils.arguments import (
    DecodeBatchArguments,
    DecodeBatchOutput,
    PackedTensorArguments,
    PrefillBatchArguments,
)
from compactor_vllm.utils.context import CompressionContext, reset_context, set_context
from compactor_vllm.utils.sequence import Sequence
from torch.multiprocessing import Event
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelRunner:
    """Per-rank execution loop. Manages model, sampler, KV cache, and warmup"""

    def __init__(
        self,
        config: LLMConfig,
        rank: int,
        batch_ready: Optional[Event] = None,
        peer_events: List[Event] = None,
    ):
        self.rank = rank
        self.config = config
        hf_config = config.hf_config
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.leverage_sketch_size = config.leverage_sketch_size
        self.show_progress_bar = config.show_progress_bar
        self.max_num_batches = config.max_num_seqs
        self.max_model_len = config.max_model_len
        self.num_layers = hf_config.num_hidden_layers
        self.model_dtype = hf_config.torch_dtype
        self.head_dim = getattr(hf_config, "head_dim", None)

        dist.init_process_group(
            "nccl",
            f"tcp://localhost:{config.nccl_port}",
            world_size=self.world_size,
            rank=rank,
            device_id=rank,
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        model_type = hf_config.model_type
        self.model = MODEL_REGISTRY[model_type](hf_config)
        self.model.load_model(
            config.path, use_tqdm=self.is_master and self.show_progress_bar
        )
        self.sampler = Sampler()

        pre_warmup_mem = torch.cuda.memory_stats().get("allocated_bytes.all.current", 0)
        self.warmup(
            num_warmup_tokens=self.max_model_len,
            attention_backend=AttentionBackend.FLASH_ATTENTION,
        )
        post_warmup_peak = torch.cuda.memory_stats().get("allocated_bytes.all.peak", 0)

        self.kv_manager = KVCacheManager(rank, config)
        self.kv_manager.init_cache(self.model)

        self.store_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        self.batch_ready = batch_ready
        self.peer_events = peer_events if peer_events is not None else []
        self.captured_graphs = {}
        self.min_captured_len = {}
        self.max_batched_tokens = self.kv_manager.estimate_max_batched_tokens(
            self.max_model_len, pre_warmup_mem, post_warmup_peak
        )
        if self.is_master:
            logger.info(f"Estimated max batched tokens of {self.max_batched_tokens}")
        if self.config.attention_backend == AttentionBackend.COMPACTOR_TRITON:
            self.warmup(
                num_warmup_tokens=self.max_model_len,
                attention_backend=AttentionBackend.COMPACTOR_TRITON,
            )

        if not self.enforce_eager:
            bs = [1 << i for i in range(self.max_num_batches.bit_length())]
            for bs in (
                tqdm(bs, desc="Capturing CUDA Graphs")
                if self.is_master and self.show_progress_bar
                else bs
            ):
                for seq_len in [1024, 4096, 8192, 16384]:
                    self.capture_cudagraph(bs, seq_len)

        self.packed_args = PackedTensorArguments(
            rank=self.rank,
            max_batched_tokens=self.max_batched_tokens,
            config=self.config,
        )
        atexit.register(self.exit)

    @torch.inference_mode()
    def warmup(self, num_warmup_tokens: int, attention_backend: AttentionBackend):
        if self.rank == 0:
            if attention_backend == AttentionBackend.COMPACTOR_TRITON:
                backend_name = "Compactor Triton"
            else:
                backend_name = "Flash"
            logger.info(f"Warming up with {backend_name} Attention Backend")
        device = torch.device(f"cuda:{self.rank}")
        input_ids = torch.tensor(
            [self.config.eos] * num_warmup_tokens, device=device, dtype=torch.int64
        )
        positions = torch.arange(num_warmup_tokens, device=device, dtype=torch.int64)
        cu_seqlens_q = torch.tensor(
            [0, num_warmup_tokens], device=device, dtype=torch.int32
        )
        cu_seqlens_k = torch.tensor(
            [0, num_warmup_tokens], device=device, dtype=torch.int32
        )
        if attention_backend == AttentionBackend.COMPACTOR_TRITON:
            success, batch_mapping = self.kv_manager.allocate_sequences(
                [-1], [num_warmup_tokens]
            )
            assert success
        else:
            batch_mapping = None
        set_context(
            is_prefill=True,
            do_compression=False,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=num_warmup_tokens,
            max_seqlen_k=num_warmup_tokens,
            batch_mapping=batch_mapping,
            attention_backend=attention_backend,
        )
        for _ in range(2):
            torch.cuda.reset_peak_memory_stats()
            self.model.compute_logits(self.model(input_ids, positions))
            dist.barrier()
            if attention_backend == AttentionBackend.COMPACTOR_TRITON:
                self.kv_manager.paged_cache.bh_seq_lens.index_fill_(
                    1, batch_mapping.to(torch.long), 0
                )
        reset_context()
        if attention_backend == AttentionBackend.COMPACTOR_TRITON:
            self.kv_manager.free_sequences([-1])

    def exit(self):
        try:
            del self.captured_graphs
        finally:
            dist.destroy_process_group()

    def loop(self):
        while True:
            if self.batch_ready.wait(1.0):
                self._process_batches_peer()

    @torch.inference_mode()
    def run_prefill(
        self, prefill_args: PrefillBatchArguments, batch_mapping: torch.Tensor
    ):
        assert prefill_args.B > 0 and prefill_args.N > 0
        max_bh_len = (
            self.kv_manager.paged_cache.bh_seq_lens.index_select(1, index=batch_mapping)
            .max()
            .item()
        )
        compression_context = CompressionContext(
            compression_method=prefill_args.compression_method,
            compression_chunk_size=prefill_args.compression_chunk_size,
            batch_tokens_to_retain=prefill_args.batch_tokens_to_retain,
            max_tokens_to_retain=prefill_args.max_tokens_to_retain,
            context_lens=prefill_args.context_lens.tolist(),
            PHI=prefill_args.PHI,
            protected_first_tokens=prefill_args.protected_first,
            protected_last_tokens=prefill_args.protected_last,
        )
        set_context(
            is_prefill=True,
            do_compression=prefill_args.do_compression,
            cu_seqlens_q=prefill_args.cu_seqlens_q,
            cu_seqlens_k=prefill_args.cu_seqlens_k,
            max_seqlen_q=prefill_args.max_seqlen_q,
            max_seqlen_k=prefill_args.max_seqlen_k,
            batch_mapping=batch_mapping,
            max_bh_len=max_bh_len,
            compression_context=compression_context,
            STORE_STREAM=self.store_stream,
            attention_backend=self.config.attention_backend,
        )
        logits = self.model.compute_logits(
            self.model(prefill_args.input_ids, prefill_args.positions)
        )
        reset_context()
        return logits

    def maybe_broadcast(self, tensor: torch.Tensor):
        if self.world_size > 1:
            return dist.broadcast(tensor, src=0)
        return None

    def maybe_release_peers(self, do_release=False):
        if self.world_size > 1:
            if self.is_master:
                if do_release:
                    for event in self.peer_events:
                        event.clear()
                dist.barrier()
            else:
                dist.barrier()

    @torch.inference_mode()
    def generate(
        self,
        all_sequences: List[Sequence],
        batch_compression_params: Optional[BatchCompressionParams] = None,
    ):
        assert self.is_master, "generate can only be called on the master process"
        for begin_execution_event in self.peer_events:
            begin_execution_event.set()
        if batch_compression_params is None:
            batch_compression_params = BatchCompressionParams()
        self._process_batches_master(all_sequences, batch_compression_params)

    @property
    def is_master(self):
        return self.rank == 0

    @torch.inference_mode()
    def _process_batches_master(
        self,
        all_sequences: List[Sequence],
        batch_compression_params: BatchCompressionParams,
    ):
        assert self.is_master
        compression_details = f"Applying Compression Method: {batch_compression_params.compression_method}"
        if any(seq.compression_params.compression_ratio < 1.0 for seq in all_sequences):
            logger.info(compression_details)
        scheduler = Scheduler(
            all_sequences=all_sequences,
            kv_manager=self.kv_manager,
            use_tqdm=self.show_progress_bar,
        )
        decode_batch = DecodeBatchArguments()
        decode_flags = torch.empty(2, dtype=torch.int32, device="cuda")
        while not scheduler.is_finished():
            sequences = scheduler.get_prefill_batch()
            seq_ids_cpu = [seq.seq_id for seq in sequences]
            scheduler.add_running_sequence_ids(seq_ids_cpu, update_status=True)
            temps = torch.tensor(
                [s.sampling_params.temperature for s in sequences],
                dtype=torch.float32,
                pin_memory=True,
            ).cuda(non_blocking=True)
            prefill_arguments = self.packed_args.build_prefill_args(
                sequences, batch_compression_params=batch_compression_params
            )
            max_ctx_lens = (
                prefill_arguments.max_new_tokens + prefill_arguments.context_lens
            )

            success, batch_mapping = self.kv_manager.allocate_sequences(
                seq_ids_cpu, max_ctx_lens.tolist()
            )
            assert success, "failed to allocate pages for sequences"

            logits = self.run_prefill(prefill_arguments, batch_mapping)
            positions, batch_mapping = prefill_arguments.context_lens, batch_mapping
            token_ids = self.sampler(logits, temps)
            # TODO: synchronize page counts accross dist
            if self.world_size == 1:
                self.kv_manager.reclaim_pages(
                    seq_ids_cpu, prefill_arguments.max_new_tokens
                )
                # with logging_redirect_tqdm():
                #     logger.info(
                #         f"Reclaimed {reclaimed_bytes / 1e6:.2f} MB from the KV cache"
                #     )
            occupancy = (
                2 * (len(sequences) // 3) if scheduler.any_pending_sequences() else -1
            )
            run_decode = len(scheduler.get_prefill_batch()) == 0
            if self.world_size > 1:
                decode_flags[0] = int(run_decode)
                decode_flags[1] = occupancy
                self.maybe_broadcast(decode_flags)
            decode_batch = decode_batch.update(
                batch_mapping,
                token_ids,
                positions,
                max_ctx_lens,
                prefill_arguments.seq_ids,
                temps,
                occupancy,
            )
            if not run_decode:
                continue
            if self.store_stream is not None:
                torch.cuda.default_stream().wait_stream(self.store_stream)

            decode_output, decode_batch = self.run_decode_loop(decode_batch)
            finished_sequence_ids = scheduler.get_finished_sequence_ids_from_unfinished(
                decode_batch.seq_ids.tolist()
            )
            scheduler.record_finished_sequence_ids(
                finished_sequence_ids, update_status=True
            )
            self.kv_manager.free_sequences(finished_sequence_ids)
            self.maybe_release_peers(scheduler.is_finished())
            scheduler.update_sequences(
                decode_output.output_tokens.tolist(),
                decode_output.output_seq_ids.tolist(),
            )
        scheduler.close()

    @torch.inference_mode()
    def _process_batches_peer(self):
        assert not self.is_master
        scheduler = Scheduler([], kv_manager=self.kv_manager)
        decode_batch = DecodeBatchArguments()
        decode_flags = torch.empty(2, dtype=torch.int32, device="cuda")
        while self.batch_ready.is_set():
            prefill_arguments = self.packed_args.build_prefill_args()

            B = prefill_arguments.B
            max_ctx_lens = (
                prefill_arguments.max_new_tokens + prefill_arguments.context_lens
            )

            seq_ids_cpu = prefill_arguments.seq_ids.tolist()
            scheduler.add_running_sequence_ids(seq_ids_cpu)
            success, batch_mapping = self.kv_manager.allocate_sequences(
                seq_ids_cpu, max_ctx_lens.tolist()
            )
            assert success, "failed to allocate pages for sequences"

            self.run_prefill(prefill_arguments, batch_mapping)
            positions, batch_mapping = prefill_arguments.context_lens, batch_mapping
            self.maybe_broadcast(decode_flags)
            run_decode = bool(decode_flags[0].item())
            occupancy = int(decode_flags[1].item())
            token_ids = torch.empty(B, dtype=torch.int64, device="cuda")
            decode_batch = decode_batch.update(
                batch_mapping,
                token_ids,
                positions,
                max_ctx_lens,
                prefill_arguments.seq_ids,
                None,  # temps not used in peer process
                occupancy,
            )

            if not run_decode:
                continue
            if self.store_stream is not None:
                torch.cuda.default_stream().wait_stream(self.store_stream)

            _, decode_batch = self.run_decode_loop(decode_batch)
            finished_sequence_ids = scheduler.get_finished_sequence_ids_from_unfinished(
                decode_batch.seq_ids.tolist()
            )
            scheduler.record_finished_sequence_ids(finished_sequence_ids)
            self.kv_manager.free_sequences(finished_sequence_ids)
            self.maybe_release_peers()
        scheduler.close()

    @torch.inference_mode()
    def run_decode_loop(
        self, decode_batch: DecodeBatchArguments
    ) -> tuple[DecodeBatchOutput, DecodeBatchArguments]:
        if self.is_master:
            tok_buffer = [decode_batch.token_ids.to("cpu", non_blocking=True)]
            seq_buffer = [decode_batch.seq_ids.to("cpu", non_blocking=True)]
        while True:
            self.maybe_broadcast(decode_batch.token_ids)
            running_batches = (decode_batch.positions < decode_batch.max_ctx_lens) & (
                decode_batch.token_ids != self.config.eos
            )
            decode_batch.token_ids = torch.masked_select(
                decode_batch.token_ids, running_batches
            )
            decode_batch.positions = torch.masked_select(
                decode_batch.positions, running_batches
            )
            decode_batch.batch_mapping = torch.masked_select(
                decode_batch.batch_mapping, running_batches
            )
            decode_batch.max_ctx_lens = torch.masked_select(
                decode_batch.max_ctx_lens, running_batches
            )
            decode_batch.seq_ids = torch.masked_select(
                decode_batch.seq_ids, running_batches
            )
            if self.is_master:
                decode_batch.temps = torch.masked_select(
                    decode_batch.temps, running_batches
                )
            num_remaining = decode_batch.token_ids.numel()
            if (
                num_remaining == 0
                or num_remaining <= decode_batch.desired_batch_occupancy
            ):
                break
            if self.enforce_eager:
                set_context(
                    is_prefill=False,
                    do_compression=False,
                    batch_mapping=decode_batch.batch_mapping,
                )
                logits = self.model.compute_logits(
                    self.model(decode_batch.token_ids, decode_batch.positions)
                )
            else:
                logits = self.run_graph_decode(
                    decode_batch.token_ids,
                    decode_batch.positions,
                    decode_batch.batch_mapping,
                )

            if self.is_master:
                decode_batch.token_ids = self.sampler(logits, decode_batch.temps)
                tok_buffer.append(decode_batch.token_ids.to("cpu", non_blocking=True))
                seq_buffer.append(decode_batch.seq_ids.to("cpu", non_blocking=True))
            decode_batch.positions += 1

        if self.is_master:
            output = DecodeBatchOutput(
                output_tokens=torch.cat(tok_buffer),
                output_seq_ids=torch.cat(seq_buffer),
            )
        else:
            output = DecodeBatchOutput(None, None)
        return output, decode_batch

    @torch.inference_mode()
    def run_graph_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch_mapping: torch.Tensor,
    ):
        set_context(
            is_prefill=False,
            do_compression=False,
            batch_mapping=batch_mapping,
        )
        bs = input_ids.shape[0]
        graph_dict = self.get_cuda_graph(bs, int(positions.max()))
        graph_dict["input_ids"][:bs] = input_ids
        graph_dict["positions"][:bs] = positions
        graph_dict["batch_mapping"].fill_(RESERVED_BATCH)
        graph_dict["batch_mapping"][:bs] = batch_mapping
        graph_dict["graph"].replay()
        return (
            graph_dict["logits"][:bs]
            if graph_dict["logits"] is not None
            else graph_dict["logits"]
        )

    @torch.inference_mode()
    def capture_cudagraph(self, batch_size: int, max_seqlen_k: int):
        dist.barrier()
        device = torch.device("cuda")
        logger.debug(
            f"Capturing CUDA graph for batch size {batch_size} ({max_seqlen_k} tokens)"
        )
        _g_input_ids = torch.zeros(batch_size, dtype=torch.int32, device=device)
        _g_positions = torch.zeros(batch_size, dtype=torch.int32, device=device)
        _g_logits = None
        key_split = num_splits_heuristic(
            batch_size * self.kv_manager.num_kv_heads,
            max_seq_len=max_seqlen_k,
            num_sms=torch.cuda.get_device_properties(device).multi_processor_count,
            max_splits=12,
        )

        success, _g_batch_mapping = self.kv_manager.allocate_sequences(
            list(range(batch_size)), [256] * batch_size
        )
        assert success

        set_context(
            is_prefill=False,
            do_compression=False,
            batch_mapping=_g_batch_mapping,
            key_split=key_split,
        )
        # warmup
        self.model.compute_logits(self.model(_g_input_ids, _g_positions))
        dist.barrier()
        decode_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(decode_graph):
            _g_logits = self.model.compute_logits(
                self.model(_g_input_ids, _g_positions)
            )
        graph_vars = {
            "graph": decode_graph,
            "input_ids": _g_input_ids,
            "positions": _g_positions,
            "batch_mapping": _g_batch_mapping,
            "logits": _g_logits,
            "key_split": key_split,
        }
        if batch_size not in self.captured_graphs:
            self.captured_graphs[batch_size] = {}
            self.min_captured_len[batch_size] = float("inf")

        self.captured_graphs[batch_size][max_seqlen_k] = graph_vars
        self.min_captured_len[batch_size] = min(
            max_seqlen_k, self.min_captured_len[batch_size]
        )
        self.kv_manager.free_sequences(list(range(batch_size)))

    def get_cuda_graph(self, batch_size: int, max_seqlen_k: int):
        batch_size = next(x for x in self.captured_graphs.keys() if x >= batch_size)
        batch_size_graphs = self.captured_graphs[batch_size]
        # we want largest seq_len that is smaller than max_seqlen_k
        best = self.min_captured_len[batch_size]
        for seq_len in batch_size_graphs.keys():
            if seq_len <= max_seqlen_k:
                best = max(best, seq_len)
        return batch_size_graphs[best]
