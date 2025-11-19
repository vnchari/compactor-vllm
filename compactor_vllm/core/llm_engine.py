import atexit
import logging
from typing import Any, List, Optional, Union

import torch.multiprocessing as mp
from compactor_vllm.compression.compression_config import (
    BatchCompressionParams,
    SequenceCompressionParams,
)
from compactor_vllm.config.engine_config import LLMConfig
from compactor_vllm.config.sampling_params import SamplingParams
from compactor_vllm.core.model_runner import ModelRunner
from compactor_vllm.models import MODEL_REGISTRY
from compactor_vllm.utils.sequence import Sequence
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

PromptLike = Union[str, List[int]]


def _runner_entry(config: LLMConfig, rank: int, evt):
    runner = None
    try:
        runner = ModelRunner(config, rank, evt)
        runner.loop()
    except Exception as e:
        logging.exception(f"Rank {rank}: {repr(e)}")
    finally:
        if runner is not None:
            runner.exit()


class LLMEngine:
    """High-level engine coordinating model runners and scheduling"""

    def __init__(self, config: LLMConfig):
        self.config = config
        if self.config.hf_config.model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model {self.config.model}")
        if config.path is None:
            from huggingface_hub import snapshot_download

            self.config.path = snapshot_download(
                repo_id=config.model, local_files_only=True
            )
            logger.info(f"Using {self.config.model} snapshot @ {self.config.path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
        if self.config.eos == -1:
            self.config.eos = self.tokenizer.eos_token_id

        self.ps = []
        world_size = int(self.config.tensor_parallel_size)
        self.events = []
        if world_size > 1:
            ctx = mp.get_context("spawn")
            for r in range(1, world_size):
                event = ctx.Event()
                p = ctx.Process(
                    target=_runner_entry,
                    args=(self.config, r, event),
                    daemon=True,
                )
                p.start()
                self.ps.append(p)
                self.events.append(event)

        self.master_model_runner = ModelRunner(
            self.config, rank=0, peer_events=self.events
        )
        atexit.register(self.exit)

    def exit(self):
        for p in self.ps:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        del self.events

    def tokenize_prompt(self, prompt: PromptLike, **tokenizer_kwargs) -> List[int]:
        """
        Turn a raw prompt into token IDs.
        """
        if isinstance(prompt, str):
            return self.tokenizer(prompt, **tokenizer_kwargs)["input_ids"]
        else:
            return list(prompt)

    def detokenize_prompt(
        self, sequences: List[Sequence], **detokenizer_kwargs
    ) -> List[str]:
        """
        Turn completed Sequences into strings.
        """
        return self.tokenizer.batch_decode(
            [s.completion_token_ids for s in sequences], **detokenizer_kwargs
        )

    def _build_sequences(
        self,
        prompts: List[PromptLike] | PromptLike,
        sampling_params: SamplingParams | List[SamplingParams],
        per_sequence_compression_params: Optional[
            SequenceCompressionParams | List[SequenceCompressionParams]
        ] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> List[Sequence]:
        """
        Build Sequence objects from prompts, sampling params, and optional
        per-sequence compression parameters.
        """
        tokenizer_kwargs = {} if tokenizer_kwargs is None else tokenizer_kwargs

        if not isinstance(prompts, list):
            prompts = [prompts]

        if isinstance(sampling_params, SamplingParams):
            sampling_params_list: List[SamplingParams] = [sampling_params] * len(
                prompts
            )
        else:
            sampling_params_list = sampling_params
            assert len(sampling_params_list) == len(prompts), (
                "sampling_params list must match prompts length"
            )

        if per_sequence_compression_params is None:
            compression_params_list: List[SequenceCompressionParams] = [
                SequenceCompressionParams(1.0) for _ in prompts
            ]
        elif isinstance(per_sequence_compression_params, SequenceCompressionParams):
            compression_params_list = [per_sequence_compression_params] * len(prompts)
        else:
            # list-like
            assert len(per_sequence_compression_params) == len(prompts), (
                "per_sequence_compression_params list must match prompts length"
            )
            compression_params_list = list(per_sequence_compression_params)

        seqs: List[Sequence] = []
        for prompt, sparams, cparams in zip(
            prompts, sampling_params_list, compression_params_list
        ):
            token_ids = self.tokenize_prompt(prompt, **tokenizer_kwargs)

            seqs.append(
                Sequence(
                    prompt_token_ids=token_ids,
                    sampling_params=sparams,
                    compression_params=cparams,
                )
            )
        return seqs

    def generate(
        self,
        prompts: List[PromptLike] | PromptLike,
        sampling_params: SamplingParams | List[SamplingParams],
        batch_compression_params: BatchCompressionParams,
        *,
        per_sequence_compression_params: Union[
            List[SequenceCompressionParams], SequenceCompressionParams
        ] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        detokenizer_kwargs: Optional[dict[str, Any]] = None,
        return_sequences: bool=False,
    ) -> List[str] | tuple[List[str], List[Sequence]]:
        """
        Accept prompts and return completed Sequences.
        Args:
            :param prompts:
                Single prompt or list of prompts, each either a raw text prompt,
                or pre-tokenized input IDs.
            :param sampling_params:
                A single SamplingParams for all prompts in this batch or a list of
                SamplingParams with the same length as ``prompts``.
            :param batch_compression_params:
                Compression settings for this batch.
            :param per_sequence_compression_params:
                Per-sequence compression parameters, including the compression
                ratio to be applied and the size of the protected regions of the
                sequence (how many start tokens and end tokens to keep uncompressed).
                If a SequenceCompressionParams instance, the same params will be
                applied to all sequences in this batch; if a list is provided,
                each SequenceCompressionParams will be attached to the corresponding
                prompt in the batch.
            :param tokenizer_kwargs:
                Extra kwargs forwarded to ``tokenizer(...)`` when tokenizing
                string prompts.
            :param detokenizer_kwargs:
                Passed through to `tokenizer.batch_decode`.
            :param return_sequences:
                Whether to return sequence objects or not
        Returns:
            :return List[Sequence]:
                One Sequence per input prompt, with `completion_token_ids`
                filled in after generation.
        """
        tokenizer_kwargs = {} if tokenizer_kwargs is None else tokenizer_kwargs
        detokenizer_kwargs = {} if detokenizer_kwargs is None else detokenizer_kwargs
        seqs = self._build_sequences(
            prompts,
            sampling_params=sampling_params,
            per_sequence_compression_params=per_sequence_compression_params,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.master_model_runner.generate(seqs, batch_compression_params)
        output_strings = self.detokenize_prompt(seqs, **detokenizer_kwargs)
        if return_sequences:
            return output_strings, seqs
        return output_strings

    def generate_chat(
        self,
        messages_batch: List[List[dict]],
        sampling_params: SamplingParams | List[SamplingParams],
        batch_compression_params: BatchCompressionParams,
        per_sequence_compression_params: Union[
            SequenceCompressionParams, List[SequenceCompressionParams]
        ],
        *,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        detokenizer_kwargs: Optional[dict[str, Any]] = None,
        return_sequences: bool = False,
    ) -> List[str] | tuple[List[str], List[Sequence]]:
        """
        Convenience API for chat-style prompts using HF `apply_chat_template`.
        Args:
            :param messages_batch:
                List of conversations, where each conversation is a list of
                message dicts like:
                    {"role": "system" | "user" | "assistant", "content": str}
            :param sampling_params:
                A single SamplingParams for all prompts in this batch or a list of
                SamplingParams with the same length as ``prompts``.
            :param batch_compression_params:
                Batch Level compression settings. Can set compression_method.
            :param per_sequence_compression_params:
                Per-sequence compression parameters, including the compression
                ratio to be applied and the size of the protected regions of the
                sequence (how many start tokens and end tokens to keep uncompressed).
                If a SequenceCompressionParams instance, the same params will be
                applied to all sequences in this batch; if a list is provided,
                each SequenceCompressionParams will be attached to the corresponding
                conversation in the batch.
            :param tokenizer_kwargs:
                Passed through to `tokenizer.apply_chat_template`.
            :param detokenizer_kwargs:
                Passed through to `tokenizer.batch_decode`.
            :param return_sequences:
                Whether to return sequence objects or not
        Returns:
            :return List[str] or tuple[List[str], List[Sequence]]:
                One string per conversation.
        """
        prompts_token_ids: List[List[int]] = []
        tokenizer_kwargs = {} if tokenizer_kwargs is None else tokenizer_kwargs
        detokenizer_kwargs = {} if detokenizer_kwargs is None else detokenizer_kwargs
        for messages in messages_batch:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                **tokenizer_kwargs,
            )
            prompts_token_ids.append(input_ids)

        return self.generate(
            prompts_token_ids,
            sampling_params=sampling_params,
            batch_compression_params=batch_compression_params,
            per_sequence_compression_params=per_sequence_compression_params,
            tokenizer_kwargs=tokenizer_kwargs,
            detokenizer_kwargs=detokenizer_kwargs,
            return_sequences=return_sequences,
        )

    def generate_from_sequences(
        self,
        seqs: List[Sequence],
        batch_compression_params: BatchCompressionParams,
    ) -> List[Sequence]:
        """
        Args:
            :param seqs:
                List of Sequence instances
            :param batch_compression_params:
                Compression settings.

        Returns:
            :return List[Sequence]:
                Same list, mutated in-place with completions.
        """
        self.master_model_runner.generate(seqs, batch_compression_params)
        return seqs
