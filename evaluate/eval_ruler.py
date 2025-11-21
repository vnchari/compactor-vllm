import argparse
import logging
import os
import json
from datetime import datetime

import torch
from datasets import load_dataset

from compactor_vllm import (
    LLM,
    LLMConfig,
    SamplingParams,
)
from compactor_vllm.compression import (
    BatchCompressionParams,
    CompressionMethod,
    SequenceCompressionParams,
)
from compactor_vllm.config.engine_config import AttentionBackend
from evaluate.ruler_metrics import score_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RULER evaluation with compactor_vllm."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--dataset-length",
        type=str,
        default="4096",
        help="Dataset configuration name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for the dataset.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of the dataset to use in (0, 1]. "
            "E.g., 0.1 uses 10%% of the shuffled dataset."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
        help="Maximum number of sequences to batch.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory to use.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism degree.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=40960,
        help="Maximum model context length.",
    )
    backend_choices = [backend.name.lower() for backend in AttentionBackend]
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="compactor_triton",
        choices=backend_choices,
        help=f"Attention backend to use. Choices: {backend_choices}",
    )
    parser.add_argument(
        "--leverage-sketch-size",
        type=int,
        default=48,
        help="Leverage sketch size for compactor attention.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 is greedy).",
    )
    method_choices = [m.name.lower() for m in CompressionMethod]
    parser.add_argument(
        "--compression-method",
        type=str,
        default="compactor",
        choices=method_choices,
        help=f"Compression method. Choices: {method_choices}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk size for chunked compression.",
    )
    parser.add_argument(
        "--no-chunked-compression",
        dest="do_chunked_compression",
        action="store_false",
        help="Disable leverage chunked compression (enabled by default).",
    )
    parser.set_defaults(do_chunked_compression=True)
    parser.add_argument(
        "--seq-compression-ratio",
        type=float,
        default=0.5,
        help="Compression ratio for SequenceCompressionParams.",
    )
    parser.add_argument(
        "--protected-first-tokens",
        type=int,
        default=8,
        help="Number of protected tokens at the beginning of each sequence.",
    )
    parser.add_argument(
        "--extra-protected-last-tokens",
        type=int,
        default=16,
        help=(
            "Extra number of protected tokens at the end, in addition to the "
            "tokenized length of answer_prefix+question."
        ),
    )
    parser.add_argument(
        "--tokenizer-add-generation-prompt",
        action="store_true",
        help="Set tokenizer_kwargs['add_generation_prompt']=True (default False).",
    )
    parser.add_argument(
        "--tokenizer-enable-thinking",
        action="store_true",
        help="Set tokenizer_kwargs['enable_thinking']=True (default False).",
    )
    parser.add_argument(
        "--no-tokenizer-continue-final-message",
        dest="tokenizer_continue_final_message",
        action="store_false",
        help="Set tokenizer_kwargs['continue_final_message']=False (default True).",
    )
    parser.set_defaults(tokenizer_continue_final_message=True)

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save detailed evaluation results.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Loading dataset %s (length=%s, split=%s)",
        "simonjegou/ruler",
        args.dataset_length,
        "test",
    )
    dataset = load_dataset(
        "simonjegou/ruler",
        args.dataset_length,
        split="test",
    )
    if args.seed is not None and args.seed >= 0:
        logger.info("Shuffling dataset with seed %d", args.seed)
        dataset = dataset.shuffle(seed=args.seed)
    if not (0 < args.fraction <= 1.0):
        raise ValueError("--fraction must be in the interval (0, 1].")
    if args.fraction < 1.0:
        n_examples = max(1, int(len(dataset) * args.fraction))
        logger.info(
            "Using %.2f fraction of data: %d / %d examples",
            args.fraction,
            n_examples,
            len(dataset),
        )
        dataset = dataset.select(range(n_examples))
    else:
        logger.info("Using full dataset: %d examples", len(dataset))
    tokenizer_kwargs = {
        "add_generation_prompt": args.tokenizer_add_generation_prompt,
        "enable_thinking": args.tokenizer_enable_thinking,
        "continue_final_message": args.tokenizer_continue_final_message,
    }
    messages = [
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": example["context"] + " " + example["question"],
            },
            {
                "role": "assistant",
                "content": example["answer_prefix"],
            },
        ]
        for example in dataset
    ]
    attention_backend = AttentionBackend[args.attention_backend.upper()]
    compression_method = CompressionMethod[args.compression_method.upper()]
    logger.info("Using model: %s", args.model)
    config = LLMConfig(
        args.model,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        attention_backend=attention_backend,
        leverage_sketch_size=args.leverage_sketch_size,
    )
    llm = LLM(config)

    end_protected_lengths = [
        args.extra_protected_last_tokens
        + len(
            llm.tokenizer(example["answer_prefix"] + example["question"])["input_ids"]
        )
        for example in dataset
    ]

    per_sequence_compression_params = [
        SequenceCompressionParams(
            args.seq_compression_ratio,
            protected_first_tokens=args.protected_first_tokens,
            protected_last_tokens=end_protected_length,
        )
        for end_protected_length in end_protected_lengths
    ]

    # Sampling params
    sampling_params = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Batch compression params
    batch_compression_params = BatchCompressionParams(
        compression_method=compression_method,
        do_chunked_compression=args.do_chunked_compression,
        chunk_size=args.chunk_size,
    )
    logger.info("Running generate_chat on %d examples.", len(messages))
    responses = llm.generate_chat(
        messages,
        sampling_params,
        batch_compression_params,
        per_sequence_compression_params=per_sequence_compression_params,
        tokenizer_kwargs=tokenizer_kwargs,
        return_sequences=False,
    )
    logger.info("Scoring responses.")
    results = {}
    per_example = []

    all_sum, all_count = 0.0, 0

    for idx, (example, response) in enumerate(zip(dataset, responses)):
        task = example["task"]
        answer = example["answer"]
        score = score_function(
            generated=response,
            ground_truth=answer,
            task_category=task,
        )
        if task not in results:
            results[task] = []
        results[task].append(score)

        all_sum += score
        all_count += 1

        per_example.append(
            {
                "index": idx,
                "task": task,
                "context": example["context"],
                "question": example["question"],
                "answer_prefix": example["answer_prefix"],
                "ground_truth": answer,
                "generated": response,
                "score": score,
                "compression_params": {
                    "seq_compression_ratio": args.seq_compression_ratio,
                    "protected_first_tokens": args.protected_first_tokens,
                    "protected_last_tokens": end_protected_lengths[idx],
                },
            }
        )

    per_task_summary = {}
    for task, scores in results.items():
        this_task_sum = sum(scores)
        this_task_count = len(scores)
        avg = this_task_sum / this_task_count
        print(task, f"{avg:.3f}")
        per_task_summary[task] = {
            "avg_score": avg,
            "num_examples": this_task_count,
            "sum_scores": this_task_sum,
        }

    overall_avg = all_sum / all_count if all_count > 0 else 0.0
    print(f"ALL: {overall_avg:.3f}")

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model.replace("/", "_")
    base_name = f"ruler_{args.dataset_length}_{safe_model_name}_{timestamp}"

    summary_path = os.path.join(args.results_dir, base_name + "_summary.json")
    details_path = os.path.join(args.results_dir, base_name + "_details.jsonl")

    logger.info("Saving summary to %s", summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "model": args.model,
                "dataset": "simonjegou/ruler",
                "dataset_length": args.dataset_length,
                "num_examples": len(dataset),
                "overall_avg_score": overall_avg,
                "per_task": per_task_summary,
                "arguments": vars(args),  # all CLI args
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Saving per-example details to %s", details_path)
    with open(details_path, "w", encoding="utf-8") as f:
        for row in per_example:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(parse_args())
