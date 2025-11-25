import json
import logging

from datasets import concatenate_datasets, load_dataset

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
from longbench_metrics import dataset2metric

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    datasets = [
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "gov_report",
        "qmsum",
        "multi_news",
        "trec",
        "triviaqa",
        "samsum",
        "passage_retrieval_en",
        "passage_count",
        "lcc",
        "repobench-p",
    ]
    dataset = concatenate_datasets(
        [
            load_dataset("THUDM/LongBench", n, split="test", trust_remote_code=True)
            for n in datasets
        ]
    ).shuffle(seed=42)

    # dataset = dataset.take(200)
    prompts = json.load(open("longbench_config/dataset2prompt.json", "r"))
    max_gen_lens = json.load(open("longbench_config/dataset2maxlen.json", "r"))

    tokenizer_kwargs = {"add_generation_prompt": True, "enable_thinking": False}
    dset_names = [
        item["dataset"] if item["dataset"][-2:] != "_e" else item["dataset"][:-2]
        for item in dataset
    ]
    gen_lengths = [max_gen_lens[dset_name] for dset_name in dset_names]

    messages = [
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": prompts[dset_name].format(**item)},
        ]
        for dset_name, item in zip(dset_names, dataset)
    ]
    # model = "Qwen/Qwen3-8B"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    # model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    config = LLMConfig(
        model,
        max_num_seqs=64,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=2,
        max_model_len=128000,
        attention_backend=AttentionBackend.COMPACTOR_TRITON,
        leverage_sketch_size=32,
    )
    llm = LLM(config)
    responses = llm.generate_chat(
        messages,
        [SamplingParams(max_new_tokens=g, temperature=0.00001) for g in gen_lengths],
        BatchCompressionParams(
            compression_method=CompressionMethod.COMPACTOR,
            do_chunked_compression=False,
            chunk_size=4096,
        ),
        per_sequence_compression_params=[
            SequenceCompressionParams(
                0.25, protected_first_tokens=8, protected_last_tokens=64
            )
        ]
        * len(messages),
        tokenizer_kwargs=tokenizer_kwargs,
        return_sequences=False,
    )
    results = {}
    for dset_name, prediction, item in zip(dset_names, responses, dataset):
        if dset_name not in results:
            results[dset_name] = []

        score = 0.0
        if dset_name in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]

        for ground_truth in item["answers"]:
            score = max(
                score,
                dataset2metric[dset_name](
                    prediction, ground_truth, all_classes=item["all_classes"]
                ),
            )
        results[dset_name].append(score)

    all_sum, all_count = 0, 0
    for task, scores in results.items():
        this_task_sum = sum(scores)
        this_task_count = len(scores)
        print(task, f"{this_task_sum / this_task_count:.2f}")
        all_sum += sum(scores)
        all_count += this_task_count
    print(f"ALL: {all_sum / all_count:.2f}")
