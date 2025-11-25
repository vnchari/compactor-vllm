from compactor_vllm.models.llama3 import LlamaForCausalLM
from compactor_vllm.models.qwen3 import Qwen3ForCausalLM
from compactor_vllm.models.qwen3_moe import Qwen3MoeForCausalLM

MODEL_REGISTRY = {
    "llama": LlamaForCausalLM,
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
}
