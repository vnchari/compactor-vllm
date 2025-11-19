import os
from glob import glob

import torch
import torch.distributed as dist
import tqdm
from safetensors import safe_open
from torch import nn
from transformers import Qwen3MoeConfig

from compactor_vllm.compression import (
    apply_postrope_compression,
    apply_prerope_compression,
)
from compactor_vllm.layers.activation import SiluAndMul
from compactor_vllm.layers.attention import Attention
from compactor_vllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from compactor_vllm.layers.layernorm import RMSNorm
from compactor_vllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from compactor_vllm.layers.moe import (
    MergedColumnParallelTritonFusedMoeLinear,
    RowParallelTritonFusedMoeLinear,
)
from compactor_vllm.layers.rotary_embedding import get_rope
from compactor_vllm.triton_kernels.routing import routing
from compactor_vllm.utils.context import get_context


class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        scores = None
        if context.is_prefill and context.do_compression:
            scores = apply_prerope_compression(q, k, v, context)

        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)

        if context.is_prefill and context.do_compression:
            scores = apply_postrope_compression(q, k, v, scores, context)

        o = self.attn(q, k, v, scores)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3MoeTritonSparseMoeBlock(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.hidden_size = hidden_size
        self.moe_intermediate_size = intermediate_size

        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.gate_up_proj = MergedColumnParallelTritonFusedMoeLinear(
            hidden_size, [intermediate_size] * 2, num_experts
        )
        self.down_proj = RowParallelTritonFusedMoeLinear(
            intermediate_size, hidden_size, num_experts
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        if x.numel() == 0:
            return x
        logits = self.gate(x)
        rdata, gather_indx, scatter_indx = routing(
            logits,
            self.num_experts_per_tok,
            simulated_ep=1,  # single device, replicated experts
        )
        x = self.gate_up_proj(x, routing_data=rdata, gather_indx=gather_indx)
        x = self.act_fn(x)
        x = self.down_proj(
            x, routing_data=rdata, scatter_indx=scatter_indx, gammas=rdata.gate_scal
        )
        return x


class Qwen3MoeBlock(Qwen3MoeTritonSparseMoeBlock):
    pass


class Qwen3MoeRMSNorm(RMSNorm):
    pass


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            sliding_window=config.sliding_window,
        )
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeBlock(
                num_experts=config.num_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                num_experts_per_tok=config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
                hidden_act=config.hidden_act,
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3MoeModel(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_ids,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.num_experts = config.num_experts
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, position_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_model(
        self,
        path: str,
        *,
        use_tqdm: bool = False,
    ) -> None:
        rank = dist.get_rank()
        all_shards = glob(os.path.join(path, "*.safetensors"))
        for file in (
            tqdm.tqdm(all_shards, desc="Loading model") if use_tqdm else all_shards
        ):
            with safe_open(file, "pt", f"cuda:{rank}") as f:
                for weight_name in f.keys():
                    weight_tensor = f.get_tensor(weight_name)
                    is_expert = "mlp.experts" in weight_name
                    is_loaded = False

                    # Process experts params name
                    if is_expert:
                        mlp_module_name, expert_module_name = weight_name.split(
                            ".experts."
                        )
                        expert_idx = int(expert_module_name.split(".")[0])
                        proj_name = expert_module_name.replace(f"{expert_idx}.", "")
                        weight_name = f"{mlp_module_name}.{proj_name}"

                    # Load packed modules
                    for k in self.packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = self.packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = self.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            if is_expert:
                                weight_loader(
                                    param, weight_tensor, expert_idx, shard_id
                                )
                            else:
                                weight_loader(param, weight_tensor, shard_id)
                            is_loaded = True
                            break

                    # Load other modules
                    if not is_loaded:
                        param = self.get_parameter(weight_name)
                        weight_loader = getattr(
                            param,
                            "weight_loader",
                            lambda p, lw: p.data.copy_(lw, non_blocking=True),
                        )
                        if is_expert:
                            weight_loader(param, weight_tensor, expert_idx)
                        else:
                            weight_loader(param, weight_tensor)
                        is_loaded = True

                    assert is_loaded, f"Weight {weight_name} not loaded"
