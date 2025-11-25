import torch
import torch.distributed as dist
from compactor_vllm.triton_kernels.matmul_ogs import matmul_ogs
from torch import nn


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class TritonFusedMoeLinearBase(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.empty((num_experts, in_features, out_features)).transpose(-1, -2)
        )
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty((num_experts, out_features)))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedTritonFusedMoeLinear(TritonFusedMoeLinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, num_experts, bias)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_idx: int
    ):
        param.data[expert_idx].copy_(loaded_weight, non_blocking=True)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        w = self.weight.transpose(-1, -2)
        assert w.is_contiguous()
        return matmul_ogs(
            x,
            self.weight,
            self.bias,
            **kwargs,
        )


class RowParallelTritonFusedMoeLinear(TritonFusedMoeLinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = False,
    ) -> None:
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(
            divide(in_features, tp_size), out_features, num_experts, bias, 2
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_idx: int
    ):
        shard_size = param.size(2)
        start_idx = self.tp_rank * shard_size
        local_shard = loaded_weight[:, start_idx : start_idx + shard_size]
        param.data[expert_idx].copy_(local_shard, non_blocking=True)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        w = self.weight.transpose(-1, -2)
        assert w.is_contiguous()
        y = matmul_ogs(
            x,
            w,
            self.bias,
            **kwargs,
        )
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


class ColumnParallelTritonFusedMoeLinear(TritonFusedMoeLinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = False,
    ) -> None:
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(
            in_features, divide(out_features, tp_size), num_experts, bias, 1
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_idx: int
    ):
        shard_size = param.size(1)
        start_idx = self.tp_rank * shard_size
        local_shard = loaded_weight[start_idx : start_idx + shard_size, :]
        param.data[expert_idx].copy_(local_shard, non_blocking=True)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        w = self.weight.transpose(-1, -2)
        assert w.is_contiguous()
        y = matmul_ogs(
            x,
            w,
            self.bias,
            **kwargs,
        )
        return y


class MergedColumnParallelTritonFusedMoeLinear(ColumnParallelTritonFusedMoeLinear):
    def __init__(
        self,
        in_features: int,
        out_feature_list: list[int],
        num_experts: int,
        bias: bool = False,
    ):
        self.out_feature_list = out_feature_list
        super().__init__(in_features, sum(out_feature_list), num_experts, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_idx: int,
        shard_id: int,
    ):
        param_data = param.data
        shard_offset = sum(self.out_feature_list[:shard_id]) // self.tp_size
        shard_size = self.out_feature_list[shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        local_weight = loaded_weight.chunk(self.tp_size, dim=self.tp_dim - 1)[
            self.tp_rank
        ]
        param_data[expert_idx].copy_(local_weight, non_blocking=True)
