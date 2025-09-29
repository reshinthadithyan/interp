import torch
import torch.nn as nn
import numpy
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    dim: int = 2048 # model width
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8 # GQA: number of KV heads (<= n_heads)
    max_seq_len: int = 4096
    rope_theta: float = 500000.0 # Llama3 uses large θ, helps long context
    rope_scale: float = 1.0 # for scaling past max training length
    norm_eps: float = 1e-5
    hidden_multiple_of: int = 256 # MLP hidden dim rounded to this
    initializer_range: float = 0.02 # std for normal init
    bias: bool = False # linear bias (False in LLaMA)

class RMSNorm(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.eps = config.norm_eps
        self.weight = nn.Parameter(
            torch.ones(config.dim)
        )
    def forward(self, x : torch.Tensor):
        #RMSNorm is the sq Root of the mean squared
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

if __name__ == "__main__":
    print(RMSNorm)

    