from mini_llm.core.llama import LlamaConfig, RMSNorm
import torch

    

def test_rmsnorm():
    config = LlamaConfig()
    rms = RMSNorm(config)
    x = torch.randn(8,16,config.dim)
    rms_x = rms(x)
