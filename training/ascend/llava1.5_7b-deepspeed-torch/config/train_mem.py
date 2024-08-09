import torch_npu
from torch_npu.contrib import transfer_to_npu
from train import train


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
