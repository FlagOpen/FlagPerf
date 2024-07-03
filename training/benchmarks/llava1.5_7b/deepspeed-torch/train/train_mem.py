# cambricon mlu import
try:
    from torch_mlu.utils.model_transfer import transfer
except ImportError:
    pass
from train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
