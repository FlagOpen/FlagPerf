from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import logging
import numpy as np
import torch
# Load model directly
from transformers import AutoModel
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
import torch_xla.core.xla_model as xm

hf_models = '/workspace/data/hugginface/models/Qwen--Qwen1.5-MoE-A2.7B'
device = xm.xla_device()
tokenizer = AutoTokenizer.from_pretrained(hf_models,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(hf_models,trust_remote_code=True)
print(model)
input_ids = torch.tensor([[1,2,3,4,5]]).to(torch.long).to(device)
output = model.to(device)(input_ids)
xm.mark_step()
print('hf_model:', hf_models)