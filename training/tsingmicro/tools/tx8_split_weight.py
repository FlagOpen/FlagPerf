import os
import argparse
import functools
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
import torch_xla.experimental.xla_sharding as xs
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import AutoModelForCausalLM,AutoConfig
import torch_xla.core.xla_model as xm
from transformers.trainer_pt_utils import get_module_class_from_name
from torch_xla.distributed.fsdp.wrap import recursive_wrap 
from torch_xla.distributed.fsdp.wrap import size_based_auto_wrap_policy,transformer_auto_wrap_policy

global layer_index
layer_index = 0
class TX8FullyShardedDataParallel(nn.Module):
  def __init__(
      self,
      module: nn.Module,
      auto_wrapper_callable = None,
      auto_wrap_policy = None,
      save_path = None,
      num_hidden_layers=1,
  ):
    if isinstance(module, TX8FullyShardedDataParallel):
      raise RuntimeError("cant be a TX8FullyShardedDataParallel")
    super().__init__()

    if auto_wrap_policy is not None:
      auto_wrap_kwargs = {
          "module": module,
          "auto_wrap_policy": auto_wrap_policy,
          "wrapper_cls": auto_wrapper_callable or TX8FullyShardedDataParallel,
          "ignored_modules": [],
          "ignored_params": [],
          "only_wrap_children": True,  
      }
      fsdp_kwargs = dict(save_path=save_path,num_hidden_layers=num_hidden_layers)
      self._auto_wrap(auto_wrap_kwargs, fsdp_kwargs)

    global layer_index
    for name, param in module.named_parameters():
        if layer_index >= num_hidden_layers:
            torch.save(param,f"{save_path}/{name}.pt")
        else: 
            torch.save(param,f"{save_path}/{name}_{layer_index}.pt")

    print(f"!!!!!!!download layer index: {layer_index}, save_path:{save_path}")
    layer_index += 1

  def _auto_wrap(
      self,
      auto_wrap_kwargs: Dict[str, Any],
      fsdp_kwargs: Dict[str, Any],
  ) -> None:
    auto_wrap_policy = auto_wrap_kwargs["auto_wrap_policy"]
    root_module = auto_wrap_kwargs["module"]
    assert auto_wrap_policy is not None
    for module_name, module in root_module.named_modules():
      if isinstance(module, TX8FullyShardedDataParallel):
        raise ValueError(
            f"Expected {module_name} to NOT be TX8FullyShardedDataParallel "
            "if using an `auto_wrap_policy`")

    recursive_wrap(**auto_wrap_kwargs, **fsdp_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/workplace/data/hugginface/models/llama3_model/Meta-Llama-3-70B", type=str)
    parser.add_argument("--save_path", default="/workplace/SPMD_TX8_DEVELOP/examples/llama3_70B_finetune/weights", type=str)
    parser.add_argument("--torch_dtype", default="float32", type=str)
    parser.add_argument("--wrap_layer", default="LlamaDecoderLayer", type=str)
    parser.add_argument("--num_hidden_layers", default=1,type=int)
    args = parser.parse_args()
    assert os.path.exists(args.model_name_or_path) , "model_name_or_path is not exists!"
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_hidden_layers = args.num_hidden_layers
    torch_dtype = getattr(torch, args.torch_dtype)
    
    print(f"!!!!! num_hidden_layers :{args.num_hidden_layers} , dtype:{torch_dtype} , wrap_layer: {args.wrap_layer}")
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config)
    auto_wrap_policy = None
    fsdp_transformer_layer_cls_to_wrap = args.wrap_layer.split(',')
    if fsdp_transformer_layer_cls_to_wrap: 
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(model, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
    model = TX8FullyShardedDataParallel(model,auto_wrap_policy=auto_wrap_policy,save_path=args.save_path,num_hidden_layers=args.num_hidden_layers)
    print(f"!!!!!!!download model over")