import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.models.llama.weight import load_from_gptq_llama
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--meta_ckpt_dir', type=str, default=None)

    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--inter_size', type=int, default=11008)
    parser.add_argument('--rms_norm_eps', type=float, default=1e-06)

    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--disable_weight_only_quant_plugin',
        default=False,
        action="store_true",
        help=
        'By default, using plugin implementation for weight quantization. Enabling disable_weight_only_quant_plugin flag will use ootb implementation instead of plugin.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4', 'int4_gptq'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]")
    parser.add_argument(
        '--per_channel',
        action="store_true",
        default=False,
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        action="store_true",
        default=False,
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--modelopt_quant_ckpt_path',
        type=str,
        default=None,
        help='Path of a quantized model checkpoint in .npz format')

    parser.add_argument(
        '--per_group',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale weights in the int4 range. '
        'per_group chooses at run time, and for each group, a custom scaling factor. '
        'The flag is built for GPTQ/AWQ quantization.')

    parser.add_argument('--load_by_shard',
                        action='store_true',
                        help='Load a pretrained model shard-by-shard.')
    parser.add_argument('--hidden_act', type=str, default='silu')

    parser.add_argument('--rotary_base', type=float, default=10000.0)

    parser.add_argument('--group_size',
                        type=int,
                        default=128,
                        help='Group size used in GPTQ quantization.'
                        )  # AWQ is only supported by quantize.py script

    parser.add_argument("--dataset-cache-dir",
                        type=str,
                        default=None,
                        help="cache dir to load the hugging face dataset")
    parser.add_argument("--load_model_on_cpu", action="store_true")
    parser.add_argument(
        '--use_parallel_embedding',
        action="store_true",
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0'
    )
    parser.add_argument(
        '--use_embedding_sharing',
        action="store_true",
        default=False,
        help=
        'Try to reduce the engine size by sharing the embedding lookup table between two layers.'
        'Note: the flag might not take effect when the criteria are not met.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument(
        '--moe_num_experts',
        default=0,
        type=int,
        help='Specify the number of experts to use for MOE layers')
    parser.add_argument(
        '--moe_top_k',
        default=0,
        type=int,
        help=
        'Specify the top_k value to use for MOE layers. Default to 1 if --moe_num_experts is set'
    )
    parser.add_argument(
        '--moe_tp_mode',
        default=MoeConfig.ParallelismMode.TENSOR_PARALLEL,
        type=int,
        help=
        'Controls how to distribute experts in TP. Check layers/moe.py for accepted values',
    )
    parser.add_argument(
        '--moe_renorm_mode',
        default=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        type=int,
        help=
        'Controls renormalization after gate logits. Check layers/moe.py for accepted values',
    )
    parser.add_argument(
        '--save_config_only',
        action="store_true",
        default=False,
        help=
        'Only save the model config w/o read and converting weights, be careful, this is for debug only'
    )

    args = parser.parse_args()
    # changing the default to be consistent as the cli help said.
    if args.moe_num_experts and args.moe_top_k == 0:
        args.moe_top_k = 1
    return args


def args_to_quantization(args: argparse.Namespace) -> QuantConfig:
    '''return config dict with quantization info based on the command line args
    '''
    quant_config = QuantConfig()
    quant_config.exclude_modules = ['lm_head']
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            quant_config.quant_algo = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            quant_config.quant_algo = QuantAlgo.W4A16
    elif args.smoothquant:
        quant_config.smoothquant_val = args.smoothquant
        if args.per_channel:
            if args.per_token:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
            else:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        else:
            if args.per_token:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
            else:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN

    if args.int8_kv_cache:
        quant_config.kv_cache_quant_algo = QuantAlgo.INT8

    if args.weight_only_precision == 'int4_gptq':
        quant_config.group_size = args.group_size
        quant_config.has_zero_point = True
        quant_config.pre_quant_scale = False
        quant_config.quant_algo = QuantAlgo.W4A16_GPTQ

    return quant_config


def convert_and_save_meta(args, rank):
    mapping = Mapping(world_size=args.tp_size * args.pp_size,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size,
                      rank=rank)
    assert not args_to_quantization(args).quant_mode.has_any_quant(), \
        "quantization from meta checkpoint or empty model were never supported"
    llama = LLaMAForCausalLM.from_meta_ckpt(
        args.meta_ckpt_dir,
        args.dtype,
        mapping,
        use_parallel_embedding=args.use_parallel_embedding,
        embedding_sharding_dim=args.embedding_sharding_dim)
    llama.save_checkpoint(args.output_dir, save_config=(rank == 0))


def args_to_build_options(args):
    return {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.use_embedding_sharing,
        'disable_weight_only_quant_plugin':
        args.disable_weight_only_quant_plugin
    }


def from_cli_args(args):
    n_kv_head = args.n_kv_head if args.n_kv_head is not None else args.n_head
    config = {
        'architecture': "LlamaForCausalLM",
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': args.n_layer,
        'num_attention_heads': args.n_head,
        'hidden_size': args.n_embd,
        'intermediate_size': args.inter_size,
        'num_key_value_heads': n_kv_head,
        'vocab_size': args.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': args.n_positions,
        'hidden_act': args.hidden_act,
        'rotary_base': args.rotary_base,
        'norm_epsilon': args.rms_norm_eps,
        'moe_num_experts': args.moe_num_experts,
        'moe_top_k': args.moe_top_k,
        'moe_tp_mode': args.moe_tp_mode,
        'moe_normalization_mode': args.moe_renorm_mode,
        'mapping': {
            'world_size': args.tp_size * args.pp_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size
        },
        'quantization': args_to_quantization(args).asdict()
    }
    config.update(args_to_build_options(args))
    return config


def preload_model(model_dir, load_model_on_cpu):
    from transformers import AutoConfig, AutoModelForCausalLM
    if "vila" in model_dir:
        sys.path.append(model_dir + "/../VILA")
        from llava.model import LlavaConfig, LlavaLlamaForCausalLM
        AutoConfig.register("llava_llama", LlavaConfig)
        AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_cls = AutoModelForCausalLM
    if hf_config.model_type == "llava":
        from transformers import LlavaForConditionalGeneration
        model_cls = LlavaForConditionalGeneration
    model = model_cls.from_pretrained(
        model_dir,
        device_map='auto' if not load_model_on_cpu else 'cpu',
        torch_dtype='auto',
        trust_remote_code=True,
    )
    if hf_config.model_type == "llava":
        model = model.language_model
    return model


def convert_and_save_hf(args):
    model_dir = args.model_dir
    load_model_on_cpu = args.load_model_on_cpu
    load_by_shard = args.load_by_shard
    world_size = args.tp_size * args.pp_size
    # Need to convert the cli args to the kay-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {'moe_tp_mode': args.moe_tp_mode}
    quantization = args_to_quantization(args)
    override_fields.update(args_to_build_options(args))

    if args.smoothquant is not None or args.int8_kv_cache:
        assert not args.load_by_shard, "When using quantization, TRT-LLM needs to load the whole HF model, thus load by shard not supported"
        assert not args.load_model_on_cpu, "When using quantization, TRT-LLM needs to load the model to GPU"
        mapping = Mapping(
            world_size=world_size,
            rank=-1,  #intentinoally make -1 to avoid mistake
            tp_size=args.tp_size,
            pp_size=args.pp_size)
        LLaMAForCausalLM.quantize(args.model_dir,
                                  args.output_dir,
                                  quantization,
                                  dtype=args.dtype,
                                  mapping=mapping,
                                  override_fields=override_fields,
                                  dataset_cache_dir=args.dataset_cache_dir)
    else:
        # When not loading by shard, preload one complete model and then slice per rank weights from this
        # this saves the disk reloading time
        hf_model = preload_model(
            model_dir, load_model_on_cpu) if not args.load_by_shard else None

        def convert_and_save_rank(args, rank):
            mapping = Mapping(world_size=world_size,
                              rank=rank,
                              tp_size=args.tp_size,
                              pp_size=args.pp_size)
            llama = LLaMAForCausalLM.from_hugging_face(
                model_dir,
                args.dtype,
                mapping=mapping,
                quantization=quantization,
                load_by_shard=load_by_shard,
                load_model_on_cpu=load_model_on_cpu,
                override_fields=override_fields,
                preloaded_model=hf_model,
            )
            llama.save_checkpoint(args.output_dir, save_config=(rank == 0))
            del llama
            release_gc()

        execute(args.workers, [convert_and_save_rank] * world_size, args)


def convert_and_save_gptq(args, rank):
    mapping = Mapping(world_size=args.tp_size * args.pp_size,
                      tp_size=args.tp_size,
                      rank=rank,
                      pp_size=args.pp_size)
    llama = LLaMAForCausalLM.from_hugging_face(
        args.model_dir,
        args.dtype,
        mapping=mapping,
        quantization=args_to_quantization(args),
        skip_loading_weights=True)
    weights = load_from_gptq_llama(llama.config, args.modelopt_quant_ckpt_path)
    llama.load(weights)
    llama.save_checkpoint(args.output_dir, rank == 0)


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()

    world_size = args.tp_size * args.pp_size
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if (args.model_dir is None
            and args.meta_ckpt_dir is None):  # generate fake config.json
        config = from_cli_args(args)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    elif args.meta_ckpt_dir is not None:
        assert args.model_dir is None, "Shall not specify both meta checkpoint dir and hugging face dir"
        execute(args.workers, [convert_and_save_meta] * world_size, args)
    elif args.weight_only_precision == 'int4_gptq':
        assert args.model_dir is not None
        assert args.modelopt_quant_ckpt_path is not None
        execute(args.workers, [convert_and_save_gptq] * world_size, args)
    else:  # all other non-gptq paths from hf model
        assert args.model_dir is not None
        assert args.modelopt_quant_ckpt_path is None, "only gptq weights only needs this option"
        convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
