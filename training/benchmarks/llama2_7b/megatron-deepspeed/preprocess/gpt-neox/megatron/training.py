# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from its original version
#

"""Pretrain utilities."""
from datetime import datetime
from functools import partial

import math
import sys
from contextlib import nullcontext

import torch
import deepspeed
from deepspeed.runtime.data_pipeline.curriculum_scheduler import CurriculumScheduler
import numpy as np

from megatron.utils import (
    Timers,
    init_wandb,
    get_ltor_masks_and_position_ids,
    reduce_losses,
)

from megatron import print_rank_0, mpu
from megatron.model import (
    GPT2ModelPipe,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
)
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.data.data_utils import build_train_valid_test_data_iterators
from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR
from megatron.logging import tb_wandb_log, training_log
from megatron.utils import (
    OverflowMonitor,
    get_noise_scale_logger,
    get_total_params,
    CharCounter,
)
from megatron.model.gpt2_model import cross_entropy

from pickle import dump
import os


def mup_weights_reinit(neox_args, model):
    def has_method(o, name):
        return callable(getattr(o, name, None))

    for layer in model.modules():
        # This normally would happen in set_base_shapes if we actually were able to use the MuReadout class
        if hasattr(layer, "mup_rescale_parameters") and layer.mup_rescale_parameters:
            layer._rescale_parameters()

        if has_method(layer, "mup_reinitialize_weights"):
            layer.mup_reinitialize_weights(neox_args)


def save_base_shapes(neox_args, base_shapes, use_cache):

    # Instantiation of the base model fails in the init function (init_functions.py) because we haven't called set_base_shapes on it at this point, so disable it temporarily here
    neox_args.use_mup = False

    base_model = GPT2ModelPipe(
        neox_args=neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    if not neox_args.is_pipe_parallel:
        base_model = base_model.to_sequential()

    try:
        import mup
    except ModuleNotFoundError:
        print("Please install mup https://github.com/microsoft/mup")
        raise Exception

    base_shapes = mup.get_shapes(base_model)

    del base_model

    old_hidden_size = neox_args.hidden_size
    neox_args.hidden_size = neox_args.hidden_size * neox_args.mup_width_scale

    delta_model = GPT2ModelPipe(
        neox_args=neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    if not neox_args.is_pipe_parallel:
        delta_model = delta_model.to_sequential()

    delta_shapes = mup.get_shapes(delta_model)

    # change back
    neox_args.use_mup = True
    neox_args.hidden_size = old_hidden_size

    save_shapes = f"{neox_args.base_shapes_file}.{torch.distributed.get_rank()}"
    print(f"saving base shapes at {save_shapes}")
    mup.make_base_shapes(base_shapes, delta_shapes, savefile=save_shapes)
    print(f"base shapes saved...exiting")
    sys.exit(1)


def mup_coord_check(neox_args, timers, lr_scheduler, train_data_iterator):
    from megatron.mup_substitute import get_coord_data
    from mup.coord_check import plot_coord_data

    def lazy_model(hidden_size):
        def gen():
            old_hidden_size = neox_args.hidden_size
            neox_args.hidden_size = hidden_size

            model, optimizer, _ = setup_model_and_optimizer(
                neox_args=neox_args, use_cache=False
            )

            neox_args.hidden_size = old_hidden_size

            return model

        return gen

    models = {}

    # Hidden size needs to be divisible by num attention heads
    for hidden_size in (neox_args.num_attention_heads * (2**p) for p in range(2, 9)):
        models[hidden_size] = lazy_model(hidden_size)

    neox_args.use_mup = True
    df_up = get_coord_data(
        neox_args, timers, lr_scheduler, models, train_data_iterator, mup=True
    )
    neox_args.use_mup = False
    df_sp = get_coord_data(
        neox_args, timers, lr_scheduler, models, train_data_iterator, mup=False
    )

    plot_coord_data(df_up, save_to=f"coord_check_up.{torch.distributed.get_rank()}.jpg")
    plot_coord_data(df_sp, save_to=f"coord_check_sp.{torch.distributed.get_rank()}.jpg")

    print_rank_0("Saved coord check plots... exiting")
    sys.exit(1)


def pretrain(neox_args):
    """Main training program.

    This function will run the following in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model.

    Arguments:
        neox_args: an instance of NeoXArgs containing the configuration for pretrain

    """
    # setup logging and timers
    init_wandb(neox_args=neox_args)
    timers = Timers(
        use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer
    )

    # Initialize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(neox_args=neox_args)

    # Model, optimizer, and learning rate.
    timers("model and optimizer").start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        neox_args=neox_args, use_cache=False, iteration=neox_args.iteration
    )
    timers("model and optimizer").stop()

    # Data stuff.
    timers("train/valid/test data iterators").start()
    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    ) = build_train_valid_test_data_iterators(neox_args=neox_args)
    timers("train/valid/test data iterators").stop()

    if neox_args.use_mup and neox_args.coord_check:
        mup_coord_check(neox_args, timers, lr_scheduler, train_data_iterator)

    # Print setup timing.
    print_rank_0("done with setups ...")
    timers.log(["model and optimizer", "train/valid/test data iterators"])
    print_rank_0("training ...")

    iteration = neox_args.iteration
    # edge case: save step 0 checkpoint if requested and we're starting from step 0
    if neox_args.save and 0 in neox_args.save_iters and iteration == 0:
        save_checkpoint(
            neox_args=neox_args,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    if neox_args.do_train and neox_args.train_iters > 0:
        iteration = train(
            neox_args=neox_args,
            timers=timers,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_data_iterator=train_data_iterator,
            valid_data_iterator=valid_data_iterator,
        )

    if neox_args.do_valid:
        prefix = "the end of training for val data"
        evaluate_and_print_results(
            neox_args=neox_args,
            prefix=prefix,
            forward_step_func=forward_step,
            data_iterator=valid_data_iterator,
            model=model,
            iteration=iteration,
            verbose=False,
            timers=timers,
        )

    if neox_args.save and iteration != 0:
        save_checkpoint(
            neox_args=neox_args,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    if neox_args.do_test:
        # Run on test data.
        prefix = "the end of training for test data"
        evaluate_and_print_results(
            neox_args=neox_args,
            prefix=prefix,
            forward_step_func=forward_step,
            data_iterator=test_data_iterator,
            model=model,
            iteration=iteration,
            verbose=True,
            timers=timers,
            chart_name="test",
        )


def _get_batch(neox_args, tokenizer, keys, data, datatype):
    """Support function for get_batch / get_batch pipe (to avoid code repetition)"""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["text"].long()
    if "label" in data_b:
        labels = torch.where(
            data_b["label"].long() >= 0,
            data_b["label"].long(),
            torch.zeros_like(data_b["label"].long()),
        )[:, 1:].contiguous()
    else:
        labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
        sliding_window_width=neox_args.sliding_window_width,
    )
    # If `label` is present, any token < 0 (e.g., -100, the default for torch) skips the loss computation
    if "label" in data_b:
        loss_mask = (data_b["label"][:, 1:] >= 0).to(loss_mask.dtype)
    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch(neox_args, data_iterator):
    """Generate a batch"""

    # Items and their type.
    keys = ["text", "label"] if neox_args.label_data_paths else ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return _get_batch(
        neox_args=neox_args,
        tokenizer=neox_args.tokenizer,
        keys=keys,
        data=data,
        datatype=datatype,
    )


def get_batch_pipe(data, neox_args, curr_scheduler=None):
    """A modification of get_batch() to work with the latest batch instead of an iterator."""
    # Items and their type.
    keys = ["text", "label"] if neox_args.label_data_paths else ["text"]
    datatype = torch.int64

    tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(
        neox_args, neox_args.tokenizer, keys, data, datatype
    )
    if curr_scheduler is not None:
        # iteration + 1 to align with how/when DeepSpeed updates the buffers
        curriculum_seqlen = curr_scheduler.update_difficulty(neox_args.iteration + 1)
        if curriculum_seqlen < tokens.size()[1]:
            # seqlen-based curriculum learning
            # input_ids, position_ids, labels have size [batch size, seqlen]
            # input_ids = input_ids[:, :curriculum_seqlen].contiguous()
            tokens = tokens[:, :curriculum_seqlen].contiguous()
            position_ids = position_ids[:, :curriculum_seqlen].contiguous()
            if labels is not None:
                labels = labels[:, :curriculum_seqlen].contiguous()
            if loss_mask is not None:
                loss_mask = loss_mask[:, :curriculum_seqlen].contiguous()
            # attention_mask has size [1, 1, seqlen, seqlen]
            attention_mask = attention_mask[
                :, :, :curriculum_seqlen, :curriculum_seqlen
            ].contiguous()

    # unpack data
    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def get_batch_sequential(forward_input, neox_args):
    """A modification of get_batch() to work with the latest batch instead of an iterator."""
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=forward_input[0],
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )
    return (forward_input[0], forward_input[1], attention_mask)


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group()
    )

    return averaged_losses


def mb_moe_loss_func(args, loss_mask, output_tensor=None):
    from megatron.model import megablocks_utils
    from megatron.model.megablocks_utils import moe

    # NOTE: For pipeline parallelism this function will be run on the
    # non-final stages to calculate load balancing loss contribution
    # for the MoE layers within the stage. For these cases, output_tensor
    # will be None.
    loss, loss_dict = (None, {})
    if False:
        assert output_tensor is not None
        loss, loss_dict = loss_func(loss_mask, output_tensor)
        assert loss.numel() == 1

    # NOTE: If recompute is enabled we will collect duplicate load
    # balancing loss contributions. Prune these before calculating
    # the load balancing loss.
    if args.checkpoint_activations:
        # Ignore load balancing loss contributions compute during
        # the forward pass if recompute is turned on.
        load_balancing_loss_data = moe.get_load_balancing_loss()
        if args.num_layers * 2 == len(load_balancing_loss_data):
            load_balancing_loss_data = load_balancing_loss_data[args.num_layers :]
            moe.clear_load_balancing_loss()
            for x in load_balancing_loss_data:
                moe.save_load_balancing_loss(x)

    # Compute the load balancing loss for all MoE layers.
    megablocks_args = args = megablocks_utils.as_megablocks_args(args)
    lbl = moe.batched_load_balancing_loss(megablocks_args)
    moe.clear_load_balancing_loss()

    # Average the load balancing loss across data parallel
    # replicas and save for logging.
    averaged_lbl = average_losses_across_data_parallel_group([lbl])
    loss_dict["load balancing loss"] = averaged_lbl[0]
    return averaged_lbl, loss_dict


def forward_step(
    data_iterator, model, neox_args, timers, return_logits=False, is_train=False
):
    """Forward step."""
    if neox_args.is_pipe_parallel:
        return model.eval_batch(data_iterator, return_logits=return_logits)

    # Get the batch.
    if neox_args.memory_profiling and neox_args.it:
        torch.cuda.nvtx.range_push(f"Get batch")
    if timers is not None:
        timers("batch generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        neox_args=neox_args, data_iterator=data_iterator
    )

    if timers is not None:
        timers("batch generator").stop()
    if neox_args.memory_profiling:
        torch.cuda.nvtx.range_pop()

    if neox_args.memory_profiling:
        torch.cuda.nvtx.range_push(f"Forward pass")
    # Sequential returns moe_losses, but this is not yet supported by pipe parallel
    maybe_tuple = model((tokens, position_ids, attention_mask), neox_args=neox_args)
    if type(maybe_tuple) is tuple:
        outputs, moe_losses = maybe_tuple
    else:
        outputs = maybe_tuple
        moe_losses = []
    if (
        is_train
        and neox_args.curriculum_learning
        and neox_args.curriculum_seqlen < neox_args.seq_length
    ):
        loss_mask = loss_mask[:, : neox_args.curriculum_seqlen].contiguous()
        labels = labels[:, : neox_args.curriculum_seqlen].contiguous()
    main_loss = cross_entropy(
        outputs, (labels, loss_mask), _fp16=neox_args.fp16_lm_cross_entropy
    )
    if neox_args.moe_num_experts > 1:
        if neox_args.moe_type == "deepspeed":
            moe_loss = neox_args.moe_loss_coeff * sum(m.item() for m in moe_losses)
        elif neox_args.moe_type == "megablocks":
            moe_loss = mb_moe_loss_func(neox_args, loss_mask, outputs)[0]
        else:
            raise ValueError(f"Unsupported moe_type: {neox_args.moe_type}")
    else:
        moe_loss = 0.0
    loss = main_loss + moe_loss
    if neox_args.memory_profiling:
        torch.cuda.nvtx.range_pop()
    if return_logits:
        return loss, outputs
    return loss


def get_model(neox_args, use_cache=False):
    """Build the model."""

    # Build model on cpu.
    print_rank_0("building GPT2 model ...")

    # Temporarily disable mup so that the base model does not use the mup init functions before set_base_shapes is called below.
    # If mup isn't being used anyways, this has no effect.
    old_use_mup = neox_args.use_mup
    neox_args.use_mup = False

    with deepspeed.zero.Init(
        config_dict_or_path=neox_args.deepspeed_config
    ) if neox_args.zero_stage == 3 else nullcontext() as gs:
        model = GPT2ModelPipe(
            neox_args=neox_args,
            num_tokentypes=0,
            parallel_output=True,
            topology=mpu.get_topology(),
            use_cache=use_cache,
        )

    ### soft prompt tuning stuff ###
    if neox_args.soft_prompt_tuning is not None and neox_args.soft_prompt_tuning.get(
        "enabled", False
    ):
        soft_prompt = SoftEmbedding(
            neox_args,
            wte=getattr(model, "0").word_embeddings,
            n_tokens=neox_args.soft_prompt_tuning.get("n_tokens", 10),
            init_string=neox_args.soft_prompt_tuning.get("init_string", ""),
            init_range=neox_args.soft_prompt_tuning.get("init_range", 0.5),
        )
        model.insert_layers(
            layers=soft_prompt, idx=1
        )  # insert the soft prompt layer directly after the word embeddings

        # freeze everything but the soft prompt
        for name, param in model.named_parameters():
            if not "soft_embedding" in name:
                param.requires_grad = False

    if not neox_args.is_pipe_parallel:
        # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
        model = model.to_sequential()

    neox_args.use_mup = old_use_mup

    if neox_args.use_mup:
        try:
            import mup
        except ModuleNotFoundError:
            print("Please install mup https://github.com/microsoft/mup")
            raise Exception

        base_shapes = f"{neox_args.base_shapes_file}.{torch.distributed.get_rank()}"

        if neox_args.save_base_shapes:
            save_base_shapes(neox_args, base_shapes, use_cache)

        mup.set_base_shapes(model, base_shapes)

        # Call the mup replacement init functions on the model now that set_base_shapes has given each weight a .infshape attribute
        mup_weights_reinit(neox_args, model)

    if neox_args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_optimizer(model, neox_args):
    """Set up the optimizer."""
    if neox_args.no_load_optim:
        return None, None

    if neox_args.optimizer is None:
        print_rank_0(
            f"ERROR: Optimizer is None. Either set the optimizer dict in your config (if training) or set no_load_optim in your config (if inference)"
        )
        exit()
    # Build parameter groups (weight decay and non-decay).
    param_groups = get_params_for_weight_decay_optimization(model, neox_args)
    print_rank_0(
        f'Configuring Optimizer type: {neox_args.optimizer_type} with params: {neox_args.optimizer["params"]}'
    )

    if neox_args.create_moe_param_group:
        from deepspeed.moe.utils import (
            is_moe_param,
            split_params_into_different_moe_groups_for_optimizer,
        )

        param_groups = split_params_into_different_moe_groups_for_optimizer(
            param_groups
        )

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group["params"]:
            if not hasattr(param, "model_parallel"):
                param.model_parallel = False

    # Filter out params that don't require a grad (for soft prompt tuning, etc.)
    _param_groups = []
    for param_group in param_groups:
        trainable_params = [p for p in param_group["params"] if p.requires_grad]
        param_group["params"] = trainable_params
        _param_groups.append(param_group)
    param_groups = _param_groups

    # If we're using mup, then the optimizer must be adam or sgd
    assert not neox_args.use_mup or (
        neox_args.optimizer_type.lower() == "adam"
        or neox_args.optimizer_type.lower() == "sgd"
    ), f"If use_mup == True, you must specify either the adam or sgd optimizers. You passed: {neox_args.optimizer_type.lower()}"

    if neox_args.optimizer_type.lower() in ["cpu_adam", "cpu_torch_adam"]:
        if neox_args.optimizer == "cpu_torch_adam":
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "onebitadam":
        assert neox_args.deepspeed
        optimizer = None
        # onebitadam needs to be instantiated within the deepspeed engine to work :|
    elif neox_args.optimizer_type.lower() == "sm3":
        from .optimizers import SM3

        optimizer = SM3(param_groups, **neox_args.optimizer["params"])
    elif neox_args.optimizer_type.lower() == "madgrad_wd":
        from .optimizers import madgrad_wd

        optimizer = madgrad_wd(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "lion":
        # if we want the deepspeed zero lion...megatron lion will throw DeepSpeed Error
        if neox_args.zero_optimization["stage"] != 0:
            from deepspeed.ops.lion import FusedLion

            lion_optimizer = FusedLion
        # if not zero
        else:
            from .optimizers import Lion

            lion_optimizer = Lion

        optimizer = lion_optimizer(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "adam":
        # Use Adam
        if neox_args.use_mup:
            try:
                from mup import MuAdam

                adam_optimizer = MuAdam
            except ModuleNotFoundError:
                print("Please install mup https://github.com/microsoft/mup")
                raise Exception
        else:
            if neox_args.use_bnb_optimizer:
                try:
                    import bitsandbytes as bnb

                    adam_optimizer = bnb.optim.Adam8bit
                except ModuleNotFoundError:
                    print(
                        "Please install bitsandbytes following https://github.com/facebookresearch/bitsandbytes."
                    )
                    raise Exception
            else:
                try:
                    # default to apex as it's slightly faster
                    from apex.optimizers import FusedAdam as Adam
                except ImportError:
                    # if apex isn't installed, use deepspeed's FusedAdam
                    print(
                        "WARNING: APEX not installed - defaulting to deepspeed's fused adam"
                    )
                    from deepspeed.ops.adam import FusedAdam as Adam
                adam_optimizer = Adam
        optimizer = adam_optimizer(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "sgd":
        try:
            from mup import MuSGD
        except ModuleNotFoundError:
            print("Please install mup https://github.com/microsoft/mup")
            raise Exception
        optimizer = MuSGD(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    else:
        raise ValueError(f"Optimizer type {neox_args.optimizer_type} not recognized")

    if neox_args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer, param_groups
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_learning_rate_scheduler(optimizer, neox_args):
    """Build the learning rate scheduler."""
    if neox_args.no_load_optim:
        # TODO: this should be configured as a separate arg
        return None
    if neox_args.deepspeed and neox_args.optimizer_type.lower() == "onebitadam":
        print_rank_0(
            "WARNING: onebitadam requires the lr scheduler be built by deepspeed - "
            "Make sure one is added to your deepspeed config"
        )
        return None

    # Add linear learning rate scheduler.
    if neox_args.lr_decay_iters is not None:
        num_iters = neox_args.lr_decay_iters
    else:
        num_iters = neox_args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = neox_args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=neox_args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=neox_args.lr_decay_style,
        last_iter=init_step,
        min_lr=neox_args.min_lr,
        use_checkpoint_lr_scheduler=neox_args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=neox_args.override_lr_scheduler,
        use_mup=neox_args.use_mup,
    )

    return lr_scheduler


def setup_model_and_optimizer(neox_args, use_cache=False, iteration=None):
    """Setup memory profiler"""
    if neox_args.memory_profiling:
        torch.cuda.memory._record_memory_history(
            True,
            # keep a maximum 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            trace_alloc_record_context=True,
        )

    """Setup model and optimizer."""
    model = get_model(neox_args=neox_args, use_cache=use_cache)
    optimizer, param_groups = get_optimizer(model=model, neox_args=neox_args)
    lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, neox_args=neox_args)

    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        if neox_args.no_load_optim:
            assert optimizer is None
            _model_params = None
            _lr_scheduler = None
        else:
            _model_params = param_groups if optimizer is None else None
            _lr_scheduler = lr_scheduler

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=neox_args,
            lr_scheduler=_lr_scheduler,
            dist_init_required=False,
            model_parameters=_model_params,
            # Need to remove the below so that it doesn't conflict with --deepspeed_config required by autotuning
            # config_params=neox_args.deepspeed_config,
            mpu=mpu if not neox_args.is_pipe_parallel else None,
        )
        if neox_args.moe_num_experts > 1 and neox_args.moe_type == "megablocks":
            # We need to additionally set this flag to ensure DS parallelism properly handles this foreign MoE.
            model.has_moe_layers = True
        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

        if neox_args.is_pipe_parallel:
            model.set_has_attention_mask(True)
            if neox_args.curriculum_learning:
                curr_scheduler = CurriculumScheduler(neox_args.curriculum_learning)
                if iteration is not None and iteration > 0:
                    curr_scheduler.update_difficulty(iteration)
            else:
                curr_scheduler = None
            model.set_batch_fn(
                partial(
                    get_batch_pipe, neox_args=neox_args, curr_scheduler=curr_scheduler
                )
            )
        else:
            model.module.set_batch_fn(
                partial(get_batch_sequential, neox_args=neox_args)
            )

    else:
        raise ValueError("Must be using deepspeed to run neox")

    if neox_args.load is not None:
        neox_args.iteration = load_checkpoint(
            neox_args=neox_args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            iteration=iteration,
        )
        print_rank_0(
            f"Loading checkpoint and starting from iteration {neox_args.iteration}"
        )
    else:
        neox_args.iteration = 0

    # need this for correct lr scheduling resume from ckpt
    # but it will not exist if this is being called for inference
    if lr_scheduler is not None:
        lr_scheduler.optimizer = model.optimizer

    return model, optimizer, lr_scheduler


def backward_step(neox_args, timers, optimizer, model, loss):
    """Backward step."""

    # Backward pass.
    timers("backward-backward").start()
    if neox_args.deepspeed:
        model.backward(loss)
    else:
        raise ValueError("Must be using deepspeed to run neox")
    timers("backward-backward").stop()

    if neox_args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers("backward-allreduce").reset()
    else:
        raise ValueError("Must be using deepspeed to run neox")


def train_step(neox_args, timers, data_iterator, model, optimizer, lr_scheduler):
    """Single training step."""

    # Pipeline parallelism schedules forward/backward/step
    if neox_args.is_pipe_parallel:
        reduced_loss = train_step_pipe(
            neox_args=neox_args, timers=timers, model=model, data_iterator=data_iterator
        )
        if (
            neox_args.memory_profiling
            and neox_args.iteration >= neox_args.profile_step_start
            and neox_args.iteration <= neox_args.profile_step_stop
            and torch.distributed.get_rank() == 0
        ):
            save_snapshot(neox_args)
    else:
        losses = []
        for _ in range(neox_args.gradient_accumulation_steps):
            # Forward model for one step.
            timers("forward").start()
            loss = forward_step(
                neox_args=neox_args,
                timers=timers,
                data_iterator=data_iterator,
                model=model,
                is_train=True,
            )
            timers("forward").stop()
            losses.append(loss)
            # Calculate gradients, reduce across processes, and clip.
            if (
                neox_args.profile
                and neox_args.iteration >= neox_args.profile_step_start
                and neox_args.iteration <= neox_args.profile_step_stop
            ):
                torch.cuda.nvtx.range_push(f"Backward pass")
            timers("backward").start()
            backward_step(
                neox_args=neox_args,
                timers=timers,
                optimizer=optimizer,
                model=model,
                loss=loss,
            )
            timers("backward").stop()
            if (
                neox_args.profile
                and neox_args.iteration >= neox_args.profile_step_start
                and neox_args.iteration <= neox_args.profile_step_stop
            ):
                torch.cuda.nvtx.range_pop()
            # Update parameters.
            if (
                neox_args.profile
                and neox_args.iteration >= neox_args.profile_step_start
                and neox_args.iteration <= neox_args.profile_step_stop
            ):
                torch.cuda.nvtx.range_push(f"Optimizer step")
            timers("optimizer").start()
            if neox_args.deepspeed:
                model.step()
            else:
                raise ValueError("Must be using deepspeed to run neox")
            timers("optimizer").stop()
            if (
                neox_args.profile
                and neox_args.iteration >= neox_args.profile_step_start
                and neox_args.iteration <= neox_args.profile_step_stop
            ):
                torch.cuda.nvtx.range_pop()
            if (
                neox_args.profile
                and neox_args.iteration >= neox_args.profile_step_start
                and neox_args.iteration <= neox_args.profile_step_stop
                and torch.distributed.get_rank() == 0
            ):
                save_snapshot(neox_args)
        reduced_loss = {
            "lm_loss": reduce_losses(losses).mean()
        }  # reduces losses across machines for logging

    if neox_args.precision == "fp16" and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    collect_loss_for_unit_test(reduced_loss["lm_loss"])
    return reduced_loss, skipped_iter


def train_step_pipe(neox_args, timers, model, data_iterator):
    """Single training step with DeepSpeed's pipeline parallel engine."""

    assert neox_args.deepspeed
    loss = model.train_batch(data_iter=data_iterator)
    loss_dict = {"lm_loss": loss}
    # Don't break Megatron's timers because we changed code paths.
    for t in [
        "forward",
        "backward",
        "allreduce",
        "optimizer",
        "batch generator",
        "data loader",
    ]:
        timers(t).reset()
    return loss_dict


def train(
    neox_args,
    timers,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator,
):
    """Train the model function."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = neox_args.iteration

    timers("interval time").start()
    report_memory_flag = True

    # get noise scale logger (if neox_args.log_gradient_noise_scale is True)
    noise_scale_logger = get_noise_scale_logger(neox_args)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)

    if neox_args.profile:
        schedule = torch.profiler.schedule(
            wait=neox_args.profile_step_start,
            warmup=1,
            active=neox_args.profile_step_stop - neox_args.profile_step_start,
        )
        prof = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                neox_args.tensorboard_dir
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
        )
        prof.start()
    while iteration < neox_args.train_iters:
        if neox_args.profile:
            prof.step()
        if neox_args.profile and iteration == neox_args.profile_step_start:
            torch.cuda.cudart().cudaProfilerStart()
        loss_dict, skipped_iter = train_step(
            neox_args=neox_args,
            timers=timers,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        if neox_args.profile and iteration == neox_args.profile_step_stop:
            torch.cuda.cudart().cudaProfilerStop()
            prof.stop()
        iteration += 1
        neox_args.iteration = iteration
        if neox_args.precision == "fp16":
            overflow_monitor.check(skipped_iter)  # check for repeated overflow
        if neox_args.log_gradient_noise_scale:  # log noise scale if applicable
            noise_scale_logger.update()

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # Logging.
        report_memory_flag = training_log(
            neox_args=neox_args,
            timers=timers,
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=lr,
            iteration=iteration,
            loss_scale=optimizer.cur_scale if neox_args.precision == "fp16" else None,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            model=model,
            optimizer=optimizer,
            noise_scale_logger=noise_scale_logger,
        )

        # Checkpointing
        if neox_args.save and iteration in neox_args.save_iters:
            save_checkpoint(
                neox_args=neox_args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
        # Evaluation
        if (
            neox_args.eval_interval
            and iteration % neox_args.eval_interval == 0
            and neox_args.do_valid
        ):
            prefix = "iteration {}".format(iteration)
            evaluate_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterator=valid_data_iterator,
                model=model,
                iteration=iteration,
                verbose=False,
                timers=timers,
            )

        if neox_args.exit_interval and iteration % neox_args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rank = torch.distributed.get_rank()
            print_rank_0(
                "rank: {} | time: {} | exiting the program at iteration {}".format(
                    rank, time_str, iteration
                )
            )
            sys.exit()

    return iteration


def evaluate(
    neox_args, forward_step_fn, data_iterator, model, verbose=False, timers=None
):
    """Evaluation.
    neox_args: NeoX Arguments
    forward_step_fn: function with args `neox_args, timers,
                    data_iterator & model that will run a forward pass on the model
    data_iterator: Iterator that iterates over batches of data. Should return data in the form:
                    {'text': np.array([tokens], dtype=np.int64)}
                    where the size of the array is the model's context size + 1
                    (`get_batch` transforms it into inputs / labels)
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    losses = []
    if neox_args.char_level_ppl:
        data_iterator = CharCounter(data_iterator, neox_args.tokenizer)

    with torch.no_grad():
        iteration = 0
        while iteration < neox_args.eval_iters:
            iteration += 1
            if verbose and iteration % neox_args.log_interval == 0:
                print_rank_0(
                    "Evaluating iter {}/{}".format(iteration, neox_args.eval_iters)
                )

            # although we're not accumulating gradients here, we count one iter as train_batch_size_per_gpu * g.a.s
            # to be consistent with deepspeed's pipe parallel engine
            # since pipe parallel already takes gradient_accumulation_steps into account - default to 1 here if pipe parallel is true
            for _ in range(
                1
                if neox_args.is_pipe_parallel
                else neox_args.gradient_accumulation_steps
            ):
                # Forward evaluation
                loss = forward_step_fn(
                    model=model,
                    data_iterator=data_iterator,
                    neox_args=neox_args,
                    timers=timers,
                )
                losses.append(loss)

            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

    # reduces losses across processes for logging & run eval harness tasks
    eval_results = {"lm_loss": reduce_losses(losses).mean().item()}
    eval_results["lm_loss_ppl"] = math.exp(eval_results["lm_loss"])

    if neox_args.char_level_ppl:
        # calculate character level perplexity, if specified
        # if neox_args.char_level_ppl:
        # unwrap the data_iterator
        tokens_per_char = data_iterator.tokens_per_char()
        print_rank_0(f"Counting chars took {data_iterator.total_time} seconds")

        data_iterator = data_iterator.data_iterator
        eval_results["lm_loss_char_lvl_ppl"] = math.exp(
            eval_results["lm_loss"] * tokens_per_char
        )

    if neox_args.eval_tasks:
        from eval_tasks import run_eval_harness

        eval_results.update(
            run_eval_harness(
                model, forward_step_fn, neox_args, eval_tasks=neox_args.eval_tasks
            ).get("results")
        )
    # Move model back to the train mode.
    model.train()
    return eval_results


def collect_loss_for_unit_test(lm_ss):
    # Logic moved to separate function to allow tracking in unit tests with unittest.mock.patch
    pass


def evaluate_and_print_results(
    neox_args,
    prefix,
    forward_step_func,
    data_iterator,
    model,
    iteration,
    verbose=False,
    timers=None,
    chart_name="validation",
):
    """Helper function to evaluate and dump results on screen."""
    total_loss_dict = evaluate(
        neox_args=neox_args,
        forward_step_fn=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        verbose=verbose,
        timers=timers,
    )
    string = f" {chart_name} results at {prefix} | "
    for k, v in total_loss_dict.items():
        if isinstance(v, dict):
            if neox_args.eval_tasks and "results" in v:
                v = v["results"]
                print(v)
            for k2, v2 in v.items():
                k3 = "_".join([k, k2])
                string += f"{k3} value: {v2:.6E} | "
                tb_wandb_log(
                    f"{chart_name}/{k3}",
                    v2,
                    iteration,
                    use_wandb=neox_args.use_wandb,
                    tensorboard_writer=neox_args.tensorboard_writer,
                )
        else:
            string += f"{k} value: {v:.6E} | "
            tb_wandb_log(
                f"{chart_name}/{k}",
                v,
                iteration,
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
            )

    length = len(string) + 1
    print_rank_0("-" * length)
    print_rank_0(string)
    print_rank_0("-" * length)


def save_snapshot(neox_args):
    assert (
        neox_args.memory_profiling_path is not None
    ), "Must pass memory_profiling_path config arg to use profiling"
    snapshot = torch.cuda.memory._snapshot()
    snapshot_path = os.path.join(neox_args.memory_profiling_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    with open(os.path.join(snapshot_path, "mem_snapshot.pickle"), "wb") as f:
        dump(snapshot, f)
