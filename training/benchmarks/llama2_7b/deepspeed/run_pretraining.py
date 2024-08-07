# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import json
import os
import ast
import argparse
import logging
import sys
from importlib import import_module

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed
import deepspeed.comm as dist

from model import get_llama_model
from dataset import get_llama_dataset


class MyLogHandler(logging.Handler, object):

    def __init__(self):
        logging.Handler.__init__(self)
        self.texts = []

    def emit(self, record):
        msg = self.format(record)
        if 'RunningAvgSamples' in msg:
            self.texts.append(msg)


def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Reserved for deepspeed framework")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--flagperf_config", type=str)
    parser.add_argument("--node_rank",
                        type=int,
                        required=True,
                        help="The rank of the node for multi-node distributed training.")
    parser.add_argument("--nnodes",
                        type=int,
                        required=True,
                        help="how many hosts to run the testcase.")
    parser.add_argument("--nproc_per_node",
                        type=int,
                        required=True,
                        help="how many processes will run on each host.")
    return parser


def train(model_engine, dataloader):
    model_engine.train()
    ave_loss = 0.0
    for step, data in enumerate(dataloader):

        fake_data = torch.tensor(data).long()
        input_ids = fake_data.to(args.local_rank)
        labels = fake_data.to(args.local_rank)
        loss = model_engine(input_ids=input_ids, labels=labels).loss
        model_engine.backward(loss)
        model_engine.step()

        ave_loss += loss
        if step % 10 == 0 and args.local_rank == 0:
            print('Step {}/{}, Loss: {}'.format(step, len(dataloader),
                                                ave_loss / 10))
            ave_loss = 0.0


def get_deepspeed_engine(args, model_config_dir, flashattn):
    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config,
                             enabled=True,
                             mem_efficient_linear=False,
                             mpu=None):
        model = get_llama_model(model_config_dir, flashattn)

    model_engine, _, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters())
    return model_engine


def get_metric(texts):
    msg = texts[-1]
    meaningful_msg = msg.split('RunningAvgSamplesPerSec=')[1]
    pure_msg = meaningful_msg.split(',')[0]
    return float(pure_msg)


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    arg_parser = deepspeed.add_config_arguments(arg_parser)
    args = arg_parser.parse_args()

    flagperf_config = {}
    sys.path.append(os.path.dirname(args.flagperf_config))
    config_file = os.path.basename(args.flagperf_config).split('.')[0]

    module = import_module(config_file)

    seqlength = getattr(module, 'seqlength')
    batchsize = getattr(module, 'batchsize')
    datafilename = getattr(module, 'datafilename')
    theoryflops = getattr(module, 'theoryflops')
    epochs = getattr(module, 'epochs')
    flashattn = getattr(module, 'flashattn')

    deepspeed.init_distributed()
    model_engine = get_deepspeed_engine(args, os.path.join("llama2_7b_hf"),
                                        flashattn)
    dataset = get_llama_dataset(args, seqlength, datafilename)

    logger = logging.getLogger("DeepSpeed")
    handler = MyLogHandler()
    logger.addHandler(handler)

    sampler = DistributedSampler(dataset,
                                 num_replicas=args.nproc_per_node * args.nnodes,
                                 rank=args.local_rank)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=batchsize,
                            pin_memory=True)

    epoch = 0
    while epoch < epochs:
        sampler.set_epoch(epoch)
        train(model_engine, dataloader)
        epoch += 1

        if args.local_rank == 0:
            tokens = seqlength * batchsize
            perf = get_metric(handler.texts)
            whole_tps = tokens * perf
            chip_tps = whole_tps / args.nproc_per_node * args.nnodes
            print("System tokens per second: ", whole_tps)
            print("Tokens/p/s: ", chip_tps)

            TFLOPS = int(theoryflops/1000000000000)
            print("Theory TFLOPS: ", TFLOPS)
            print("Tokens/TFLOPS: ", chip_tps / TFLOPS)
            print("MFU: ", chip_tps * 7000000000.0 * 6 / theoryflops)
