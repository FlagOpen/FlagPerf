import os
import time
from loguru import logger
import sys
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import default_data_collator

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH,
                                             "../../")))  # benchmarks目录
from driver.helper import InitHelper
from driver import dist_pytorch
from utils.memory_utils import MemoryTrace
from evaluate import evaluate_MMLU
from model import get_llama_model
from dataset import get_llama_dataset
from schedulers import create_scheduler
from optimizers import create_optimizer
import config


def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer,
          lr_scheduler, gradient_accumulation_steps, train_config):
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []
    epoch_times = []
    iter_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(colour="blue",
                        desc=f"Training Epoch: {epoch+1}",
                        total=total_length,
                        dynamic_ncols=True)
            torch.cuda.synchronize()
            iter_start = time.perf_counter()
            for step, batch in enumerate(train_dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to('cuda:0')
                with autocast():
                    loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                torch.cuda.synchronize()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    torch.cuda.synchronize()
                    if (step + 1
                        ) % gradient_accumulation_steps == 0 or step == len(
                            train_dataloader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1
                        ) % gradient_accumulation_steps == 0 or step == len(
                            train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(1)
                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
                )
                torch.cuda.synchronize()
                iter_end = time.perf_counter()
                iter_time = iter_end - iter_start
                iter_times.append(iter_time)

                iter_start = time.perf_counter()

            pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA devic
        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        # Update the learning rate as needed
        lr_scheduler.step()

        logger.info(f"Max CUDA memory allocated was {memtrace.peak} GB")
        logger.info(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
        logger.info(
            f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
        logger.info(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
        logger.info(
            f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"
        )

        if train_config.run_validation:
            local_rank = None
            eval_ppl, eval_epoch_loss = evaluation(model, train_config,
                                                   eval_dataloader, local_rank,
                                                   tokenizer)
            checkpoint_start_time = time.perf_counter()
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                if train_config.enable_fsdp:
                    dist.barrier()
                if train_config.use_peft:
                    print(f"we are about to save the PEFT modules")
                    model.save_pretrained(train_config.output_dir)
                    print(
                        f"PEFT modules are saved in {train_config.output_dir} directory"
                    )
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(best_val_loss)
            val_prep.append(eval_ppl)
        logger.info(
            f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
        )
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_iter_time = sum(iter_times) / (len(iter_times))
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(
        checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    mmlu_ret = evaluate_MMLU(model, train_config, tokenizer)
    if mmlu_ret >= train_config.target_MMLU:
        logger.info("MMLU: {} achieve  the target:{}.".format(
            train_config.mmlu_ret, train_config.target_MMLU))
    else:
        logger.warning("MMLU target is not achieved")
    return results, avg_iter_time


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(
                tqdm(eval_dataloader,
                     colour="green",
                     desc="evaluating Epoch",
                     dynamic_ncols=True)):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(),
                                       skip_special_tokens=True))

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank == 0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    return eval_ppl, eval_epoch_loss


def main():
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    train_config = config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    config.distributed = dist_pytorch.get_world_size() > 1

    model = get_llama_model(train_config)
    dataset_train, dataset_val, tokenizer = get_llama_dataset(train_config)

    train_sampler = None
    val_sampler = None
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    eval_dataloader = None
    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )

    optimizer = create_optimizer(model, train_config)
    scheduler = create_scheduler(optimizer, train_config)

    results, avg_iter_time = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
    )
    return results, train_config, avg_iter_time


if __name__ == "__main__":
    results, train_config, avg_iter_time = main()

    if config.local_rank == 0:
        tokens = train_config.seq_length * train_config.batch_size_training
        chip_tps = tokens / avg_iter_time / train_config.nproc / train_config.nnodes
        print("System tokens per second: ", tokens)
        print("avg_iter_time(s):", avg_iter_time)
        print("Tokens/p/s: ", chip_tps)
        print(
            "MFU: ",
            round(chip_tps * 7000000000.0 * 2 / train_config.theory_flops, 3))
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
