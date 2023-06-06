import torch
import torch.distributed as dist
import os

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity


def adjust_learning_rate(iteration, epoch, optimizer, learning_rate,
                         anneal_steps, anneal_factor, rank):

    p = 0
    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p + 1

    if anneal_factor == 0.3:
        lr = learning_rate * ((0.1**(p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate * (anneal_factor**p)

    if optimizer.param_groups[0]['lr'] != lr:
        DLLogger.log(step=(epoch, iteration),
                     data={
                         'learning_rate changed':
                         str(optimizer.param_groups[0]['lr']) + " -> " +
                         str(lr)
                     })

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(model, optimizer, scaler, epoch, config, output_dir,
                    model_name, local_rank, world_size):

    random_rng_state = torch.random.get_rng_state().cuda()
    cuda_rng_state = torch.cuda.get_rng_state(local_rank).cuda()

    random_rng_states_all = [
        torch.empty_like(random_rng_state) for _ in range(world_size)
    ]
    cuda_rng_states_all = [
        torch.empty_like(cuda_rng_state) for _ in range(world_size)
    ]

    if world_size > 1:
        dist.all_gather(random_rng_states_all, random_rng_state)
        dist.all_gather(cuda_rng_states_all, cuda_rng_state)
    else:
        random_rng_states_all = [random_rng_state]
        cuda_rng_states_all = [cuda_rng_state]

    random_rng_states_all = torch.stack(random_rng_states_all).cpu()
    cuda_rng_states_all = torch.stack(cuda_rng_states_all).cpu()

    if local_rank == 0:
        checkpoint = {
            'epoch': epoch,
            'cuda_rng_state_all': cuda_rng_states_all,
            'random_rng_states_all': random_rng_states_all,
            'config': config,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }

        checkpoint_filename = "checkpoint_{}_{}.pt".format(model_name, epoch)
        checkpoint_path = os.path.join(output_dir, checkpoint_filename)
        print("Saving model and optimizer state at epoch {} to {}".format(
            epoch, checkpoint_path))
        torch.save(checkpoint, checkpoint_path)

        symlink_src = checkpoint_filename
        symlink_dst = os.path.join(output_dir,
                                   "checkpoint_{}_last.pt".format(model_name))
        if os.path.exists(symlink_dst) and os.path.islink(symlink_dst):
            print("Updating symlink", symlink_dst, "to point to", symlink_src)
            os.remove(symlink_dst)

        os.symlink(symlink_src, symlink_dst)


def get_last_checkpoint_filename(output_dir, model_name):
    symlink = os.path.join(output_dir,
                           "checkpoint_{}_last.pt".format(model_name))
    if os.path.exists(symlink):
        print("Loading checkpoint from symlink", symlink)
        return os.path.join(output_dir, os.readlink(symlink))
    else:
        print("No last checkpoint available - starting from epoch 0 ")
        return ""


def load_checkpoint(model, optimizer, scaler, epoch, filepath, local_rank):

    checkpoint = torch.load(filepath, map_location='cpu')

    epoch[0] = checkpoint['epoch'] + 1
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state_all'][device_id])
    if 'random_rng_states_all' in checkpoint:
        torch.random.set_rng_state(
            checkpoint['random_rng_states_all'][device_id])
    elif 'random_rng_state' in checkpoint:
        torch.random.set_rng_state(checkpoint['random_rng_state'])
    else:
        raise Exception(
            "Model checkpoint must have either 'random_rng_state' or 'random_rng_states_all' key."
        )
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    return checkpoint['config']


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if rt.is_floating_point():
        rt = rt / num_gpus
    else:
        rt = torch.div(rt, num_gpus, rounding_mode='floor')
    return rt


def init_dllogger(config):
    if config.local_rank == 0:
        # log_file = os.path.join(config.output, config.log_file)
        DLLogger.init(backends=[
            JSONStreamBackend(Verbosity.DEFAULT, config.log_file),
            StdOutBackend(Verbosity.VERBOSE)
        ])
    else:
        DLLogger.init(backends=[])

    DLLogger.log(step="PARAMETER", data={'model_name': 'WaveGlow'})
    DLLogger.metadata('run_time', {'unit': 's'})
    DLLogger.metadata('val_loss', {'unit': None})
    DLLogger.metadata('train_items_per_sec', {'unit': 'items/s'})
    DLLogger.metadata('val_items_per_sec', {'unit': 'items/s'})
    return DLLogger
