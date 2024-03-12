import os
import sys
import torch
import numpy as np
from functools import partial

from torch.cuda.amp import autocast

from misc.utils import check_validity, convert_predictions_to_mols, predictions_to_smiles, check_novelty
from runtime.common import get_newest_checkpoint, load_state
from data.data_loader import NumpyTupleDataset
from data import transform
from runtime.generate import infer
from model.model import MoFlow

# add benchmarks directory to sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../../")))
from driver import dist_pytorch


class Evaluator:

    def __init__(self):
        pass

    def evaluate(self, args, config, acc_logger):
        device = args.device
        snapshot_path = get_newest_checkpoint(args.results_dir)
        args.jit = True
        model = MoFlow(config)

        dist_pytorch.main_proc_print(f"snapshot_path: {snapshot_path}")

        if snapshot_path is not None:
            epoch, ln_var = load_state(snapshot_path, model, device=device)
        elif args.allow_untrained:
            epoch, ln_var = 0, 0
        else:
            raise RuntimeError(
                'Generating molecules from an untrained network! '
                'If this was intentional, pass --allow_untrained flag.')

        model.to(device)
        model.eval()

        if args.steps == -1:
            args.steps = 1
        else:
            args.steps = 1000

        dist_pytorch.main_proc_print(f"ln_var ===> {ln_var}. args.steps:{args.steps}")

        valid_idx = transform.get_val_ids(config, args.data_dir)
        dataset = NumpyTupleDataset.load(
            os.path.join(args.data_dir, config.dataset_config.dataset_file),
            transform=partial(transform.transform_fn, config=config),
        )
        train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        train_x = torch.Tensor(np.array([a[0] for a in train_dataset]))
        train_adj = torch.Tensor(np.array([a[1] for a in train_dataset]))

        train_smiles = set(predictions_to_smiles(train_adj, train_x, config))

        metrics = dict()
        with autocast(enabled=args.amp):
            for i in range(args.steps):
                results = infer(model,
                                config,
                                ln_var=ln_var,
                                temp=args.temperature,
                                batch_size=args.train_batch_size,
                                device=device)

                mols_batch = convert_predictions_to_mols(
                    *results,
                    correct_validity=args.correct_validity,
                )
                validity_info = check_validity(mols_batch)
                novel_r, abs_novel_r = check_novelty(
                    validity_info['valid_smiles'],
                    train_smiles,
                    len(mols_batch),
                )
                _, nuv = check_novelty(
                    list(set(validity_info['valid_smiles'])),
                    train_smiles,
                    len(mols_batch),
                )
                metrics = {
                    'validity': validity_info['valid_ratio'],
                    'novelty': novel_r,
                    'uniqueness': validity_info['unique_ratio'],
                    'abs_novelty': abs_novel_r,
                    'abs_uniqueness': validity_info['abs_unique_ratio'],
                    'nuv': nuv,
                }
                dist_pytorch.main_proc_print("metrics", i, metrics)
                acc_logger.update(metrics)

        stats = acc_logger.summarize(step=tuple())
        return stats
