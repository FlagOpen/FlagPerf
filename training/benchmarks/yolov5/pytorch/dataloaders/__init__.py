from .dataloader import (build_train_dataset, build_eval_dataset,
                         build_train_dataloader, build_eval_dataloader)
import config
import yaml

hyp = config.hyp
if isinstance(hyp, str):
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict