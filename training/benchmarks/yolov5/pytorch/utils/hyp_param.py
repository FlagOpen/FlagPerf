import yaml
import config

# Hyperparameters
if isinstance(config.hyp, str):
    with open(config.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict