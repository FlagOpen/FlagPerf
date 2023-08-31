import os

import torch
from loguru import logger

from flagai.auto_model.auto_loader import AutoLoader


def create_model(config):
    logger.info("build model...")
    if os.path.exists(config.download_path):
        model_dict = torch.load(config.download_path)
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
    else:
        loader = AutoLoader(
        "lm",
        model_dir=config.state_dict,
        model_name=config.model_name,
        use_cache=True,
        fp16=True)
        model = loader.get_model()
        tokenizer = loader.get_tokenizer()
        torch.save({"model":model, "tokenizer":tokenizer}, config.download_path)

    model.cuda()
    model.eval()
    if config.fp16:
        model.half()

    return model
