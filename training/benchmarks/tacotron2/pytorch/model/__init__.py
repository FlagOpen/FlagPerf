from .models.model_helper import get_model_config, get_model


def create_model(args):
    model_config = get_model_config(args)
    uniform_initialize_bn_weight = not args.disable_uniform_initialize_bn_weight
    model = get_model(
        model_config,
        cpu_run=False,
        uniform_initialize_bn_weight=uniform_initialize_bn_weight)
    return model


def create_model_config(args):
    model_config = get_model_config(args)
    return model_config