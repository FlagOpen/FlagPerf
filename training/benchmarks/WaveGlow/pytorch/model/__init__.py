from . import models

def create_model(config):
    config.model_config = models.get_model_config(config.name, config)
    model = models.get_model(config.name, config.model_config,
                             cpu_run=False,
                             uniform_initialize_bn_weight=not config.disable_uniform_initialize_bn_weight)
    
    return model