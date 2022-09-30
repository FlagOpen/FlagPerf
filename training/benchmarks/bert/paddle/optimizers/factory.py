import paddle


def create_optimizer(name: str, params, config,decay_params,lr_scheduler):
    name = name.lower()
    if name == "lamb":
        return paddle.optimizer.Lamb(
            parameters=params, learning_rate=lr_scheduler,
            beta1=config.opt_lamb_beta_1, beta2=config.opt_lamb_beta_2, epsilon=1e-6,
            #lamb_weight_decay=config.weight_decay_rate, 
            # #exclude_from_weight_decay_fn=lambda x: x in decay_params,
            )
        raise Exception("Not Implementation Lamb Error")

    if name == "adamw":
        return paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=params,
        #weight_decay=config.weight_decay_rate,
        beta1=config.opt_lamb_beta_1, beta2= config.opt_lamb_beta_2,
        epsilon=1e-6
        #apply_decay_param_fun=lambda x: x in decay_params, 
        )

    raise RuntimeError(f"Not found optimier {name}.")
