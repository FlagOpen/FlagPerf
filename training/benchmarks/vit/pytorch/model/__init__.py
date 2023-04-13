import timm

def create_model(model_name):
    model = timm.models.create_model(
        'vit_base_patch16_224',
        # in_chans=in_chans,
        # num_classes=args.num_classes,
        # drop_rate=args.drop,
        # drop_path_rate=args.drop_path,
        # drop_block_rate=args.drop_block,
        # global_pool=args.gp,
        # bn_momentum=args.bn_momentum,
        # bn_eps=args.bn_eps,
        # scriptable=args.torchscript,
        # checkpoint_path=args.initial_checkpoint,
        # **args.model_kwargs,
    )    
 
    return model
