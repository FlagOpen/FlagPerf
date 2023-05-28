from utils.datasets import create_dataloader


# TODO
def build_train_dataloader():
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=True, cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect, rank=LOCAL_RANK, workers=workers,
                                              image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True)
    return train_loader, dataset 

# TODO
def build_eval_dataloader():
    val_loader, dataset = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache,
                                       rect=True, rank=-1, workers=workers * 2, pad=0.5,
                                       prefix=colorstr('val: '))
    return val_loader, dataset


