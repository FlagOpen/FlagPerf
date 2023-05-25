from wav2vec2.model import Wav2Vec2Model


def create_model(args, target_dictionary=None):

    # cfg = AttrDict(vars(args))
    cfg=args
    assert target_dictionary is None
    model = Wav2Vec2Model(cfg)

    return model