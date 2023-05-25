from wav2vec2.model import Wav2Vec2Model
from common.utils import AttrDict

def create_model(args, target_dictionary=None):

    # cfg = AttrDict(vars(args))
    cfg=args
    assert target_dictionary is None
    model = Wav2Vec2Model(cfg)

    # sequence_generator = None
    # tokenizer = None

    # actualized_cfg = getattr(model, "cfg", None)
    # print("2121212",actualized_cfg)
    # if actualized_cfg is not None and "w2v_args" in actualized_cfg:
    #     cfg.w2v_args = actualized_cfg.w2v_args

    return model