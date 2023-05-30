from .model import Wav2Vec2Model


def create_model(args):

    model = Wav2Vec2Model(args)

    return model