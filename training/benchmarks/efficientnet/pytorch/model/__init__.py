import torchvision


def create_model(config):
    model = torchvision.models.efficientnet_v2_s()
    return model
