import torch


class Device(object):
    """device"""

    @staticmethod
    def get_device(args):
        """get device"""
        device = None
        if torch.cuda.is_available():
            if args.gpu:
                device = torch.device("cuda:{}".format(args.gpu))
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        return device
