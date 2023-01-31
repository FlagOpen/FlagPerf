import torch
import torch.backends.cudnn as cudnn


class Device(object):
    
    @staticmethod
    def get_device(args):
        device = None
        if torch.cuda.is_available():
            if args.gpu:
                device = torch.device('cuda:{}'.format(args.gpu))
            else:
                device = torch.device("cuda")
        elif hasattr(torch.backends, "mps")  and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")    

        return device 