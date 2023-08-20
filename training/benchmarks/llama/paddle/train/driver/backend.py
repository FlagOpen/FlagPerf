class Backend(object):
    NAME: str = ""

    def __init__(self) -> None:
        pass

    def synchronize(self):
        raise "Not implemented."

    def set_device(self, local_rank):
        raise "Not implemented."


class CudaBackend(Backend):
    NAME = 'CUDA'

    def __init__(self) -> None:
        super().__init__()

    def synchronize(self):
        import paddle
        paddle.cuda.synchronize()

    def set_device(self, rank):
        import paddle
        paddle.cuda.set_device(rank)
