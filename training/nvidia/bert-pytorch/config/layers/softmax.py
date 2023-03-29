import torch
import ext_ops

###########################################################################################


class FastSoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(cxt, input, dim, batch, seqlen, heads, stream, sync, timers):
        if timers: timers['start_fprop'].record()
        ext_ops.FastSoftmaxFprop(input, batch, seqlen, heads, stream, sync)
        if timers: timers['stop_fprop'].record()

        cxt.save_for_backward(input, seqlen)
        cxt.dim = dim
        cxt.batch = batch
        cxt.heads = heads
        cxt.stream = stream
        cxt.sync = sync
        cxt.timers = timers
        return input

    @staticmethod
    def backward(cxt, grad_output):
        output, seqlen, = cxt.saved_tensors
        dim = cxt.dim
        batch = cxt.batch
        heads = cxt.heads

        if cxt.timers: cxt.timers['start_dgrad'].record()
        ext_ops.FastSoftmaxBprop(output, grad_output, batch, seqlen, heads,
                                 cxt.stream, cxt.sync)
        if cxt.timers: cxt.timers['stop_dgrad'].record()
        return grad_output, None, None, None, None, None, None, None


class FastSoftmax(torch.nn.Module):

    def __init__(self, dim=None, stream=True, sync=True, timer=False):
        super(FastSoftmax, self).__init__()
        self.dim = dim
        self.stream = stream
        self.sync = sync
        if timer:
            self.timers = {
                'start_fprop': torch.cuda.Event(enable_timing=True),
                'start_dgrad': torch.cuda.Event(enable_timing=True),
                'stop_fprop': torch.cuda.Event(enable_timing=True),
                'stop_dgrad': torch.cuda.Event(enable_timing=True)
            }
        else:
            self.timers = None

    def forward(self, input, batch, seqlen, heads):
        return FastSoftmaxFunction.apply(input, self.dim, batch, seqlen, heads,
                                         self.stream, self.sync, self.timers)


###########################################################################################


class FastMaskSoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(cxt, input, mask, dim, batch, seqlen, heads, stream, sync,
                timers):
        if timers: timers['start_fprop'].record()
        ext_ops.FastMaskSoftmaxFprop(input, mask, batch, seqlen, heads, stream,
                                     sync)
        if timers: timers['stop_fprop'].record()

        cxt.save_for_backward(input, seqlen)
        cxt.dim = dim
        cxt.batch = batch
        cxt.heads = heads
        cxt.stream = stream
        cxt.sync = sync
        cxt.timers = timers
        return input

    @staticmethod
    def backward(cxt, grad_output):
        output, seqlen, = cxt.saved_tensors
        dim = cxt.dim
        batch = cxt.batch
        heads = cxt.heads

        if cxt.timers: cxt.timers['start_dgrad'].record()
        ext_ops.FastSoftmaxBprop(output, grad_output, batch, seqlen, heads,
                                 cxt.stream, cxt.sync)
        if cxt.timers: cxt.timers['stop_dgrad'].record()
        return grad_output, None, None, None, None, None, None, None, None, None, None, None


class FastMaskSoftmax(torch.nn.Module):

    def __init__(self, dim=None, stream=True, sync=True, timer=False):
        super(FastMaskSoftmax, self).__init__()
        self.dim = dim
        self.stream = stream
        self.sync = sync
        if timer:
            self.timers = {
                'start_fprop': torch.cuda.Event(enable_timing=True),
                'start_dgrad': torch.cuda.Event(enable_timing=True),
                'stop_fprop': torch.cuda.Event(enable_timing=True),
                'stop_dgrad': torch.cuda.Event(enable_timing=True)
            }
        else:
            self.timers = None

    def forward(self, input, mask, batch, seqlen, heads):
        return FastMaskSoftmaxFunction.apply(input, mask, self.dim, batch,
                                             seqlen, heads, self.stream,
                                             self.sync, self.timers)


###########################################################################################


class FastMaskSoftmaxDropoutFunction(torch.autograd.Function):

    @staticmethod
    def forward(cxt, input, mask, dim, batch, seqlen, heads, dropout_prob,
                stream, sync, timers, is_training):
        if timers: timers['start_fprop'].record()
        output, dropout_mask, = ext_ops.FastMaskSoftmaxDropoutFprop(
            input, mask, batch, seqlen, heads, dropout_prob, stream, sync,
            is_training)
        if timers: timers['stop_fprop'].record()

        cxt.save_for_backward(input, dropout_mask, seqlen)
        cxt.dim = dim
        cxt.batch = batch
        cxt.heads = heads
        cxt.dropout_prob = dropout_prob
        cxt.stream = stream
        cxt.sync = sync
        cxt.timers = timers
        return output

    @staticmethod
    def backward(cxt, grad_output):
        output, dropout_mask, seqlen, = cxt.saved_tensors
        dim = cxt.dim
        batch = cxt.batch
        heads = cxt.heads
        dropout_prob = cxt.dropout_prob

        if cxt.timers: cxt.timers['start_dgrad'].record()
        ext_ops.FastMaskSoftmaxDropoutBprop(output, grad_output, dropout_mask,
                                            batch, seqlen, heads, dropout_prob,
                                            cxt.stream, cxt.sync)
        if cxt.timers: cxt.timers['stop_dgrad'].record()
        return grad_output, None, None, None, None, None, None, None, None, None, None, None, None, None


class FastMaskSoftmaxDropout(torch.nn.Module):

    def __init__(self,
                 dim=None,
                 dropout_prob=None,
                 stream=True,
                 sync=True,
                 timer=False):
        super(FastMaskSoftmaxDropout, self).__init__()
        self.dim = dim
        self.dropout_prob = dropout_prob
        self.stream = stream
        self.sync = sync
        if timer:
            self.timers = {
                'start_fprop': torch.cuda.Event(enable_timing=True),
                'start_dgrad': torch.cuda.Event(enable_timing=True),
                'stop_fprop': torch.cuda.Event(enable_timing=True),
                'stop_dgrad': torch.cuda.Event(enable_timing=True)
            }
        else:
            self.timers = None

    def forward(self, input, mask, batch, seqlen, heads, is_training):
        return FastMaskSoftmaxDropoutFunction.apply(input, mask, self.dim,
                                                    batch, seqlen, heads,
                                                    self.dropout_prob,
                                                    self.stream, self.sync,
                                                    self.timers, is_training)


###########################################################################################
