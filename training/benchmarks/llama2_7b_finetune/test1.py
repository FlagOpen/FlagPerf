import numpy as np
import torch
import torch.nn as nn


grad_ins = None
num = 0


def module_hook1(module, grad_in, grad_out):
    assert len(grad_in) == 3
    assert grad_in[0].shape == (5, 3, 6, 6)
    assert grad_in[1].shape == (5, 3, 2, 2)
    assert grad_in[2].shape == (5,)
    assert len(grad_out) == 1
    assert grad_out[0].shape == (5, 5, 5, 5)
    print("3"*10)


def module_hook2(module, grad_in, grad_out):
    global grad_ins, num
    for g in grad_in:
        if g is not None:
            g = g + 1
    grad_ins = grad_in
    num += 1
    print("2"*10)



def module_hook3(module, grad_in, grad_out):
    global grad_ins
    for g1, g2 in zip(grad_in, grad_ins):
        if g1 is None and g2 is None:
            continue
        assert bool((g1 == g2).all())
        print("1"*10)
    assert num == 1


class TestHook(object):

    def test_backward_hook(self):
        conv = nn.Conv2d(3, 5, (2, 2), bias=True)
        bn = nn.BatchNorm2d(5)
        conv.register_backward_hook(module_hook1)
        bn.register_backward_hook(module_hook2)
        bn.register_backward_hook(module_hook3)
        x_array = np.random.randn(5 * 3 * 6 * 6).reshape(5, 3, 6, 6)
        # x = DArray.darray(x_array, arch=HostT, dtype=ds.Float32T)
        x = torch.ones(5 * 3 * 6 * 6).reshape(5, 3, 6, 6)
        x = torch.tensor(x)
        y = conv(x)
        y = bn(y)
        y = sum(y).item()
        y.backward()

m = TestHook()
m.test_backward_hook()