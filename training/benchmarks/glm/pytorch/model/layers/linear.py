import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def _initialize_affine_weight(weight,
                              output_size,
                              input_size,
                              per_partition_size,
                              partition_dim,
                              init_method,
                              stride=1,
                              return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = 1
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None


class ColumnLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 bias=True,
                 gather_output=True,
                 init_method=init.xavier_normal_,
                 stride=1,
                 keep_master_weight_for_test=False):
        super(ColumnLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = 1
        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(
            torch.Tensor(self.output_size_per_partition, self.input_size))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            self.bias.model_parallel = True
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.output_size,
            self.input_size,
            self.output_size_per_partition,
            0,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = input_
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        output = output_parallel
        return output


class RowLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_,
                 stride=1,
                 keep_master_weight_for_test=False):
        super(RowLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = 1
        self.input_size_per_partition = divide(input_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(
            torch.Tensor(self.output_size, self.input_size_per_partition))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.output_size,
            self.input_size,
            self.input_size_per_partition,
            1,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        # Matrix multiply.
        output_ = F.linear(input_, self.weight)

        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output


if __name__ == "__main__":
    print("test for ColumnLinear")
    hidden_size = 1024
    test_col_layer = ColumnLinear(
        hidden_size,
        3 * hidden_size,
        stride=3,
        gather_output=False,
    )
    inputs = torch.rand([2, 512, hidden_size])
    output = test_col_layer(inputs)
    print(output.shape)

    print("test for RowLinear ")
    test_row_layer = RowLinear(hidden_size, hidden_size, True, True)
    inputs_1 = torch.rand([2, 512, hidden_size])
    output = test_row_layer(inputs_1)
    print(output.shape)
