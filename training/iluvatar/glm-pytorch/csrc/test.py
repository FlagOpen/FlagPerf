import os
import torch
import numpy as np
import time
from collections import OrderedDict
from self_multihead_attn import SelfMultiheadAttn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0"

# test_args
# batch_size, seq_len, hidden_size
args = [[10, 512, 1024], [20, 512, 1024], [32, 512, 1024], [64, 512, 1024],
        [80, 512, 1024]]

forward_pass_threshold = 1e-4
backward_pass_threshold = 1e-4
num_repeats = 20

hidden_size = 1024
num_attention_heads = 16
attention_dropout_prob = 0
output_dropout_prob = 0


def build_mask_matrix(hidden_states, seq_length, sep):
    hidden_states = hidden_states.transpose(0, 1)
    batch_size = hidden_states.size(0)
    m = hidden_states.new_ones((1, seq_length, seq_length))
    m = torch.tril(m)
    m = m.expand(batch_size, -1, -1)
    ids = torch.arange(seq_length, device=sep.device,
                       dtype=sep.dtype).view(1, -1)
    mask = ids < sep.view(-1, 1)
    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
    m = m.unsqueeze(1)
    mask = m
    mask = mask.repeat([1, num_attention_heads, 1, 1])
    mask = mask.view([-1, mask.shape[-2], mask.shape[-1]])
    return 1 - mask


def gen_inputs(batch_size, seq_len, hidden_size):
    data = np.random.rand(seq_len, batch_size, hidden_size)

    inputs1 = torch.tensor(data,
                           dtype=torch.float16,
                           device=device,
                           requires_grad=True)
    inputs2 = torch.tensor(data,
                           dtype=torch.float16,
                           device=device,
                           requires_grad=True)

    mask_data = np.random.randint(0, seq_len, batch_size)

    mask_1 = torch.tensor(mask_data,
                          dtype=torch.int64,
                          device=device,
                          requires_grad=False)
    mask_2 = torch.tensor(mask_data,
                          dtype=torch.int64,
                          device=device,
                          requires_grad=False)

    mask_1 = build_mask_matrix(
        inputs1, seq_len,
        mask_1).byte()  # batch_size*num_heads,seq_len,seq_len
    mask_2 = build_mask_matrix(inputs2, seq_len, mask_2).byte()

    return inputs1, inputs2, mask_1, mask_2


if __name__ == "__main__":
    layer1 = SelfMultiheadAttn(hidden_size,
                               num_attention_heads,
                               bias=True,
                               impl='default').to(device).half()
    layer2 = SelfMultiheadAttn(hidden_size,
                               num_attention_heads,
                               bias=True,
                               impl='fast').to(device).half()

    layer1.k_weight.data = layer2.k_weight.data.clone()
    layer1.q_weight.data = layer2.q_weight.data.clone()
    layer1.v_weight.data = layer2.v_weight.data.clone()

    layer1.out_proj_weight.data = layer2.out_proj_weight.data.clone()

    forward_test_results = OrderedDict()
    backward_test_results = OrderedDict()

    for case_arg in args:
        case_str = str(case_arg)
        forward_test_results[case_str] = dict()
        backward_test_results[case_str] = dict()
        inputs1, inputs2, mask1, mask2 = gen_inputs(*case_arg)
        input_args_1 = {
            "query": inputs1,
            "attn_mask": mask1,
            "is_training": True
        }
        input_args_2 = {
            "query": inputs2,
            "attn_mask": mask2,
            "is_training": True
        }

        # forward result check
        out1 = layer1(**input_args_1)
        out2 = layer2(**input_args_2)

        out1 = layer1(**input_args_1)
        out2 = layer2(**input_args_2)

        diff = torch.abs(out1 - out2)
        if ((diff < forward_pass_threshold).all()):
            forward_test_results[case_str]['correctness'] = 'PASS'
        else:
            forward_test_results[case_str]['correctness'] = 'FAILED'
            print(diff.max())
        forward_test_results[case_str]['correct_rate'] = (
            diff <
            forward_pass_threshold).sum().cpu().item() / torch.numel(diff)

        # backward
        bk_data = torch.rand_like(out1)
        out1.backward(bk_data.clone())
        out2.backward(bk_data.clone())

        grad1 = inputs1.grad
        grad2 = inputs2.grad
        diff = torch.abs(out1 - out2)
        if ((diff < backward_pass_threshold).all()):
            backward_test_results[case_str]['correctness'] = 'PASS'
        else:
            backward_test_results[case_str]['correctness'] = 'FAILED'
            print(diff.max())
        backward_test_results[case_str]['correct_rate'] = (
            diff <
            backward_pass_threshold).sum().cpu().item() / torch.numel(diff)

        perf_results_1 = []
        perf_results_2 = []

        bk_perf_results_1 = []
        bk_perf_results_2 = []

        for i in range(num_repeats):
            torch.cuda.synchronize()
            start_time = time.time()
            out1 = layer1(**input_args_1)
            torch.cuda.synchronize()
            duration = time.time() - start_time
            perf_results_1.append(duration)

            start_time = time.time()
            out1.backward(bk_data.clone())
            torch.cuda.synchronize()
            duration = time.time() - start_time
            bk_perf_results_1.append(duration)

            start_time = time.time()
            out2 = layer2(**input_args_2)
            torch.cuda.synchronize()
            duration = time.time() - start_time
            perf_results_2.append(duration)

            start_time = time.time()
            out2.backward(bk_data.clone())
            torch.cuda.synchronize()
            duration = time.time() - start_time
            bk_perf_results_2.append(duration)

        def compute_impl_time(time_results):
            return (sum(time_results) -
                    max(time_results)) / (len(time_results) - 1)

        forward_test_results[case_str]['native_impl_time'] = compute_impl_time(
            perf_results_1)
        forward_test_results[case_str][
            'speedup_impl_time'] = compute_impl_time(perf_results_2)
        forward_test_results[case_str]['speedup_ratio'] = compute_impl_time(
            perf_results_1) / compute_impl_time(perf_results_2)

        backward_test_results[case_str][
            'native_impl_time'] = compute_impl_time(bk_perf_results_1)
        backward_test_results[case_str][
            'speedup_impl_time'] = compute_impl_time(bk_perf_results_2)
        backward_test_results[case_str]['speedup_ratio'] = compute_impl_time(
            bk_perf_results_1) / compute_impl_time(bk_perf_results_2)

    print("### forward ")
    print(
        f"| No. | parameters | native_impl_time | speedup_impl_time | speedup_ratio | correctness | correct_rate |"
    )
    print('| --- | --- | --- | --- | --- | --- | --- |')
    for i, key in enumerate(forward_test_results):
        print(
            f'| {i+1}. | {key} | {forward_test_results[key]["native_impl_time"]} | {forward_test_results[key]["speedup_impl_time"]} | {forward_test_results[key]["speedup_ratio"]} | {forward_test_results[key]["correctness"]} | {forward_test_results[key]["correct_rate"]} |'
        )

    print("### backward ")
    print(
        f"| No. | parameters | native_impl_time | speedup_impl_time | speedup_ratio | correctness | correct_rate |"
    )
    print('| --- | --- | --- | --- | --- | --- | --- |')
    for i, key in enumerate(forward_test_results):
        print(
            f'| {i+1}. | {key} | {backward_test_results[key]["native_impl_time"]} | {backward_test_results[key]["speedup_impl_time"]} | {backward_test_results[key]["speedup_ratio"]} | {backward_test_results[key]["correctness"]} | {backward_test_results[key]["correct_rate"]} |'
        )
