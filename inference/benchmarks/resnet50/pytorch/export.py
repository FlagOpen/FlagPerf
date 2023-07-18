import torch
import os


def export_model(model, config):
    dummy_input = torch.randn(config.batch_size, 3, 224, 224)

    if config.fp16:
        dummy_input = dummy_input.half()
    dummy_input = dummy_input.cuda()

    onnx_path = config.perf_dir + "/" + config.onnx_path

    dir_onnx_path = os.path.dirname(onnx_path)
    os.makedirs(dir_onnx_path, exist_ok=True)

    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      verbose=False,
                      input_names=["input"],
                      output_names=["output"],
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True)
