import torch
import os


def export_model(model, config):
    if config.exist_onnx_path is not None:
        return config.exist_onnx_path

    filename = config.case + "_bs" + str(config.batch_size)
    filename = filename + "_" + str(config.framework)
    filename = filename + "_fp16" + str(config.fp16)
    filename = "onnxs/" + filename + ".onnx"
    onnx_path = config.perf_dir + "/" + filename

    img = torch.randn(config.batch_size, 3, 1024, 1024).cuda()
    points = torch.ones(config.batch_size, 1, 1, 2).cuda()

    if config.fp16:
        img = img.half()
    dummy_input = (img, points)

    dir_onnx_path = os.path.dirname(onnx_path)
    os.makedirs(dir_onnx_path, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          verbose=False,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True)

    return onnx_path
