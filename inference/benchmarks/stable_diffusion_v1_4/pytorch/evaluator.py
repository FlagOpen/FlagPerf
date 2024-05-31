import torch


def evaluator(metric, image, prompt, config):
    scores = []
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    image = torch.tensor(image)
    for i in range(config.batch_size):
        scores.append(float(metric(image[i], prompt[i])))
    return scores
