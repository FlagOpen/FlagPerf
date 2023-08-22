from loguru import logger
import torch
import numpy as np
import time
from tools import torch_sync
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchmetrics.multimodal import CLIPScore


def cal_perf(config, dataloader_len, duration, core_time, str_prefix):
    model_forward_perf = config.repeat * dataloader_len * config.batch_size * config.num_inference_steps / duration
    logger.info(str_prefix + "(" + config.framework + ") Perf: " +
                str(model_forward_perf) + " ips")
    model_forward_core_perf = config.repeat * dataloader_len * config.batch_size * config.num_inference_steps / core_time
    logger.info(str_prefix + "(" + config.framework + ") core Perf: " +
                str(model_forward_core_perf) + " ips")
    return round(model_forward_perf, 3), round(model_forward_core_perf, 3)


def model_forward(model, dataloader, evaluator, config):
    if config.no_validation:
        return None, None, None
    vae = AutoencoderKL.from_pretrained(config.data_dir + "/" + config.weights,
                                        subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(config.data_dir + "/" +
                                              config.weights,
                                              subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.data_dir + "/" +
                                                 config.weights,
                                                 subfolder="text_encoder")
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    vae.eval()
    text_encoder.eval()

    metric = CLIPScore(model_name_or_path=config.data_dir + "/" +
                       config.eval_weights)
    metric.eval()

    generator = torch.Generator().manual_seed(config.random_seed)

    start = time.time()
    core_time = 0.0
    scores = []
    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))
        for step, prompt in enumerate(dataloader):
            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                             str(len(dataloader)))
                             
            with torch.no_grad():
                text_input = tokenizer(prompt,
                                       padding="max_length",
                                       max_length=tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors="pt")

                text_embeddings = text_encoder(text_input.input_ids)[0]

                max_length = text_input.input_ids.shape[-1]
                uncond_input = tokenizer([""] * config.batch_size,
                                         padding="max_length",
                                         max_length=max_length,
                                         return_tensors="pt")

                uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
                text_embeddings = torch.cat(
                    [uncond_embeddings, text_embeddings])

                latents = torch.randn(
                    (config.batch_size, config.in_channels, config.height //
                     config.scale_size, config.width // config.scale_size),
                    generator=generator)

                noise_scheduler.set_timesteps(config.num_inference_steps)

                timesteps_tensor = torch.linspace(
                    config.num_train_timesteps -
                    config.num_train_timesteps // config.num_inference_steps,
                    0, config.num_inference_steps).int()

                for t in timesteps_tensor:
                    latent_model_input = torch.cat([latents] * 2)

                    torch_sync(config)
                    core_time_start = time.time()
                    if config.fp16:
                        noise_pred = model(
                            latent_model_input.cuda().to(torch.float16),
                            t.cuda(),
                            text_embeddings.cuda().to(torch.float16))
                    else:
                        noise_pred = model(latent_model_input.cuda(), t.cuda(),
                                           text_embeddings.cuda())

                    torch_sync(config)
                    core_time += time.time() - core_time_start

                    noise_pred = noise_pred.to(torch.float32).cpu()

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + config.guidance_scale * (
                        noise_pred_text - noise_pred_uncond)

                    latents = noise_scheduler.step(noise_pred, t,
                                                   latents).prev_sample

                latents = 1 / 0.18215 * latents               
                image = vae.decode(latents).sample

                scores_iter = evaluator(metric, image, prompt, config)
                for score in scores_iter:
                    scores.append(score)                  

    duration = time.time() - start
    logger.info("CLIP Scores: " + str(np.mean(scores)))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time, "Validation")

    return model_forward_perf, model_forward_core_perf, round(
        float(np.mean(scores)), 3)


def engine_forward(model, dataloader, evaluator, config):
    vae = AutoencoderKL.from_pretrained(config.data_dir + "/" + config.weights,
                                        subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(config.data_dir + "/" +
                                              config.weights,
                                              subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.data_dir + "/" +
                                                 config.weights,
                                                 subfolder="text_encoder")
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    vae.eval()
    text_encoder.eval()

    metric = CLIPScore(model_name_or_path=config.data_dir + "/" +
                       config.eval_weights)
    metric.eval()

    generator = torch.Generator().manual_seed(config.random_seed)

    start = time.time()
    core_time = 0.0
    scores = []
    for times in range(config.repeat):

        logger.debug("Repeat: " + str(times + 1))
        for step, prompt in enumerate(dataloader):
            if step % config.log_freq == 0:
                logger.debug("Step: " + str(step) + " / " +
                             str(len(dataloader)))

            with torch.no_grad():
                text_input = tokenizer(prompt,
                                       padding="max_length",
                                       max_length=tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors="pt")

                text_embeddings = text_encoder(text_input.input_ids)[0]

                max_length = text_input.input_ids.shape[-1]
                uncond_input = tokenizer([""] * config.batch_size,
                                         padding="max_length",
                                         max_length=max_length,
                                         return_tensors="pt")

                uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
                text_embeddings = torch.cat(
                    [uncond_embeddings, text_embeddings])

                latents = torch.randn(
                    (config.batch_size, config.in_channels, config.height //
                     config.scale_size, config.width // config.scale_size),
                    generator=generator)

                noise_scheduler.set_timesteps(config.num_inference_steps)

                timesteps_tensor = torch.linspace(
                    config.num_train_timesteps -
                    config.num_train_timesteps // config.num_inference_steps,
                    0, config.num_inference_steps).int()

                for t in timesteps_tensor:
                    latent_model_input = torch.cat([latents] * 2)

                    inputs = [latent_model_input, t, text_embeddings]
                    if config.fp16:
                        inputs = [
                            latent_model_input.to(torch.float16), t,
                            text_embeddings.to(torch.float16)
                        ]

                    torch_sync(config)
                    core_time_start = time.time()
                    outputs = model(inputs)
                    noise_pred = outputs[0]
                    foo_time = outputs[1]

                    torch_sync(config)
                    core_time += time.time() - core_time_start

                    noise_pred = noise_pred[0].float()
                    noise_pred = noise_pred.reshape(
                        config.batch_size * 2, config.in_channels,
                        config.height // config.scale_size,
                        config.width // config.scale_size)
                    noise_pred = noise_pred.cpu()

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + config.guidance_scale * (
                        noise_pred_text - noise_pred_uncond)

                    latents = noise_scheduler.step(noise_pred, t,
                                                   latents).prev_sample

                latents = 1 / 0.18215 * latents               
                image = vae.decode(latents).sample

                scores_iter = evaluator(metric, image, prompt, config)
                for score in scores_iter:
                    scores.append(score)                

    duration = time.time() - start
    logger.info("CLIP Scores: " + str(np.mean(scores)))

    duration = time.time() - start
    model_forward_perf, model_forward_core_perf = cal_perf(
        config, len(dataloader), duration, core_time, "Inference")

    return model_forward_perf, model_forward_core_perf, round(
        float(np.mean(scores)), 3)
