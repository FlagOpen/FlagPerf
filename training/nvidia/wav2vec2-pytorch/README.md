
<<<<<<< HEAD
### 数据集下载

1. Clone the repository.
   ```bash
    cd <FlagPerf>/training/benchmarks/wav2vec2/pytorch/scripts/
   ```

2.  Build the 22.11-py3 PyTorch NGC container and start an interactive session to run training/inference. `DATASET_DIR` on the host will be mounted as `/datasets` inside the container.
    ```bash
    bash scripts/docker/build.sh
    DATASET_DIR=[PATH] bash scripts/docker/run.sh
    ```

3.  Download and preprocess the dataset. The dataset size is about 70GB and this step could take up to a few hours to complete.
    ```bash
    bash scripts/download_data.sh
    ```

4.  Generate filelists.
    ```bash
    bash scripts/generate_filelists.sh
    ```

5. Start training.
    ```bash
    NUM_GPUS=[NUM] UPDATE_FREQUENCY=[NUM] NUM_CONCAT_BATCHES=[NUM] BF16=[true|false] FP16=[true|false] \
        bash scripts/pretrain_base.sh
    ```
    Adjust the variables to maintain `NUM_GPUS x NUM_CONCAT_BATCHES x UPDATE_FREQUENCY = 64`.
    For more details, refer to [Adjusting batch size and the number of GPUs](#adjusting-batch-size-and-the-number-of-gpus) and [Adjusting mixed precision](#adjusting-mixed-precision).

    For PerfPerf, the default config from :
    ```bash
    # precision training on 4x A100 40GB
    NUM_GPUS=8 NUM_CONCAT_BATCHES=4 UPDATE_FREQUENCY=2 bash scripts/pretrain_base.sh
    ```


### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度(mAP) | 性能（samples/s） |
| -------- | --------------- | ----------- | -------- | ------------- | ----------------- |
| 单机8卡  | config_A100x1x8 | 3-4天    | 0.605    | 0.3520        | 1363049      |
=======
### 数据集准备

https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2

### 运行情况
| 训练资源 | 配置文件        | 运行时长(s) | 目标精度 | 收敛精度(mAP) | 性能（ntokens/s） |
| -------- | --------------- | ----------- | -------- | ------------- | ----------------- |
| 单机8卡  | config_A100x1x8 | 3-4天    | 0.605    | 0.605       | 1363049      |
>>>>>>> d9f0d2f51a94ff4b7e8ed42c1ddc40d6434b2deb

ps：
因大模型运行较久，参考精度通过resume+ckpt方式训练得到，将如下文件中resume = True && no_save = False 即可支持训练过程保持ckpt，任务断掉后继续接着训练。(wav2vec2_Perf/training/benchmarks/wav2vec2/pytorch/config/_base.py)

* resume=False #[bool], default False, if True, read last_checkpoint from ckpt saved path
* ckpt=None   #[str,path], default None, if True, given ckpt path
* no_save=True #[bool], default True, if True , do not save ckpt