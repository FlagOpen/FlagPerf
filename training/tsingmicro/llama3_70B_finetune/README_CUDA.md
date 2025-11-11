
1、下载源码：
    git clone http://gitlab.tsingmicro.com/likang/examples.git
    cd examples/llama3_70B_finetune

2、安装CUDA环境(pip源：index-url = https://pypi.tuna.tsinghua.edu.cn/simple)
    conda create -n train_cuda python=3.10 -y
    conda activate train_cuda
    pip install transformers==4.46.1
    pip install numpy==1.24.4
    pip install torch==2.5.1
    pip install datasets accelerate tensorboard ninja
    pip install flash-attn==2.6.3

3、run llama70b, 单机单卡1block
    bash finetuning_cuda.sh 1 1 1 1 0 1 128