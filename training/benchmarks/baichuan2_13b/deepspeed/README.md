## 模型信息

Baichuan 2 is the new generation of large-scale open-source language models launched by Baichuan Intelligence inc.. It is trained on a high-quality corpus with 2.6 trillion tokens and has achieved the best performance in authoritative Chinese and English benchmarks of the same size.

## 模型配置及tokenizer准备

本测试样例为预训练case，需要下载baichuan2\_13模型huggingface所有文件， 存放在data_dir/baichuan2_13b_hf目录。

## 数据准备

本测试样例数据准备共分为4个步骤

1. 下载openwebtext原始压缩文件，即：

   https://drive.google.com/drive/folders/1IaD_SIIB-K3Sij_-JjWoPy_UrWqQRdjx 中12GB的openwebtext.tar.xz

2. 全部解压缩

   解压上述12GB的文件后，会出现若干形如urlsf_subsetxxxxxx.xz的压缩文件，将所有压缩文件解压到同一个目录，最终可获得7000000余个txt文件

3. 运行数据预处理文件

   执行preprocess/data_process.py，配置好其中的4个命令行参数。推荐的默认token数量为100M，即1亿个token。此配置在H800 8卡上预计训练xxx小时

4. 将outputfile（通常为openwebtext_baichuan2_100M.npy）放置在data_dir下

## 厂商特殊配置

本测试样例除基础的config_\<vendor\>x\<nnodes>x\<nproc\>.py（如config_H800x2x8.py）与requirements.txt外，还提供同目录下的ds_config.json与net.sh供厂商适配使用：

1. 厂商可在ds_config.json中新增或改写benchmarks/目录下的ds_config.json，以nv-H800为例，可以将gradient_accu_steps修改为64，以供精度对齐实验
2. 厂商可在net.sh中**用一行环境变量命令**配置集群网络环境，以nv-H800为例，配置了socket_ifname以及IB相关配置