### 模型信息
- 模型介绍
>Swin Transformer (the name Swin stands for Shifted window) is initially described in arxiv, which capably serves as a general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection.

-Paper链接
> https://arxiv.org/abs/2103.14030

- 模型代码来源
> https://github.com/microsoft/Swin-Transformer

### 数据集
- 数据集下载地址
> `https://image-net.org/download.php`  (Imagenet2012 1K)

- 预处理
> 无需预处理 


### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | ✅ |N/A  |N/A|
| 昆仑芯 XPU | ✅ |N/A  |N/A|
| 天数智芯GPU | ✅ |N/A  |N/A|


