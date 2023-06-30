### 模型信息
- 模型介绍
>Swin Transformer (the name Swin stands for Shifted window) is initially described in arxiv, which capably serves as a general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection.

- Paper链接
> https://arxiv.org/abs/2103.14030

- 模型代码来源

This case includes code from the MIT License open source project at 
https://github.com/microsoft/Swin-Transformer (commit b720b4191588c19222ccf129860e905fb02373a7)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.



### 数据集
- 数据集下载地址
> https://www.image-net.org/challenges/LSVRC/2012/  (Imagenet2012 1K)

- 预处理
> 无需预处理 


### 框架与芯片支持情况
|     | Pytorch  |Paddle|TensorFlow2|
|  ----  | ----  |  ----  | ----  |
| Nvidia GPU | ✅ |N/A  |N/A|
| 昆仑芯 XPU | N/A |N/A  |N/A|
| 天数智芯GPU | ✅  |N/A  |N/A|


