## 模型信息
- Introduction
todo

## 数据准备
### 数据集准备

本测试样例数据使用FlagScale-llama2预处理好的数据集，下载链接为

https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/wurui04/llama_00_text_document.bin

https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/wurui04/llama_00_text_document.idx

在data_dir下创建dataset目录，将上述两个文件放置于data_dir/dataset下。

### 目录树
todo

### 数据集引用

```
@article{pile,
  title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
```