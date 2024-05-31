## 模型信息

aquila2是北京人工智能研究院开源的语言模型，包含基础语言模型 **Aquila2-7B** 和 **Aquila2-34B** ，对话模型 **AquilaChat2-7B** 和 **AquilaChat2-34B**，长文本对话模型**AquilaChat2-7B-16k** 和 **AquilaChat2-34B-16k**

## 模型配置及tokenizer准备

本测试样例为微调case，需要下载tokenizer，本测试样例为微调case，需要下载tokenizer, 模型config文件以及模型权重，

下载链接为：https://model.baai.ac.cn/model-detail/100098

在data_dir中分别创建tokenizer文件夹和checkpoint文件夹存放相应文件

本测试样例对应FlagScale版本为FlagScale仓库ed55532这一commit版本

## 数据准备

本测试样例数据使用FlagScale-aquila2预处理好的alpaca数据集，下载链接为

https://github.com/FlagAI-Open/FlagAI/blob/40a72ad518aa5c4ee639e7cee77b77d4472b4bb8/examples/Aquila/Aquila-chat/data/alpaca_data_train.jsonl

在data_dir中创建dataset文件夹，将上述文件放置于data_dir/dataset下。
