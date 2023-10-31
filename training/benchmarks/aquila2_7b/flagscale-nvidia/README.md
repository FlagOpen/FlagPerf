## 模型信息

aquila2是北京人工智能研究院开源的语言模型，包含基础语言模型 **Aquila2-7B** 和 **Aquila2-34B** ，对话模型 **AquilaChat2-7B** 和 **AquilaChat2-34B**，长文本对话模型**AquilaChat2-7B-16k** 和 **AquilaChat2-34B-16k**

## 模型配置及tokenizer准备

本测试样例为预训练case，需要下载tokenizer，下载链接为https://github.com/FlagOpen/FlagScale/tree/main/examples/aquila/tokenizer。需要在data_dir下创建tokenizer目录，将上述链接中的三个文件下载到此目录中

## 数据准备

本测试样例数据使用FlagScale-aquila2预处理好的PILE数据集，下载链接为

https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin

https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx

将上述两个文件放置于data_dir下。