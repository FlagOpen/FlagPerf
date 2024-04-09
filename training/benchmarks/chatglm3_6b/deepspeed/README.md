## 模型信息

ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的新一代对话预训练模型。ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：

1. **更强大的基础模型：** ChatGLM3-6B 的基础模型 ChatGLM3-6B-Base 采用了更多样的训练数据、更充分的训练步数和更合理的训练策略。在语义、数学、推理、代码、知识等不同角度的数据集上测评显示，**ChatGLM3-6B-Base 具有在 10B 以下的基础模型中最强的性能**。
2. **更完整的功能支持：** ChatGLM3-6B 采用了全新设计的 [Prompt 格式](https://github.com/THUDM/ChatGLM3/blob/main/PROMPT.md)，除正常的多轮对话外。同时原生支持[工具调用](https://github.com/THUDM/ChatGLM3/blob/main/tool_using/README.md)（Function Call）、代码执行（Code Interpreter）和 Agent 任务等复杂场景。
3. **更全面的开源序列：** 除了对话模型 [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b) 外，还开源了基础模型 [ChatGLM3-6B-Base](https://huggingface.co/THUDM/chatglm3-6b-base)、长文本对话模型 [ChatGLM3-6B-32K](https://huggingface.co/THUDM/chatglm3-6b-32k)。以上所有权重对学术研究**完全开放**，在填写[问卷](https://open.bigmodel.cn/mla/form)进行登记后**亦允许免费商业使用**。

## 模型配置及tokenizer准备

本测试样例为预训练case，需要下载模型config文件，以及tokenizer。

本测试样例目录下已提供处理好的chatglm3_6b_hf/目录

## 数据准备

本测试样例数据准备共分为4个步骤

1. 下载openwebtext原始压缩文件，即：

   https://drive.google.com/drive/folders/1IaD_SIIB-K3Sij_-JjWoPy_UrWqQRdjx 中12GB的openwebtext.tar.xz

2. 全部解压缩

   解压上述12GB的文件后，会出现若干形如urlsf_subsetxxxxxx.xz的压缩文件，将所有压缩文件解压到同一个目录，最终可获得7000000余个txt文件

3. 运行数据预处理文件

   执行preprocess/data_process.py，配置好其中的4个命令行参数。推荐的默认token数量为100M，即1亿个token。此配置在A800 8卡上预计训练1小时

4. 将outputfile（通常为openwebtext_chatglm3_100M.npy）放置在data_dir下

值得注意的是，由于原始Google Drive存储内容变动，自2024.03起，可忽略上述第1步骤，从链接中下载20个子压缩目录，随后全部解压到同一个目录，继续执行第3、4步骤。
