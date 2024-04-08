## 模型信息

LLaVA is an open-source chatbot trained by fine-tuning LLaMA/Vicuna on GPT-generated multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture.

Compared with older predecessors such as diffusion and CLIP, LLaVA is an LMM rather than a simple multi-modal model. 

Compared with other LMMs, LLaVA has three advantages:

A. LLaVA is completely open source, including training scripts, training data sets and model evaluation results, which is more fair and credible;

B. LLaVA is based on LLaMA or finetuned LLaMA, which is the most widely used open source LLM, and it is easy for manufacturers to adapt;

C. In the pretraining process of LLaVA, only the connector part is trained, so the computing resource overhead is small, and the manufacturer has low resource requirements for completing the evaluation.

## 模型配置及tokenizer准备

本测试样例为预训练case，需要下载模型config文件，以及tokenizer。

1.下载Vicuna checkpoints https://huggingface.co/lmsys/vicuna-13b-v1.5/tree/main ，放置在data_dir/LLaVA-Pretrain/checkpoints下

2.下载openai/clip-vit-large-patch14-336
https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main ，放置在data_dir/LLaVA-Pretrain/checkpoints下



## 数据准备

本测试样例数据准备共分为4个步骤

1. 下载558K的LAION-CC-SBU数据集子集和BLIP captions，即：

   https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main 中的blip_laion_cc_sbu_558k.json和images.zip，下载完后放在data_dir/LLaVA-Pretrain下

2. 下载llava的混合指令微调数据llava_v1_5_mix665k.json和对应的图片数据集

   混合指令微调数据：https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json，下载完后放在data_dir/LLaVA-Finetune下

   图片数据集：
   
   下载混合指令微调数据[llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), 将其放置在data_dir/LLaVA-Finetune下，之后下载对应的图片数据集：

   - COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
   - GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
   - OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **下载的图片需要全部保存为 `.jpg`**
   - TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
   - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

   下载完成后放置在 `data_dir/LLaVA-Finetune/data`,目录结构如下

   ```
   ├── coco
   │   └── train2017
   ├── gqa
   │   └── images
   ├── ocr_vqa
   │   └── images
   ├── textvqa
   │   └── train_images
   └── vg
      ├── VG_100K
      └── VG_100K_2
   ```

3. 下载评测数据集MMMU：https://huggingface.co/datasets/MMMU/MMMU/tree/main ，下载完后放置在data_dir下

最后的data_dir目录如下：
```
data_dir/
├── LLaVA-Finetune
│   ├── checkpoint
│   ├── data
│   └── llava_v1_5_mix665k.json
├── LLaVA-Pretrain
│   ├── blip_laion_cc_sbu_558k.json
│   ├── checkpoints
│   └── images
├── MMMU
│   ├── MMMU
└── Output
    ├── checkpoints_finetune
    └── checkpoints_pretrain
```