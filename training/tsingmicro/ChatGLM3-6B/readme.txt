使用transformers 4.46.1版本，需要将模型路径下的tokenization_chatglm.py文件的274行添加padding_side: Optional[str] = None。所需的数据集：/login_home/yuancong/workplace/data/hugginface/datasets/flagperf
