#!/bin/bash
export https_proxy=http://10.1.0.34:7890
pip install deepspeed==0.11.1

wget https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/zhangling21_baichuan2/xmlir_fixeq.run

bash xmlir_fixeq.run
XFLAGS --disable megatron_23_05
