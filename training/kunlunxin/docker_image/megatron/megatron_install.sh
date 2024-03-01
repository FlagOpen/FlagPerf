#!/bin/bash
# using github mirrors to avoid github TTL
#export https_proxy=http://10.1.0.34:7890
git clone https://githubfast.com/FlagOpen/FlagScale
cd FlagScale

git checkout eb0438a5459404e2e4c70b15fa37e9a197ab159d
echo 'export PYTHONPATH=$PYTHONPATH:/home/FlagScale' >> /root/.bashrc
source /root/.bashrc

wget https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/zhangling21_llama70B/xmlir201_5.run
bash xmlir201_5.run
XFLAGS --enable transformer_engine
XFLAGS --enable flagscale