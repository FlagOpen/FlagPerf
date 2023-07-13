export PYTHON_COMMAND=python3.7

export TRAIN_DATA_PATH=/home/datasets/imagenet/train/
export EVAL_DATA_PATH=/home/datasets/imagenet/val/
export EPOCH_SIZE=90

export RANK_SIZE=8
export DEVICE_NUM=8

# need if rank_size > 1
export RANK_TABLE_FILE=/home/lcm/tool/rank_table_8p.json

# cluster need for node info
#export NODEINFO_FILE=/home/lcm/tool/ssh64_66.json
