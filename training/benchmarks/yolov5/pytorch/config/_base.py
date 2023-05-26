
# vendor
vendor="nvidia"

# driver config
# local_rank=0
log_freq=100
name="yolov5"
device=None
n_device=1
fp16=True
data_dir=""  # data dir, save as CASES's data dir in test_conf.py
gradient_accumulation_steps=1
dist_backend="nccl"
do_train=True
train_batch_size=64

target_map=0.363

# yolov5 config
weights=""
cfg="yolov5s.yaml"
data="coco.yaml"
hyp="hyps/hyp.scratch-low.yaml"
epochs=10
batch_size=train_batch_size
imgsz=640
rect=False
resume=False
nosave=False
noval=False
noautoanchor=False
noplots=True
evolve=None
cache=None
image_weights=False
# device=0,1
multi_scale=False
single_cls=False
optimizer="SGD"
sync_bn=False
workers=8
exist_ok=False
quad=False
cos_lr=False
label_smoothing=0.0
patience=100
freeze=[0]
save_period=-1
seed=2023
local_rank=-1
entity=None
upload_dataset=False
bbox_interval=-1
artifact_alias="latest"

project="result/runs"





