# =================================================
# Export variables
# =================================================
export OMP_NUM_THREADS=1

# 遇到从pytorch官网下载resnet权重比较慢的情况，可以手动拷贝本地的resnet50.pth到以下路径
# mkdir -p ~/.cache/torch/hub/checkpoints
# cp [your path of resnet50.pth] ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth