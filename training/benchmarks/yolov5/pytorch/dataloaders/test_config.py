
data_dir = "/data/sen.li/workspace/datasets/yolov5/coco/images"
train_data = "train2017"
eval_data = "val2017"
imgsz = 640
batch_size = 64


gs = 32 # grid size (max stride)
single_cls = True
pad = 0.5
# hyp is path/to/hyp.yaml or hyp dictionary
hyp = "/data/sen.li/workspace/code/yolov5/data/coco.yaml"
augment = True
cache = False
rect = True

rank = -1
workers = 4
image_weights = True
quad = False
prefix = ''
shuffle = True




