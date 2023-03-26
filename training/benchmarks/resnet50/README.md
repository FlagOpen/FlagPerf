
## Model Introduction
### What is ResNet?
ResNet stands for Residual Network and is a specific type of convolutional neural network (CNN) introduced in the 2015 paper "Deep Residual Learning for Image Recognition" by He Kaiming, Zhang Xiangyu, Ren Shaoqing, and Sun Jian. CNNs are commonly used to power computer vision applications.

ResNet-50 is a 50-layer convolutional neural network (48 convolutional layers, one MaxPool layer, and one average pool layer). Residual neural networks are a type of artificial neural network (ANN) that forms networks by stacking residual blocks.


### ResNet-50 Architecture
![avata](https://iq.opengenus.org/content/images/2020/03/Screenshot-from-2020-03-20-15-49-54.png)

Now we are going to discuss about Resnet 50 and also the architecture for the above talked 18 and 34 layer ResNet is also given residual mapping and not shown for simplicity.

There was a small change that was made for the ResNet 50 and above that before this the shortcut connections skipped two layers but now they skip three layers and also there was 1 * 1 convolution layers added that we are going to see in detail with the ResNet 50 Architecture.

![avata](https://iq.opengenus.org/content/images/2020/03/Screenshot-from-2020-03-20-15-56-22.png)


So as we can see in the table 1 the resnet 50 architecture contains the following element:

A convoultion with a kernel size of 7 * 7 and 64 different kernels all with a stride of size 2 giving us 1 layer.
Next we see max pooling with also a stride size of 2.
In the next convolution there is a 1 * 1,64 kernel following this a 3 * 3,64 kernel and at last a 1 * 1,256 kernel, These three layers are repeated in total 3 time so giving us 9 layers in this step.
Next we see kernel of 1 * 1,128 after that a kernel of 3 * 3,128 and at last a kernel of 1 * 1,512 this step was repeated 4 time so giving us 12 layers in this step.
After that there is a kernal of 1 * 1,256 and two more kernels with 3 * 3,256 and 1 * 1,1024 and this is repeated 6 time giving us a total of 18 layers.
And then again a 1 * 1,512 kernel with two more of 3 * 3,512 and 1 * 1,2048 and this was repeated 3 times giving us a total of 9 layers.
After that we do a average pool and end it with a fully connected layer containing 1000 nodes and at the end a softmax function so this gives us 1 layer.
We don't actually count the activation functions and the max/ average pooling layers.

so totaling this it gives us a 1 + 9 + 12 + 18 + 9 + 1 = 50 layers Deep Convolutional network.



Please refer to this paper for a detailed description of Deep Residual network:
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)






## Model source code

| model url                                                                | commit_id | date      |
| ------------------------------------------------------------------------ | --------- | --------- |
| https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py | 7dc5e5b   | 2023-1-11 |




## Dataset

> Download website：https://image-net.org/

ImageNet2012
| Dataset                       | FileName  | Size  | Checksum                              |
| ----------------------------- |----- | ----- | ------------------------------------- |
| Training image (Task 1 & 2)   |ILSVRC2012_img_train.tar | 138GB | MD5: ccaf1013018ac1037801578038d370da |
| Validation images (all tasks) | ILSVRC2012_img_val.tar |6.3GB | MD5: 29b22e2961454d5413ddabcf34fc5622 |
```
file list:
ILSVRC2012_img_train.tar
ILSVRC2012_img_val.tar
```

Dataset format conversion：
https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh

make sure ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar are in the same directory with extract_ILSVRC.sh.
```bash
sh extract_ILSVRC.sh
```

preview directory structures of decompressed dataset.

```bash
tree -d -L 1
```

```
.
├── train
└── val
```

## Checkpoint
> None

## AI Frameworks && Accelerators supports

|            | Pytorch | Paddle | TensorFlow2 |
| ---------- | ------- | ------ | ----------- |
| Nvidia GPU | ✅       | N/A    | N/A         |
