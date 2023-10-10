# 训练日志
1. 文件名规范：\<model\>\-\<vendor\>-\<nnodes\>x\<nprocs\>.zip
> 示例
> - longformer-nvidia-1x1.zip
> - t5_small-kunlunxin-2x8.zip
> - swin_transformer-iluvatar-1x8.zip

2. 步骤及命令：
   1. 创建实验目录
   ```bash
   mkdir <model>-<vendor>-<nnodes>x<nprocs>
   ```
   
   2. 移动训练日志目录到新创建的实验目录   
   ```bash
   mv run<timestamp> <model>-<vendor>-<nnodes>x<nprocs>
   ```
   
   3. 压缩
   ```bash
   zip -r <model>-<vendor>-<nnodes>x<nprocs>.zip <target_directory>
   ```
   
   注：
   - 这里的target_directory，为\<model\>-\<vendor\>-\<nnodes\>x\<nprocs\>
   - **建议在linux下压缩**，MacOS下压缩，会增加无效的文件或目录(.DS_Store文件及__MACOSX目录)
   - 如果机器上没有安装zip命令，通过 ```bash sudo apt-get install zip```来安装