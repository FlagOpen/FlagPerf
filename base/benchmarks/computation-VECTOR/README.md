# Vector 算力测试

## 测试目标

测量 GPU 的 Vector ALU（向量算术逻辑单元）性能，通过逐元素运算（element-wise operations）评估向量计算吞吐量。

## 测试原理

### 测试方法
使用向量融合乘加操作（Fused Multiply-Add, FMA）：
```
result = a * b + c
```

其中 a, b, c 都是大型向量（数百万到数亿个元素）。

### 性能计算
```
TOPS = (向量大小 × 2 × 迭代次数) / 时间 / 1e12

其中：
- 向量大小：向量的元素数量
- 2：每个元素执行 2 次操作（1 次乘法 + 1 次加法）
- TOPS：Tera Operations Per Second (万亿次操作/秒)
```

## 配置参数

### 通用配置 (case_config.yaml)
- `VECTOR_SIZE`: 256M elements (约 1GB for FP32)
- `WARMUP`: 50 次预热
- `ITERS`: 10000 次迭代

### NVIDIA A100 特定配置
- `VECTOR_SIZE`: 512M elements (约 2GB for FP32)
- `DIST_BACKEND`: "nccl"

## 运行测试

### 1. 修改 host.yaml
```yaml
VENDOR: "nvidia"
CASES:
    "computation-VECTOR:A100": "pytorch_2.3"
```

### 2. 执行测试
```bash
cd FlagPerf/base/
python3 run.py
```

### 3. 查看结果
结果会以如下格式输出：
```
[FlagPerf Result]Rank 0's computation-VECTOR=XXX.XX TOPS
```

## 与其他算力测试的区别

| 测试类型 | 操作类型 | 主要硬件单元 | 数据访问模式 |
|---------|---------|-------------|------------|
| **Matrix (FP16/FP32)** | 矩阵乘法 | Tensor Core / CUDA Core | 密集计算 |
| **Vector (本测试)** | 逐元素运算 | Vector ALU | 带宽密集 |

### 适用场景
- **Matrix 测试**: 评估深度学习训练/推理的计算能力
- **Vector 测试**: 评估数据处理、激活函数、归一化等操作的性能

## 预期性能参考

### NVIDIA A100 (40GB)
- **理论峰值**: ~19.5 TFLOPS (FP32)
- **实测 Vector TOPS**: 预计 10-15 TOPS
  - Vector 操作受带宽限制，通常达不到峰值算力

### 性能影响因素
1. **显存带宽**: Vector 操作是带宽密集型
2. **向量大小**: 太小无法充分利用硬件，太大可能超出显存
3. **缓存命中率**: 重复访问相同数据可提高性能

## 扩展支持其他厂商

### 添加新厂商支持
1. 创建目录：`computation-VECTOR/<vendor>/`
2. 添加配置：`<vendor>/<chip>/case_config.yaml`
3. 更新 `drivers/utils.py` 如果需要特殊同步逻辑

### 示例：添加昇腾支持
```bash
mkdir -p computation-VECTOR/ascend/A3/
# 创建 case_config.yaml, env.sh, requirements.txt
```

## 故障排查

### 常见问题

1. **显存不足 (OOM)**
   - 减小 `VECTOR_SIZE` 参数
   - 检查是否有其他进程占用显存

2. **性能异常低**
   - 检查是否开启了功耗限制（thermal throttling）
   - 确认没有其他负载竞争资源

3. **分布式环境错误**
   - 检查 `DIST_BACKEND` 配置
   - 确认多机 SSH 互信配置正确

## 参考资料

- FlagPerf 官方文档: https://github.com/FlagOpen/FlagPerf
- NVIDIA CUDA 编程指南: https://docs.nvidia.com/cuda/
- PyTorch 性能优化: https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
