# 参评AI芯片信息

* 厂商：{{ vendor }}

* 产品名称：{{ v_chip }}
* 产品型号：{{ chip }}
* SHM_SIZE：{{ shm_size }}


# 评测结果


| op_name  | dtype | shape_detail | 无预热时延(Latency-No warmup) | 预热时延(Latency-Warmup) | 原始吞吐(Raw-Throughput) | 核心吞吐(Core-Throughput)|实际算力开销|实际算力利用率|实际算力开销(内核时间)|实际算力利用率(内核时间)|
| -------- | -------------- | -------------- | ------------ | ------ | ----- | -------- |--------|------|-------|------|
| {{ op_name }} | {{ dtype }}    | {{ shape_detail }}       | {{ no_warmup_latency}}  | {{ warmup_latency }} | {{ raw_throughput }} | {{core_throughput}} |{{ctflops}}|{{cfu}}|{{ktflops}}|{{kfu}}|


