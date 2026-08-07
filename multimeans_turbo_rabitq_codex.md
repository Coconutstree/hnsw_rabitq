# Multi-Means RaBitQ + Turbo 4-bit Layout 实施方案

## 1. 任务目标

在不修改 HNSW 搜索逻辑、不改变现有 4-bit RaBitQ 距离公式、不引入 1→2→4 bit 动态精化的前提下，完成两项独立优化并做严格消融：

1. Multi-Means RaBitQ：通过多个局部质心减小残差和主码误差，验证能否在相同 Recall 下减少 ef、visited nodes 和 distance computations。
2. Turbo 4-bit Layout：只调整 4-bit code 的内存排列和 SIMD 解码方式，验证能否在搜索路径和距离计算次数不变时降低单次距离成本。

最终形成四组结果：

- Baseline：K=1 + Sequential layout
- Multi-Means：K=K* + Sequential layout
- Turbo：K=1 + Turbo128 layout
- Multi-Means + Turbo：K=K* + Turbo128 layout

核心原则：Multi-Means 负责减少“需要计算多少个节点”，Turbo 负责降低“每个节点的距离计算有多贵”。两项必须先单独验证，再组合。

## 2. 本阶段禁止修改的内容

为保证归因清晰，本阶段禁止修改：

- HNSW candidate_set 和 top_candidates 的排序规则
- lowerBound 判断逻辑
- 搜索停止条件
- 邻居扩展顺序
- ef 的语义
- HNSW 图结构、层级、entry point 和邻接表
- 现有 4-bit RaBitQ 距离估计公式
- residual rerank 的数学公式
- nested4x4 的多质心支持
- 1-bit short filter、动态精化和渐进距离策略

本阶段只允许修改：质心训练与加载、RaBitQ payload 编码、QueryContext 初始化方式、4-bit 主码布局、SIMD 距离内核、性能统计和测试程序。

## 3. 相关文件

重点修改：

- space_rabitq.h
- rabitq_hnsw.h
- hnswalg.h

原则上不修改：

- hnswlib.h
- space_l2.h
- space_ip.h
- stop_condition.h
- visited_list_pool.h

另需新增：

- K-means 质心训练或质心加载工具
- Sequential→Turbo layout 转换工具
- 距离内核 microbenchmark
- Multi-Means 单元测试
- Turbo layout 单元测试
- 统一 Recall/QPS benchmark
- CSV 结果输出

## 4. 阶段 0：建立基线

### 4.1 固定实验设置

第一轮使用 DBpedia/OpenAI-1536 1M：

- N=1M
- D=1536
- Float32 构图
- M=16
- efConstruction=200
- k=100
- 查询数至少 1000
- rerank_candidates 固定，例如 100
- ef={32,64,96,128,192,256,320,460,640}
- 单线程查询作为主要结果
- 每组预热后运行 5 次，报告中位数
- 固定 CPU 核心，避免其他高负载任务

### 4.2 必须记录的指标

每个 ef 输出：

- Recall@100
- QPS
- avg latency
- P50/P95/P99 latency
- visited nodes
- distance computations
- prepare_query_us
- traversal_us
- rerank_us
- total_query_us
- bytes_per_vector

不要在每次距离计算内部使用 std::chrono。查询阶段使用外层计时；单次距离成本由独立 microbenchmark 测量。

### 4.3 距离内核 microbenchmark

固定一个 query，对大量数据库向量连续调用 queryDistanceLong，至少完成 1000 万次距离计算，输出：

- ns/distance
- cycles/distance
- instructions/distance
- IPC
- L1/L2/LLC miss
- branch miss
- 内存带宽

使用编译参数：

- -O3
- -march=native

使用 perf stat 和 perf record 分析热点。

### 4.4 基线结论

阶段 0 必须回答：

- queryDistanceLong 及其 SIMD 内核占总查询时间多少
- 图随机访问和优先队列占多少
- residual rerank 占多少
- 达到目标 Recall 所需的最小 ef
- 当前主要瓶颈是访问节点数还是单次距离成本

## 5. 阶段 1：Multi-Means RaBitQ

### 5.1 基本定义

原始单质心编码：

- residual = x - c

Multi-Means：

- 为数据库训练 K 个质心 c_0...c_{K-1}
- 每个向量选择最近质心 m(x)
- residual = x - c_m(x)
- 使用当前 4-bit RaBitQ 编码 residual
- payload 中保存 centroid_id

目标不是只降低 MSE，而是降低主码导航误差，使相同 Recall 所需的 ef 和访问节点数下降。

### 5.2 质心训练

对数据库随机抽样训练 K-means：

- sample_size=min(100000,N)
- K-means++ 初始化
- Lloyd 迭代 20 次
- 固定随机种子
- 保存 Float32 质心文件
- 文件头保存 magic、version、dimension、centroid_count

测试：

- K={1,8,16,32,64,100}

若使用 Cosine/IP 且数据库向量已归一化，使用球面 K-means或每轮重新归一化质心，不能直接使用未归一化 L2 质心。

### 5.3 构建流程

对每个 K：

1. 加载同一份 Float32 HNSW 图。
2. 创建 RaBitQSpace(dim,K,...,SequentialNibble)。
3. 调用 setCentroids 加载质心。
4. 对所有数据库向量调用 assignCentroid。
5. 相对对应质心计算 residual。
6. 使用当前 4-bit RaBitQ 编码。
7. payload 保存 centroid_id。
8. 使用 importGraphFromFloatIndexWithPayloads 复制原图并替换 payload。
9. 保存为独立索引。

建议命名：

- dbpedia_k1_seq
- dbpedia_k8_seq
- dbpedia_k16_seq
- dbpedia_k32_seq
- dbpedia_k64_seq
- dbpedia_k100_seq

必须验证所有索引的：

- 内部节点 ID 一致
- label 一致
- 图层级一致
- entry point 一致
- 邻接表一致

### 5.4 Eager Multi-Means

先保留当前 prepare_query 的处理方式：

- 查询开始时为全部 K 个质心创建 QueryContext
- 作为正确性基线
- 统计 prepare_query_us

### 5.5 Lazy Multi-Means

在 Eager 正确后实现 Lazy QueryContext。

PreparedQuery 保存：

- rotated_query
- raw_query_norm_sqr
- vector<QueryContext> centroid_queries
- vector<uint8_t> ready
- active_centroid_count

第一次访问 centroid_id=m 时：

1. 检查 ready[m]。
2. 未初始化则构建 q-c_m 对应的 QueryContext。
3. 设置 ready[m]=1。
4. active_centroid_count++。
5. 后续复用。

注意线程安全：PreparedQuery 当前为 thread_local，每个查询线程只操作自己的实例，不需要全局锁。

### 5.6 Multi-Means 单元测试

必须完成：

#### K=1 兼容性

- 使用原全局质心
- 编码结果与原实现一致或浮点误差范围内一致
- 距离一致
- Recall、visited nodes、distance computations 一致

#### 质心分配正确性

随机抽取 1000 个向量：

- assignCentroid 结果
- 暴力计算到全部质心的最小距离
- 两者必须一致

#### 距离正确性

随机 query/vector：

- 比较量化距离
- 比较解码后距离
- 比较 Float32 真值
- 检查 NaN、Inf、越界 centroid_id

#### 保存加载

- centroid_count 一致
- centroid values 一致
- centroid_id 一致
- 加载前后搜索结果一致

#### Eager/Lazy 一致性

- 相同 query 和索引
- 距离结果一致
- top-k 结果一致
- 仅 prepare_query 和初始化时机不同

### 5.7 Multi-Means Benchmark

每个 K 分别测试 Eager 和 Lazy，输出：

- quantization_MSE
- average_relative_distance_error
- Recall-ef
- Recall-QPS
- Recall-visited_nodes
- Recall-distance_computations
- prepare_query_us
- active_centroid_count
- bytes_per_vector

重点比较目标 Recall：

- 0.90
- 0.95
- 0.98，或当前能稳定达到的最高 Recall

对每个目标 Recall，寻找最小 ef，计算：

- ef_reduction=(ef_K1-ef_K)/ef_K1
- visited_reduction=(V_K1-V_K)/V_K1
- distance_reduction=(D_K1-D_K)/D_K1

### 5.8 Multi-Means 保留标准

满足以下任一主要条件且最终 QPS 提升才保留：

- 相同 Recall 下 visited nodes 下降至少 10%
- 相同 Recall 下最小 ef 下降至少 10%
- 最终 QPS 提升至少 5%
- P95 latency 不恶化
- prepare_query_us 不吞掉搜索收益

停止继续增加 K 的条件：

- MSE 下降但 ef 和 visited nodes 不下降
- visited nodes 下降但 QPS 不提升
- active centroid 数量接近 K
- prepare_query 占总查询时间超过 10%
- K 增大后收益趋近于零或变差

最终选择：在相同目标 Recall 下 QPS 最高的 K*，而不是 MSE 最低的 K。

## 6. 阶段 2：Turbo 4-bit Layout

### 6.1 基本要求

Turbo 只改变 4-bit code 的物理排列，不改变：

- 每维量化值
- long distance 数学公式
- short/remaining 的定义
- HNSW 搜索逻辑
- 每个向量主码字节数

第一版只优化主码热路径，不修改 residual code 布局。

### 6.2 Layout 枚举

在 RaBitQSpace 中增加：

- SequentialNibble
- Turbo128

索引文件必须保存 layout_id。升级 magic/version，加载旧索引时默认 SequentialNibble。加载 Turbo 索引时必须选择对应内核，禁止错误布局和错误内核组合。

### 6.3 Turbo128 物理布局

每 128 维为一个 block，共 64 字节，解释为 16 个 uint32_t word。

对于 lane=0...15：

- word[lane] 保存 code[lane], code[16+lane], code[32+lane], ..., code[112+lane]
- 每个 code 占 4 bit

打包关系：

- word[lane] 的第 slot 个 nibble 保存 code[slot*16+lane]
- slot=0...7

距离计算时：

- 一次加载 16 个 uint32_t，即 64 字节
- shift 0/mask 得到第 0-15 维
- shift 4/mask 得到第 16-31 维
- 依次到 shift 28
- 每轮得到 16 个完整 4-bit code
- 与对应连续 16 个 query float 累加

### 6.4 新增函数

在 space_rabitq.h 中增加：

- packSequentialNibble
- packTurbo128
- unpackTurbo128Scalar
- convertSequentialToTurbo
- dotLongTurbo128Scalar
- dotLongTurbo128Avx512
- shortCodeIpTurbo128Scalar
- shortCodeIpTurbo128Avx512
- layout dispatch

保留当前 Sequential 距离内核作为基线。

建议实现独立转换工具：读取已有 Sequential 索引，在不重新量化的情况下转换 payload 主码布局并保存 Turbo 索引。这样可保证 Turbo 前后 code 数值完全一致。

### 6.5 QueryContext

第一版可以继续保留 rotated_residual 的逻辑顺序，不必永久重排 query。

每处理一个 Turbo128 block：

- slot 0 使用 query[offset+0...15]
- slot 1 使用 query[offset+16...31]
- 依次处理 8 个 slot

Sequential 所需的 even/odd query 数组继续保留，保证两套布局可并存。

### 6.6 AVX512 与 fallback

第一版优先完成 AVX512。

- 支持 AVX512：使用 Turbo128 AVX512
- 不支持 AVX512：使用 Turbo128 scalar，不能直接调用 Sequential decoder
- 后续如有需要再实现 AVX2 Turbo64

### 6.7 Turbo 单元测试

#### 编码可逆

随机生成 4-bit code：

- Sequential pack
- Turbo pack
- Turbo unpack
- 每一维 code 必须完全一致

覆盖：

- D=128
- D=256
- D=960
- D=1536
- padding 后 code_dim

#### 距离一致性

随机生成至少 10000 组 query/code，比较：

- Sequential scalar
- Sequential SIMD
- Turbo scalar
- Turbo AVX512

要求：

- 最大绝对误差不超过 1e-4，或使用当前数值误差标准
- 不出现 NaN/Inf
- 差异只能来自浮点累加顺序

#### 搜索一致性

固定同一张图、同一份量化 code、同一 ef：

- Sequential 搜索
- Turbo 搜索

比较：

- Recall 差异小于 0.001
- visited nodes 基本一致
- distance computations 一致
- top-k 结果高度一致
- 候选集合高度一致

#### 索引格式

- 旧 Sequential 索引可加载
- 新 Turbo 索引可加载
- 错误 layout 不会被误读
- 保存加载后距离和搜索结果一致

### 6.8 Turbo Microbenchmark

比较：

- Sequential scalar
- Sequential AVX512
- Turbo scalar
- Turbo AVX512

输出：

- ns/distance
- cycles/distance
- instructions/distance
- IPC
- L1/L2/LLC miss
- memory bandwidth

### 6.9 Turbo End-to-End Benchmark

固定：

- K=1
- 同一张图
- 同一份量化 code
- 同一 ef
- 同一 rerank_candidates

只更改 layout，输出：

- Recall
- QPS
- P50/P95/P99
- traversal_us
- visited nodes
- distance computations
- bytes_per_vector

### 6.10 Turbo 保留标准

建议达到：

- microbenchmark 单次距离至少提升 1.2x
- 端到端 QPS 至少提升 8%
- Recall 差异小于 0.001
- visited nodes 和 distance computations 基本不变
- 主码内存不增加

若单次距离提升明显但端到端提升不足 3%，记录结论：当前主要瓶颈不是 SIMD 解包，而是随机访存或访问节点数，不继续过度优化内核。

## 7. 阶段 3：组合实验

确定最佳 K* 后构建四组：

- K=1 + Sequential
- K=K* + Sequential
- K=1 + Turbo128
- K=K* + Turbo128

必须使用：

- 同一 Float32 图结构
- 相同 ef 集合
- 相同 rerank_candidates
- 相同 query 和 ground truth

分析：

- Multi-Means 是否降低 distance computations
- Turbo 是否降低 cycles/distance
- 二者是否近似叠加
- Multi-Means 减少访问后 Turbo 收益是否缩小
- Turbo 加速后随机访存是否成为新瓶颈

可使用以下性能分解：

- T_total ≈ N_dist × C_dist + T_graph + T_rerank + T_prepare
- Multi-Means 主要影响 N_dist
- Turbo 主要影响 C_dist

## 8. hnswalg.h 修改要求

只增加低开销统计，不改变搜索决策：

- per-query visited_nodes
- per-query distance_computations
- traversal_us
- rerank_us
- total_query_us

如有必要，通过 getter 或 metrics struct 返回，不在热循环打印日志。

禁止修改：

- candidate_set
- top_candidates
- lowerBound
- stop condition
- 邻居处理顺序

## 9. rabitq_hnsw.h 修改要求

增加或明确：

- centroid_count 参数
- layout 参数
- loadCentroids 或 setCentroids 接口
- save/load 时的配置校验
- benchmark 所需 metrics 接口

保持 importGraphFromFloatIndexWithPayloads 的使用方式，保证所有实验复制同一张 Float32 图。

暂不让 nested4x4 支持多质心。

## 10. space_rabitq.h 修改要求

需要完成：

- layout enum 和 layout_id
- 新索引格式兼容
- Turbo pack/unpack
- Turbo scalar/AVX512 距离内核
- Sequential/Turbo dispatch
- Eager/Lazy Multi-Means QueryContext
- active centroid 统计
- K=1 行为兼容
- 所有新增接口的边界检查

禁止删除现有 Sequential 实现，必须保留用于消融和回归测试。

## 11. CSV 输出格式

每次实验输出一行，至少包含：

- dataset
- layout
- centroid_count
- centroid_mode=eager/lazy
- M
- efConstruction
- ef
- k
- rerank_candidates
- recall
- qps
- avg_latency_us
- p50_us
- p95_us
- p99_us
- visited_nodes
- distance_computations
- prepare_query_us
- traversal_us
- rerank_us
- total_query_us
- active_centroids
- bytes_per_vector
- quantization_mse
- average_relative_distance_error
- ns_per_distance
- cycles_per_distance

## 12. 最终图表

至少绘制：

- Recall-QPS
- Recall-visited nodes
- Recall-distance computations
- Recall-ef
- centroid_count-active_centroids
- layout-cycles_per_distance

## 13. 最终决策

### Multi-Means 有效，Turbo 有效

保留组合方案，下一步进入 1→2→4 bit 渐进码和动态精化。

### Multi-Means 有效，Turbo 无效

主要瓶颈是访问节点过多。下一步优先优化导航精度和关键排序翻转。

### Multi-Means 无效，Turbo 有效

当前 4-bit 主码导航精度已基本够用，主要瓶颈是距离内核和内存布局。继续做 SIMD、预取和 batch distance。

### 二者都无效

当前瓶颈更可能是 HNSW 图访问、候选队列、随机访存，或量化目标与导航排序不匹配。下一步直接进入 graph-aware progressive quantization。

## 14. 严格执行顺序

1. 建立 K=1 Sequential 完整基线。
2. 增加 K-means 质心训练和加载。
3. 验证 K=1 完全兼容。
4. 实现 Eager Multi-Means。
5. 测试 K=8、16、32、64、100。
6. 实现 Lazy Multi-Means。
7. 确定最佳 K*。
8. 固定 K=1，实现 Turbo128。
9. 完成 Turbo 距离一致性测试。
10. 完成 Turbo microbenchmark。
11. 完成 Turbo end-to-end benchmark。
12. 组合 Turbo + K*。
13. 完成四组消融。
14. 根据 ef、visited nodes、distance computations 和 cycles/distance 决定下一阶段。

## 15. Codex 交付要求

每完成一个阶段，先提交独立 commit，不要一次性混合全部修改。

建议 commit 顺序：

- baseline metrics
- multimeans centroid training and loading
- multimeans eager query context
- multimeans lazy query context
- multimeans benchmarks
- turbo128 layout and scalar reference
- turbo128 avx512 kernel
- turbo128 index compatibility
- turbo benchmarks
- multimeans turbo ablation

每个 commit 必须：

- 可独立编译
- 单元测试通过
- 不改变未涉及模式的结果
- 输出清晰的 benchmark 数据
- 说明修改文件、核心逻辑、正确性验证和性能变化

最终不要只报告 QPS。必须同时报告 Recall、ef、visited nodes、distance computations、prepare_query_us、traversal_us、rerank_us 和 cycles/distance，确保能够判断性能提升的真实来源。
