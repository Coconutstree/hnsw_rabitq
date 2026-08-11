# Primary4 4-bit -> 4-bit D 构图优化说明

本文档记录当前 `hnsw_rabitq` 中 D 实验的优化实现。它不只包含
Primary4 对称距离的 AVX-512 VNNI 内核，也包含这次新接上的 HNSW
构图 prepared/batch 路径和 striped 统计计数。当前目标是让
`RABITQ_BUILD_DISTANCE=symmetric4` 的 D 构图真正走优化后的
`Primary4 -> Primary4` 距离核。

需要特别说明：这里是在原有 D 写法上做加速优化，不是提出新的构图算法。
HNSW 的插入式构图流程、neighbor selection 规则、RaBitQ 对称距离公式、
索引格式、payload 格式和 residual sidecar 格式都保持不变；改变的是距离计算
和构图热路径的工程接线方式。

## 1. 当前结论

D 实验使用 `4-bit primary -> 4-bit primary` 对称距离构图，block16
residual4 只用于最终 rerank。当前改动是在原有 D 写法上加速，包含三层：

- packed-nibble SIMD 距离核：避免逐坐标解包成 `double`，直接在 4-bit
  packed byte 上计算 centered inner product。
- symmetric prepared build path：构图时对当前插入点或当前候选点 prepare
  一次，再对邻居列表批量计算 Primary4 距离。
- striped build distance counters：保留日志中的 distance call 统计，但避免
  多线程构图时所有线程抢同一个 atomic counter。

当前完整 DBpedia 日志已经重新生成，D 的 graph build 从同配置 C 的
`531.043 s` 降到 `103.086 s`，构图阶段加速 `5.15x`；总 build time 从
`557.319 s` 降到 `131.367 s`，加速 `4.24x`。D 与 C 的存储占用保持一致，
都是 `2579.28 MB`。

### A/B/C/D 方法表

| Case | 构图距离 | 查询返回/精排 | 图是否共享 | 备注 |
|---|---|---|---|---|
| A | Float32 query -> Primary4 code 非对称距离 | Primary4 only | 与 B/C 共享 | 不使用 residual rerank |
| B | Float32 query -> Primary4 code 非对称距离 | 全维单尺度 residual4 rerank | 与 A/C 共享 | `residual_block_size=2048` |
| C | Float32 query -> Primary4 code 非对称距离 | block16 residual4 rerank | 与 A/B 共享 | 当前主要 baseline |
| D | Primary4 code -> Primary4 code 对称距离 | block16 residual4 rerank | 独立图 | 当前优化目标 |

### 完整构图和存储表

数据来自现有日志：

- raw HNSW: `/home/kai3/coco/hnsw/build/hnsw_dbpedia_ef_400_M_32_omp64.log`
- A/B/C/D: `logs/dbpedia/dbpedia_{A,B,C,D}_ef_400_M_32.log`

| 方法 | M | efConstruction | 构图距离 | Graph construction time s | Build time s | Index storage MB | 构图距离统计 |
|---|---:|---:|---|---:|---:|---:|---|
| hnsw | 32 | 400 | raw float32 L2 | 416.680 | 427.211 | 6356.12 | raw float in index |
| A | 32 | 400 | asymmetric4 | 531.043 | 557.319 | 2579.28 | `asymmetric_distance_calls=10860469602` |
| B | 32 | 400 | asymmetric4 | 531.043 | 27.858 | 2327.82 | 复用 A/C 图；Build time 不是 full rebuild |
| C | 32 | 400 | asymmetric4 | 531.043 | 557.319 | 2579.28 | 与 A 同图，`encoded_distance_calls=0` |
| D optimized | 32 | 400 | symmetric4 | 103.086 | 131.367 | 2579.28 | `symmetric_prepared_distance_calls=10875130926` |

### C vs D 优化收益表

| 指标 | C asymmetric4 | D optimized symmetric4 | D/C |
|---|---:|---:|---:|
| Graph construction time s | 531.043 | 103.086 | 0.194x |
| Graph build speedup | 1.00x | 5.15x | 5.15x |
| Build time s | 557.319 | 131.367 | 0.236x |
| Build speedup | 1.00x | 4.24x | 4.24x |
| Index storage MB | 2579.28 | 2579.28 | 1.000x |
| highest Recall@10 | 0.99546 | 0.99535 | -0.00011 |
| QPS at highest recall point | 162.419 | 164.805 | 1.015x |

### Recall/QPS 表

| 方法 | ef | Recall@10 | QPS | total us/query |
|---|---:|---:|---:|---:|
| hnsw | 30 | 0.95380 | 1010.81 | 989.302 |
| hnsw | 100 | 0.98820 | 390.87 | 2558.420 |
| hnsw | 460 | 0.99770 | 109.00 | 9174.040 |
| A | 30 | 0.93082 | 1335.32 | 748.885 |
| A | 100 | 0.96368 | 554.760 | 1802.580 |
| A | 460 | 0.96996 | 164.871 | 6065.350 |
| B | 30 | 0.91970 | 1178.32 | 848.664 |
| B | 100 | 0.95024 | 468.071 | 2136.430 |
| B | 460 | 0.95567 | 157.565 | 6346.600 |
| C | 30 | 0.94909 | 1264.46 | 790.854 |
| C | 100 | 0.98749 | 505.12 | 1979.730 |
| C | 460 | 0.99546 | 162.419 | 6156.910 |
| D optimized | 30 | 0.95072 | 1233.24 | 810.872 |
| D optimized | 100 | 0.98758 | 505.143 | 1979.640 |
| D optimized | 460 | 0.99535 | 164.805 | 6067.770 |

### 距离核微基准表

原始优化说明中的 hot single-pair 微基准用于解释距离核本身的算术收益。它不是
完整 HNSW 构图时间。

| 距离核 | 10M 次耗时 s | ns/distance | 相对旧实现 |
|---|---:|---:|---:|
| 旧 Primary4 -> Primary4 标量 double 解包 | 54.962083 | 5496.208 | 1.00x |
| AVX-512 VNNI packed-nibble | 0.849696 | 84.970 | 64.68x |

小切片 sanity test 中，DBpedia 20k base、`M=16`、`efConstruction=80`、
160 线程，D 的 graph build 为 `0.691706 s`，并确认
`symmetric_prepared_distance_calls=30060844`。

因此，论文或实验表述中应把它描述为 D 的实现优化或系统优化，例如
`optimized symmetric4 construction`，而不是把它描述成一个新的 ANN 图构建方法。

## 2. Primary4 数据布局

`SequentialNibble` 布局中，一个 byte 保存两个 4-bit 量化值：

```text
packed byte
+---------------+---------------+
| high nibble   | low nibble    |
| coordinate 1  | coordinate 0  |
| bits [7:4]    | bits [3:0]    |
+---------------+---------------+
```

每个 nibble 是 `x in [0, 15]` 的无符号整数，centered 值是 `x - 7.5`。
DBpedia 的原始维度是 1536，RaBitQ 当前补齐后的 `code_dim` 是 2048，因此
Primary4 code 为 1024 bytes，不含 encoded header。

这里算的是带尺度恢复的数值内积，不是 Hamming 距离，所以不能用
`xor + popcount` 代替。

## 3. 数学等价变换

原始 centered code inner product 为：

```text
code_ip = sum_i (x_i - 7.5) * (y_i - 7.5)
```

定义 doubled centered code：

```text
a_i = 2 * x_i - 15
b_i = 2 * y_i - 15
```

则：

```text
4 * code_ip = sum_i a_i * b_i
            = sum_i (2 * x_i - 15) * b_i
            = 2 * sum_i x_i * b_i - 15 * sum_i b_i
```

右侧只包含整数：

- `x_i` 是 `[0, 15]` unsigned byte。
- `b_i` 是 `[-15, 15]` signed byte。
- 这正好匹配 AVX-512 VNNI `vpdpbusd` 的 unsigned byte * signed byte
  累加到 int32 语义。

直接计算 `a_i * b_i` 会遇到 VNNI 第一个 byte 操作数必须是 unsigned 的限制。
上面的改写保留左侧为 unsigned nibble，同时额外累加 `sum_i b_i`，避免逐坐标
转换成浮点数。

## 4. VNNI 数据流

AVX-512 VNNI 主循环每轮读取左右各 64 个 packed bytes：

```text
lhs/rhs each 64 packed bytes
        |
        +-- mask 0x0f ------------> 64 low nibbles
        +-- shift 4 + mask 0x0f --> 64 high nibbles

rhs low/high: b = 2 * y - 15
        |
        +-- vpdpbusd(lhs, b)  --> sum x_i * b_i
        +-- vpdpbusd(ones, b) --> sum b_i

final integer = 2 * sum x_i * b_i - 15 * sum b_i
code_ip       = 0.25 * final integer
```

每轮使用四次 `_mm512_dpbusd_epi32`：

- lhs low nibble * rhs low centered nibble。
- lhs high nibble * rhs high centered nibble。
- ones * rhs low centered nibble。
- ones * rhs high centered nibble。

一轮处理 128 个 4-bit 坐标；`code_dim=2048` 时主循环 16 轮。不足 64 个
packed bytes 的尾部由 scalar helper 处理。

数值范围安全：`2x - 15 in [-15, 15]`，2048 维时
`abs(sum_i a_i * b_i) <= 460800`，不会使 int32 accumulator 溢出。

## 5. AVX2 和 scalar fallback

AVX2 路径每轮读取 16 个 packed bytes，处理 32 个坐标：

- 对两侧 nibble 都计算 `2x - 15`。
- 用 `_mm256_cvtepi8_epi16` 扩展成 int16。
- 用 `_mm256_madd_epi16` 做乘法和相邻项累加。
- 水平求和后处理 scalar tail。

scalar fallback 也保持 packed-byte 实现：每次读取一个 byte，分别处理
low/high nibble，累加 `(2x - 15) * (2y - 15)`。

当前是编译期 ISA 分派：

```text
AVX512F + AVX512BW + AVX512VNNI
    -> AVX2
        -> scalar
```

这不是运行时 CPUID dispatch。使用 `-march=native` 编译时，二进制可移植性取决于
编译机器和运行机器的 ISA 兼容性。

## 6. 为什么可以逐 bit 一致

新内核只替换 centered code dot 的计算方式：

- 旧路径累加 `(x - 7.5) * (y - 7.5)`。
- 新路径精确计算整数 `4 * code_ip`，最后乘以 `0.25`。

每一项都是 `0.25` 的整数倍，2048 维累计仍在 double 的精确整数表示范围内，
因此中间 code inner product 可以保持逐 bit 一致。`distanceBetweenEncoded()`
里的 header 读取、norm、sqrt、scale 恢复、`[-1, 1]` clamp 和最终 L2 公式不变。

虽然 VNNI 改写式看起来对 lhs/rhs 不对称，但整数代数上等于
`sum_i (2x_i - 15) * (2y_i - 15)`，所以最终距离仍然对称。

## 7. D 构图路径接线

只有距离核还不够。D 的 HNSW 构图必须真的把插入流程接到这个内核上，否则
完整构图时间不会接近预期。

这部分仍然沿用原来的 HNSW `addPoint -> searchBaseLayer ->
getNeighborsByHeuristic2 -> mutuallyConnectNewElement` 流程；prepared/batch 只替换
其中的距离调用方式，不改变候选集合语义和选边规则。

当前接线如下：

```text
RABITQ_BUILD_DISTANCE=symmetric4
  -> tests/cpp/sift_1b.cpp
     -> appr_alg->setSymmetricBuildPrepared(true)
        -> HierarchicalNSW::addPoint(...)
           -> prepare_symmetric_build_query(encoded_query)
           -> searchBaseLayer(..., symmetric_query_context)
              -> symmetric_build_distance_batch(...)
           -> getNeighborsByHeuristic2(...)
              -> prepared candidate-to-selected distance with early stop
           -> mutuallyConnectNewElement(...)
              -> batch distance for reverse-link replacement
```

对应实现点：

| 功能 | 位置 |
|---|---|
| symmetric prepared 虚接口 | `hnswlib/hnswlib.h` |
| RaBitQ prepared context 和 batch distance | `hnswlib/space_rabitq.h` |
| HNSW 构图路径接入 | `hnswlib/hnswalg.h` |
| RaBitQHNSW wrapper | `hnswlib/rabitq_hnsw.h` |
| D 模式启用和日志字段 | `tests/cpp/sift_1b.cpp` |
| smoke test 覆盖 | `tests/cpp/rabitq_hnsw_smoke.cpp` |

D 构图日志应出现：

```text
build_distance=extended_rabitq_symmetric4
build_query=primary4 build_database=primary4 ... symmetric_build_prepared=1
build_stage=symmetric4_graph_build ... asymmetric_distance_calls=0 ... symmetric_prepared_distance_calls=...
```

## 8. striped counter 优化

构图热路径会产生大量 distance call 统计。旧写法如果所有线程都对同一个
`std::atomic<uint64_t>` 做 `fetch_add`，距离核加速后 counter 本身会成为明显瓶颈。

当前 `HierarchicalNSW` 使用 256 个 cache-line 对齐的 counter stripe：

```text
BuildDistanceCounterStripe {
  asymmetric
  encoded
  symmetric_prepared
}
```

OpenMP 构图时按 `omp_get_thread_num() % 256` 选择 stripe；读取日志统计时再汇总
所有 stripe。这样保留了：

```text
asymmetric_distance_calls
encoded_distance_calls
symmetric_prepared_distance_calls
```

同时避免多线程持续抢同一个 atomic cache line。

## 9. 正确性验证

smoke test 当前覆盖：

- `prepare_symmetric_build_query()` 与 one-shot symmetric distance 等价。
- symmetric D 构图能触发 `symmetricPreparedBuildDistanceCalls() > 0`。
- SIMD `distanceBetweenEncoded()` 与 scalar reference 逐 bit 一致。
- `distance(lhs, rhs)` 与 `distance(rhs, lhs)` 逐 bit 一致。

编译和测试命令：

```bash
cd /home/kai3/coco/hnsw_rabitq

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target main rabitq_hnsw_smoke_test distance_kernel_benchmark -j 64
ctest --test-dir build -R '^rabitq_hnsw_smoke$' --output-on-failure
```

可用反汇编确认 VNNI 指令：

```bash
objdump -d -C build/distance_kernel_benchmark | rg -i 'vpdpbusd'
```

距离核微基准命令格式：

```bash
./build/distance_kernel_benchmark \
  /home/kai3/coco/data/dbpedia_openai1536/dbpedia_openai1536_base.fvecs \
  <RABITQ_STATE_OR_INDEX_RABITQ_FILE> \
  10000000 5 1536 0 1 1000000
```

## 10. 完整 D 实验命令

当前 `logs/dbpedia/dbpedia_D_ef_400_M_32.log` 已有一次完整 D optimized 结果。
如果需要覆盖重跑，使用：

```bash
cd /home/kai3/coco/hnsw_rabitq

mkdir -p logs/dbpedia

OMP_NUM_THREADS=64 \
OMP_PROC_BIND=close \
OMP_PLACES=cores \
RABITQ_FORCE_REBUILD=1 \
RABITQ_DATASET=dbpedia_openai1536 \
RABITQ_CENTROID_COUNT=1 \
RABITQ_EF_CONSTRUCTION=400 \
RABITQ_M=32 \
RABITQ_BUILD_DISTANCE=symmetric4 \
RABITQ_ABC_ABLATION=D \
RABITQ_PAPER_PRUNE_COMPARE=active \
RABITQ_PAPER_EPSILON0=1.9 \
RABITQ_BUILD_REPORT_EVERY=10000 \
./build/main > logs/dbpedia/dbpedia_D_ef_400_M_32.log 2>&1
```

`RABITQ_BUILD_REPORT_EVERY=10000` 只控制构图过程日志。它不会改变算法、索引或
recall。若不设置，默认是 `0`，构图中间只会看到：

```text
Entering symmetric graph addpoint loop count=990000 report_every=0
```

然后等待阶段结束后才输出：

```text
build_stage=symmetric4_graph_build us=...
Graph construction time: ...
Build time: ...
```

看 `quantized_graph_build_setup ... reuse_graph=0` 可以确认没有复用共享图。

## 11. 性能解读边界

需要区分两个数字：

- 热距离核微基准：只反复计算同一对 Primary4 code 的距离，能体现 VNNI kernel
  的算术加速。
- 完整 HNSW D 构图：还包含 priority queue、visited set、随机邻接表访问、
  link-list lock、neighbor heuristic、反向边更新、NUMA 和日志计数等成本。

因此 hot kernel 的几十倍加速不能直接等价为完整构图的几十倍加速。当前优化的重点是：

- 保证 D 构图中大量 query-to-neighbor 和 reverse-link candidate distance
  走 prepared/batch Primary4 距离核。
- 保留 neighbor heuristic 的早停行为，避免为了 batch 而多算明显没必要的距离。
- 降低统计计数在多线程下的 cache-line contention。

## 12. 当前边界

- VNNI 优化只覆盖 `SequentialNibble` 的 `Primary4 -> Primary4` centered dot。
- `Turbo128` 布局仍走原有布局感知路径。
- 当前是编译期 ISA dispatch，不是运行时 CPUID dispatch。
- 不改变 index 文件格式、payload 格式、residual sidecar 格式。
- D 构图仍然是 HNSW 插入式构图，不是 QG-style offline builder。
- 当前完整 DBpedia `M=32, efConstruction=400` D optimized 日志已经生成；如果代码继续变动，需要按第 10 节命令覆盖重跑。
