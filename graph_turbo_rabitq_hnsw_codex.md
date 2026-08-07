# Graph-Turbo RaBitQ-HNSW：邻接感知的查询加速方案

## 0. 任务背景

当前系统采用以下流程：

1. 使用原始 Float32 向量构建 HNSW 图；
2. 图结构构建完成后保持不变；
3. 将每个节点的 Float32 payload 替换为 4-bit RaBitQ 编码；
4. 查询阶段使用量化距离执行 HNSW 导航；
5. 最终候选再执行 residual rerank。

已有实验结论：

- Multi-Means 在 DBpedia 上不能降低达到相同 Recall 所需的 `ef`；
- `visited nodes` 和 `distance computations` 基本不变；
- `K > 1` 只增加查询开销；
- 正式默认配置应为 `K=1`；
- Turbo128 在当前 CPU 和当前 RaBitQ 距离内核上比 Sequential 慢约 38%；
- 不继续迁移 Turbo LVQ 的 128 维重排方式。

本任务不再优化单个向量内部的 nibble 解包，而是吸收 Turbo LVQ 的核心思想：

> 数据布局和计算顺序必须围绕真实热路径设计。

当前系统的真实热路径是：

> 扩展一个 HNSW 节点，读取一组邻居 ID，随机访问多个邻居的量化 payload，计算距离并更新候选队列。

因此，本方案将优化单位从“一个向量的 128 维 block”改为“一次 HNSW 邻居扩展”。

---

# 1. 方案名称

建议暂定为：

**Graph-Turbo RaBitQ-HNSW**

更正式的描述为：

**Adjacency-Aware Priority Routing and Progressive Distance Evaluation for RaBitQ-HNSW**

核心机制：

1. Float32 图构建完成后，为每个节点生成一个轻量 `route_code`；
2. 在不增加邻接条目字节数的前提下，尝试将 `route_code` 写入邻居 ID 的空余高位；
3. 查询时先用 `route_code` 对当前节点的邻居进行低成本粗排序；
4. 优先计算更有希望的邻居，尽早收紧 `lowerBound`；
5. 对剩余邻居再使用已有的 1-bit short lower bound；
6. 只有不能安全排除的邻居才执行完整 4-bit long distance；
7. 第一阶段先实现 batch、prefetch 和优先级处理；暂不引入 2-bit 中间层。

---

# 2. 本阶段目标

## 假设 A：邻居批处理可以隐藏随机访存延迟

将逐邻居执行：

`读取 ID → 检查 visited → 读取 payload → 算距离 → 更新队列`

改为：

`批量收集邻居 → 批量预取 payload → 批量计算距离 → 更新队列`

目标是在不改变距离计算次数和搜索语义的情况下提高 QPS。

## 假设 B：粗路由排序可以更早收紧 lowerBound

用 8～12 bit `route_code` 粗略估计一组邻居的优先级，先计算 Top-P 邻居的完整 4-bit 距离，使 `top_candidates` 更早得到较好节点，从而让 `lowerBound` 更快变严格。

## 假设 C：更严格的 lowerBound 可以提高 short bound 的拒绝率

原 short filter 可能在 `lowerBound` 尚未收紧时执行，导致区间过宽、拒绝率低。

新的顺序为：

`route 粗排序 → Top-P 完整计算 → 更新 lowerBound → 剩余邻居 short bound → 必要时完整 4-bit`

目标是减少完整 4-bit long distance 的调用数量。

---

# 3. 硬性约束

## 3.1 不修改图拓扑

必须保持：

- 节点层级不变；
- entry point 不变；
- 邻接关系不变；
- `M` 不变；
- `efConstruction` 不变；
- label 到内部 ID 的映射不变；
- graph fingerprint 不变。

允许修改的是：

- 邻居条目的物理编码；
- 节点量化 payload 的布局；
- 查询时邻居处理顺序；
- 是否调用完整 4-bit 距离。

## 3.2 不复制完整量化向量到边上

每个节点的完整 4-bit RaBitQ payload 仍然只保存一份。

禁止：

- 在每条边后复制目标节点完整 4-bit code；
- 为每条边保存完整 short record；
- 建立会按平均度放大向量码内存的冗余边级 payload。

## 3.3 route code 第一版只能用于排序，不能直接剪枝

`route_code` 不提供严格数学界。

因此第一版只能用于：

- 选择优先计算的邻居；
- 调整 prefetch 顺序；
- 调整完整距离计算顺序。

禁止：

- 仅凭 Hamming 距离删除邻居；
- 用经验阈值硬剪枝；
- 将 route score 直接放入 HNSW candidate queue。

## 3.4 只有严格安全的 short lower bound 才能剪枝

用于跳过完整 4-bit 距离的 lower bound，必须严格下界于当前搜索实际使用的 full long distance。

需要确认：

- 它约束的是当前 `queryDistanceLong()` 的距离值；
- 不是只约束 Float32 真值；
- 不是只约束 residual 重建值；
- 对 L2 和 IP/cosine 的距离方向均正确；
- 对当前量化 scale、correction 和 centroid 配置均成立。

如果无法证明或通过测试验证，则只能统计，不能用于在线剪枝。

## 3.5 默认正式配置仍为 K=1

保留 Lazy QueryContext 代码用于实验和其他数据集，但正式热路径必须提供 K=1 专用分支：

- 不读取 `centroid_id`；
- 不做 Lazy ready 检查；
- 不索引多个 QueryContext；
- 不执行 Multi-Means 分支。

---

# 4. 阶段 0：代码审计

在修改任何邻居 ID 表示之前，先完成代码审计。

重点检查文件：

- `hnswalg.h`
- `hnswlib.h`
- `rabitq_hnsw.h`
- `space_rabitq.h`
- 相关保存与加载代码

## 4.1 检查邻居 ID 类型

确认：

- `tableint` 的实际类型；
- 是否固定为 32 bit；
- 邻接表中是否直接存储 `tableint`；
- 邻居 ID 高位是否被删除标记、锁标记、层级标记或其他用途占用；
- 所有读取邻居 ID 的位置；
- 所有写入邻居 ID 的位置；
- 保存、加载、图导入、图复制和图校验是否直接使用原始 ID。

输出一份审计结果：

- `sizeof(tableint)`；
- 最大允许 `max_elements`；
- 当前使用的保留位；
- 可安全使用的 route bits；
- 需要修改的所有读写点。

## 4.2 route bits 的计算

禁止写死 DBpedia 的 20-bit ID。

使用：

```text
id_bits = ceil_log2(max_elements)
route_bits = 32 - reserved_bits - id_bits
```

要求：

- `id_bits` 至少为 1；
- `route_bits >= 0`；
- 当 `route_bits < 4` 时自动禁用 packed route 模式；
- 保存索引时写入 `id_bits`、`route_bits` 和格式版本；
- 加载时验证 `max_elements < 2^id_bits`。

DBpedia 1M 的预期情况：

```text
id_bits ≈ 20
route_bits ≈ 12
```

但必须以实际代码审计结果为准。

## 4.3 如果高位不可用

如果高位已被占用，不要强行覆盖。

回退方案按优先级排列：

1. 重新定义一个 32-bit packed edge entry，在保持总宽度 32 bit 的前提下重新分配 ID 和标志位；
2. 如果无法保持 32 bit，则先只实现 batch + prefetch，不实现 route code；
3. 不要第一版直接添加 64-bit 邻居条目；
4. 不要悄悄增加索引内存后仍宣称“零内存增长”。

---

# 5. 邻居条目编码

在高位可安全使用时，增加统一辅助函数。

逻辑形式：

```text
packed_entry = node_id | (route_code << id_bits)
node_id      = packed_entry & id_mask
route_code   = (packed_entry >> id_bits) & route_mask
```

必须集中封装，不允许在搜索热路径之外散布手写位运算。

建议增加：

```cpp
struct PackedNeighborCodec {
    uint32_t id_bits;
    uint32_t route_bits;
    uint32_t id_mask;
    uint32_t route_mask;

    uint32_t pack(tableint id, uint32_t route_code) const;
    tableint unpack_id(uint32_t packed) const;
    uint32_t unpack_route(uint32_t packed) const;
};
```

具体命名可按项目风格调整。

## 5.1 必须修改的路径

至少检查并统一修改：

- level 0 邻接表读取；
- upper level 邻接表读取；
- 新增边；
- 删除边；
- 更新边；
- 图导入；
- Float32 图复制；
- 保存索引；
- 加载索引；
- graph fingerprint；
- 调试输出；
- 完整性检查；
- 所有直接把邻居条目当作 node ID 使用的位置。

## 5.2 graph fingerprint

graph fingerprint 应只基于解包后的 `node_id` 和图结构计算，不应把 `route_code` 纳入拓扑 fingerprint。

同时增加一个独立 fingerprint：

```text
route_code_fingerprint
```

用于确认 route code 保存和加载一致。

---

# 6. route code 的生成

## 6.1 第一版原则

第一版不训练复杂模型，不增加新的大矩阵，不做每节点独立投影。

直接复用 RaBitQ 已经计算的中心化/旋转后表示。

对每个数据库节点生成一个全局一致的符号码：

```text
route_code(x)[i] = sign(transformed_x[route_dims[i]])
```

查询使用同一组维度：

```text
route_code(q)[i] = sign(transformed_q[route_dims[i]])
```

然后使用：

```text
route_score(q, x) = popcount(route_code(q) XOR route_code(x))
```

较小的 Hamming distance 表示更高优先级。

## 6.2 route_dims 的选择

第一版实现三种可切换策略，用于消融。

### 策略 A：固定等间隔维度

从 `code_dim` 中等间隔选择 `route_bits` 个维度。

### 策略 B：全局高方差维度

在构建阶段从训练样本的旋转后表示中计算各维度方差，选择方差最大的 `route_bits` 个维度。

### 策略 C：当前 1-bit short code 中的代表位

从现有 1-bit primary code 中选取 `route_bits` 个代表维度。

第一阶段不要实现学习型 graph-aware 投影。先比较 A/B/C。

## 6.3 中心化和旋转必须一致

数据库 route code 和查询 route code 必须使用完全相同的处理流程：

- 相同 centroid；
- 相同随机旋转；
- 相同 padding；
- 相同维度索引；
- 相同符号判定；
- 相同零值规则。

K=1 时建议直接从唯一的 transformed residual 生成。

## 6.4 route code 的写入时机

完整流程：

1. Float32 HNSW 构图完成；
2. 图结构固定；
3. 生成所有节点的 RaBitQ payload；
4. 为每个节点计算唯一 `route_code[node_id]`；
5. 遍历所有邻接表；
6. 对每条边 `u -> v` 写入 `pack(v, route_code[v])`；
7. 保存量化索引。

---

# 7. 查询模式设计

必须保留多个运行模式，便于严格消融：

```text
baseline
batch_prefetch
route_priority
route_priority_short
```

后续可选：

```text
route_priority_short_2bit
```

但本阶段不实现 2-bit 在线路径。

---

# 8. 模式一：baseline

保持当前行为：

- 按邻接表原顺序处理；
- 每个未访问邻居直接执行完整 4-bit long distance；
- 按现有逻辑更新 `candidate_set` 和 `top_candidates`；
- 不读取 route code；
- 不执行 batch；
- 不执行 short pruning。

---

# 9. 模式二：batch_prefetch

目标：只优化访存和调用开销，不改变搜索语义。

## 9.1 邻居收集

扩展当前节点时：

1. 连续读取整段邻接表；
2. 解包 `node_id`；
3. 检查 visited；
4. 将未访问邻居写入固定 scratch 数组；
5. 保留其原始邻接表位置；
6. 按当前算法相同的时机标记 visited。

建议 scratch 数据：

```cpp
struct NeighborScratch {
    tableint ids[MAX_DEGREE];
    uint16_t original_pos[MAX_DEGREE];
    dist_t distances[MAX_DEGREE];
    uint32_t count;
};
```

不要在查询循环中进行堆分配。

如果 `M` 运行时可变，使用每线程预分配 scratch 或固定支持到实际最大 `maxM0_`。

## 9.2 payload prefetch

测试：

```text
prefetch_distance = 2, 4, 8, 16
```

优先 prefetch：

- 4-bit long code；
- 计算 long distance 必需的 header/factor；
- 不要预取 residual cold data。

## 9.3 batch distance 接口

在 `RaBitQSpace` 增加 K=1 专用批量接口，概念上为：

```cpp
void query_distance_long_batch_k1(
    const PreparedQuery& query,
    const tableint* ids,
    size_t count,
    dist_t* out_distances);
```

第一版允许内部循环调用现有 Sequential long kernel，但必须：

- 内联热路径；
- QueryContext 只获取一次；
- 公共 query 指针只解析一次；
- 公共 scale/correction 配置只解析一次；
- 使用循环展开；
- 配合 prefetch；
- 不进入 Multi-Means 分支。

## 9.4 原顺序回放

所有距离计算完成后，按照原邻接表顺序更新：

- `candidate_set`；
- `top_candidates`；
- `lowerBound`。

这一模式必须尽量保持 Recall、visited nodes、distance computations 和候选处理顺序一致。

## 9.5 验收标准

- Recall 差异不超过 0.001；
- visited nodes 基本一致；
- distance computations 一致；
- QPS 提升至少 5%；
- P95 延迟不恶化；
- 无索引内存增长。

---

# 10. 模式三：route_priority

目标：使用粗路由排序更早找到好邻居，但不减少完整距离计算次数。

## 10.1 查询 route code

在 `prepare_query()` 中为 K=1 查询生成一次 `query_route_code`。

要求：

- 不重新执行一套高成本变换；
- 从已有 transformed query 或 short code 中生成；
- 单独记录其开销。

## 10.2 粗评分

```text
route_score[i] = popcount(query_route_code XOR neighbor_route_code[i])
```

route score 越小，优先级越高。

## 10.3 Top-P 选择

不做完整排序，只选最优 Top-P：

```text
P = 2, 4, 8
```

优先使用固定长度插入选择，避免通用 `std::sort`。

## 10.4 处理顺序

先处理 Top-P：

1. prefetch Top-P 完整 payload；
2. 执行完整 4-bit long distance；
3. 立即更新候选队列；
4. 更新 `lowerBound`。

再处理剩余邻居：

- 仍然全部执行完整 4-bit long distance；
- 不使用 short bound；
- 对“route 顺序处理”和“剩余按原顺序处理”分别提供实验开关。

## 10.5 必须增加的统计

- route Top-1 是否包含 Float32 最优邻居；
- route Top-4 对 Float32 Top-1 的覆盖率；
- route Top-8 对 Float32 Top-4 的覆盖率；
- Top-P 完整计算后 `lowerBound` 的变化；
- route score 计算时间；
- Top-P 选择时间。

统计必须支持采样模式，避免影响正式 benchmark。

## 10.6 验收标准

- Float32 Top-1 进入 route Top-4 的比例较高；
- Float32 Top-4 进入 route Top-8 的覆盖率约 70%～80%；
- lowerBound 明显更早收紧；
- route 计算开销低；
- Recall 没有明显下降。

---

# 11. 模式四：route_priority_short

目标：先通过 route priority 收紧 lowerBound，再用严格 short lower bound 跳过完整 4-bit 距离。

## 11.1 处理流程

1. 收集未访问邻居；
2. 计算 route score；
3. 选 Top-P；
4. Top-P 执行完整 4-bit long distance；
5. 立即更新 `candidate_set`、`top_candidates` 和 `lowerBound`；
6. 对剩余邻居计算 short lower bound；
7. 若 `short_lower_bound > current_lowerBound`，则安全拒绝；
8. 否则执行完整 4-bit long distance；
9. 按当前 HNSW 规则更新队列。

## 11.2 threshold 单调性

确认：

- 对“小值更优”的内部距离表示，`lowerBound` 只会变小或保持；
- 一旦严格 lower bound 大于当前 `lowerBound`，后续更严格阈值不会使该节点重新可接受；
- IP/cosine 已转换为统一“小值更优”的语义。

## 11.3 visited 语义

默认保持现有行为：节点首次出现时即标记 visited。

前提是 short bound 对 full long distance 严格安全且阈值单调。

若无法确认，提供实验开关：

```text
mark_visited_before_bound
mark_visited_after_full_distance
```

## 11.4 必须增加的计数器

```text
neighbors_seen
neighbors_unvisited
route_scored
priority_full_distance_count
short_checked_count
short_reject_count
short_ambiguous_count
remaining_full_distance_count
total_full_distance_count
full_distance_saved
```

时间统计：

```text
route_score_us
top_p_select_us
short_bound_us
full_distance_us
queue_update_us
traversal_us
```

## 11.5 安全性 shadow test

在真正启用剪枝前：

- 在线逻辑仍计算完整 long distance；
- 同时记录 short 是否会拒绝；
- 检查拟拒绝节点是否满足：
  `full_long_distance > lowerBound_at_decision_time`。

要求：

```text
unsafe_reject_count == 0
```

## 11.6 保留标准

- unsafe reject 为 0；
- 完整 4-bit long distance 调用减少至少 20%；
- Recall 下降不超过 0.001；
- QPS 提升至少 5%～10%；
- short bound 的额外时间小于节省的 full distance 时间；
- P95 延迟不恶化。

---

# 12. 暂不实现的内容

第一阶段不要实现：

- 2-bit 中间码；
- 1→2→4 bit 在线渐进计算；
- SAQ 风格 code adjustment；
- 学习型 graph-aware 投影；
- 每节点独立 route_dims；
- 修改 HNSW 停止条件；
- 修改 candidate queue 数据结构；
- 修改图构建算法；
- 使用量化距离重新构图；
- 复制完整向量到边；
- 64-bit edge entry；
- 新的 residual routing。

---

# 13. 后续可扩展方向：1→2→4 bit

只有 `route_priority_short` 的 1-bit 模糊比例仍然很高时，再考虑第二阶段。

将原 4-bit 主码重新排列为：

```text
plane0: 1 bit/dim
plane1: 1 bit/dim
plane23: 2 bit/dim
```

DBpedia 1536 维：

```text
plane0  = 192 B/vector
plane1  = 192 B/vector
plane23 = 384 B/vector
total   = 768 B/vector
```

总主码大小不变。

不能直接假定当前 4-bit 高两位是有效 2-bit 量化。后续必须离线比较：

- 当前 4-bit 高两位；
- 独立原生 2-bit RaBitQ；
- 真正嵌套 1→2→4 编码；
- SAQ 风格可渐进前缀码。

重点评估：

- 邻居排序一致率；
- candidate 加入判断一致率；
- 区间宽度；
- 可安全拒绝比例；
- 搜索路径分叉率。

---

# 14. 代码修改建议

## 14.1 `space_rabitq.h`

增加或调整：

- K=1 专用 QueryContext 快速路径；
- `compute_route_code_from_database_vector()`；
- `compute_route_code_from_prepared_query()`；
- `query_distance_long_batch_k1()`；
- short lower bound shadow 验证；
- route code 配置保存；
- route dimension 保存与加载；
- 统计接口。

保持当前 Sequential 4-bit 内核、residual rerank 和完整 long distance 公式。

## 14.2 `hnswalg.h`

增加：

- packed neighbor 解码；
- 邻居 compact；
- thread-local/fixed scratch；
- payload prefetch；
- batch full distance；
- route score；
- Top-P 选择；
- route priority 模式；
- short pruning 模式；
- shadow validation；
- 完整统计计数器。

所有旧的直接读取 neighbor ID 的路径都必须解包，upper layer 和 level 0 都要检查。

## 14.3 `rabitq_hnsw.h`

在 Float32 图导入并替换 payload 后：

1. 计算所有节点的 route code；
2. 遍历邻接表；
3. 将目标节点 route code 写入 packed edge；
4. 保存 packed edge 元数据；
5. 保存 route dimensions 和生成策略；
6. 加载时验证格式；
7. 提供旧索引兼容路径。

## 14.4 索引格式

建议升级 magic/version，并保存：

```text
format_version
packed_neighbor_enabled
id_bits
route_bits
route_strategy
route_dim_count
route_dims[]
route_zero_sign_rule
```

旧索引自动进入 baseline，并禁用 route priority。

---

# 15. 单元测试

## 15.1 pack/unpack

要求：

```text
unpack_id(pack(id, code)) == id
unpack_route(pack(id, code)) == code
```

覆盖 0、最大 ID、最大 route code 和边界值。

## 15.2 图拓扑一致性

- 解包 ID 后逐边比较；
- 节点度数一致；
- 层级一致；
- entry point 一致；
- graph fingerprint 一致。

## 15.3 route code 一致性

- 数据库 route code 与重算结果一致；
- 查询 route code 与相同向量作为数据库向量时一致；
- 保存加载后 route fingerprint 一致。

## 15.4 batch distance 一致性

比较单条 Sequential long distance 和 batch long distance。

## 15.5 baseline 回归

旧 baseline 与新 baseline：

- Recall 基本一致；
- visited nodes 一致；
- distance computations 一致；
- 搜索结果重合率高；
- packed edge 不破坏图逻辑。

## 15.6 short bound shadow test

随机采样大量查询和邻居决策，unsafe reject 必须为 0。

---

# 16. Benchmark 设计

## 16.1 固定条件

建议：

- 同一张 Float32 HNSW 图；
- 同一 graph fingerprint；
- `M=16`；
- `efConstruction=200`；
- `K=1`；
- Sequential 4-bit；
- 相同 rerank 候选数；
- 相同查询集合和 ground truth；
- 固定 CPU 核；
- 预热；
- 单线程为主；
- 每组至少运行 5 次并报告中位数。

测试：

```text
ef = 64, 96, 128, 192, 256, 460
```

## 16.2 实验矩阵

### A. batch/prefetch

```text
baseline
batch_prefetch(pf=2)
batch_prefetch(pf=4)
batch_prefetch(pf=8)
batch_prefetch(pf=16)
```

### B. route bits

```text
route_bits = 6, 8, 10, 12
```

### C. route strategy

```text
equal_interval
high_variance
short_code_selected
```

### D. Top-P

```text
P = 2, 4, 8
```

### E. route + short

```text
route_priority
route_priority_short_shadow
route_priority_short_active
```

## 16.3 CSV 字段

```text
dataset
mode
route_bits
route_strategy
top_p
prefetch_distance
ef
recall
qps
avg_latency_us
p50_us
p95_us
p99_us
visited_nodes
distance_computations
neighbors_seen
neighbors_unvisited
priority_full_count
short_checked
short_rejected
short_ambiguous
remaining_full_count
full_distance_saved
unsafe_reject_count
route_score_us
short_bound_us
full_distance_us
queue_update_us
traversal_us
rerank_us
graph_fingerprint
route_fingerprint
index_bytes
```

---

# 17. 停止条件

## 17.1 batch_prefetch

停止条件：

- QPS 提升低于 3%；
- P95 延迟恶化；
- perf 显示 memory stall 没有改善；
- 代码复杂度明显增加但无收益。

## 17.2 route priority

停止条件：

- route Top-4 覆盖率低；
- lowerBound 未明显提前收紧；
- route score 和 Top-P 选择开销抵消收益；
- Recall 明显下降；
- visited nodes 增加。

## 17.3 short pruning

停止条件：

- unsafe reject 非 0；
- full 4-bit 调用减少低于 20%；
- QPS 无提升；
- short bound 时间过高；
- Recall 下降超过 0.001。

---

# 18. 推荐执行顺序

## Phase 0：审计

1. 检查 `tableint` 和邻居条目格式；
2. 检查高位占用；
3. 列出所有邻居 ID 读写位置；
4. 确认可用 route bits；
5. 输出审计结论。

## Phase 1：packed neighbor

1. 实现统一 pack/unpack；
2. 升级索引格式；
3. 修改全部邻接读写路径；
4. 先写 0 route code；
5. 验证 graph fingerprint 和搜索结果。

## Phase 2：K=1 batch + prefetch

1. 增加 K=1 快速路径；
2. 邻居 compact；
3. batch long distance；
4. payload prefetch；
5. 原顺序回放；
6. 测试 QPS。

## Phase 3：route code

1. 实现三种 route dimension 策略；
2. 生成数据库 route code；
3. 写入 packed edge；
4. 查询生成 query route code；
5. 测试局部排序覆盖率。

## Phase 4：route priority

1. 实现 Top-P；
2. 优先完整计算；
3. 不剪枝；
4. 测试 lowerBound 收紧速度；
5. 选择最佳 route bits、策略和 P。

## Phase 5：short shadow

1. 剩余邻居计算 short lower bound；
2. 不实际跳过 full distance；
3. 检查 unsafe reject；
4. 测量理论 full-distance 节省上限。

## Phase 6：short active

1. 仅在 shadow 安全后启用；
2. 实际跳过 full long distance；
3. 比较 Recall、QPS、visited 和 full-distance count；
4. 根据保留标准决定是否保留。

---

# 19. 最终需要回答的问题

1. 当前主要瓶颈是否为随机 payload 访存？
2. batch + prefetch 是否能隐藏一部分访存延迟？
3. 8～12 bit route code 是否能保持局部邻居优先级？
4. route priority 是否能更早收紧 lowerBound？
5. lowerBound 收紧后 short reject ratio 是否显著提高？
6. 完整 4-bit long distance 是否至少减少 20%？
7. QPS 是否在 Recall 基本不变的情况下提升？
8. 是否值得进入后续 1→2→4 bit 渐进量化阶段？

---

# 20. 最终实现原则

本任务的核心不是照搬 Turbo LVQ 的 128 维存储排列，而是迁移它的设计原则：

> 按照实际计算热路径重新组织数据和执行顺序。

对于当前 RaBitQ-HNSW：

- Turbo LVQ 的热路径是“单向量 SIMD 解包”；
- 当前系统的热路径是“HNSW 一次扩展中的多邻居随机访问”。

因此本方案的核心是：

> 利用 Float32 构图后已知的邻接关系，将轻量 route code 编入邻居记录；查询时先对一组邻居进行粗优先级判断，优先精算最有希望的邻居，提前收紧 lowerBound，再用严格 short bound 减少剩余完整 4-bit 距离计算。

第一阶段成功的标准不是量化 MSE 更低，而是：

- Recall 基本不变；
- full 4-bit distance count 明显减少；
- visited nodes 不增加；
- QPS 提升；
- 索引主码内存不增加；
- 图拓扑完全不变。
