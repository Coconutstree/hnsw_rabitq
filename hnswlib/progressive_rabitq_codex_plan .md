# HNSW + Extended RaBitQ 三级距离精化方案

## 1. 任务目标

请在现有的 HNSW + Extended RaBitQ 项目中，实现一种不依赖原始 float32 向量最终重排的查询方法。

目标查询流程：

```text
Short Code
    ↓
4-bit Long Code
    ↓
Residual Correction Code
    ↓
直接返回 Top-K
```

与当前方法相比，需要删除下面这个独立阶段：

```text
量化图搜索
    ↓
Top-L 候选
    ↓
读取原始 float32 向量
    ↓
精确距离重排
```

新的方法要求把距离精化嵌入 HNSW 搜索过程。

候选节点先使用便宜的短码距离进行初步判断；只有真正可能影响搜索路径或最终 Top-K 的候选，才升级到完整 4-bit 距离；仍然无法稳定排序的候选，再使用 residual correction code 提高距离精度。

最终返回结果时，不读取原始 float32 向量，也不执行独立的 float32 rerank。

构图方式必须保持为：构图距离使用 float32，节点最终存储使用量化码。实现过程中不得先生成一张完整 float32 HNSW，再复制图结构或替换节点数据。

---

# 2. 当前项目假设

当前项目已经具备以下能力：

1. HNSW 插入、候选搜索、邻居筛选和连边决策使用 float32 距离。
2. 构图过程中通过临时内存或外部原始数据文件访问已有节点的 float32 向量。
3. 每个节点在构图阶段同时生成 Extended RaBitQ 4-bit 编码。
4. 最终索引直接保存当前构建出的 HNSW 图结构和 4-bit 节点编码，不复制另一张图。
5. 最终索引不保存数据库节点的 float32 向量。
6. 查询向量保持 float32。
7. 查询阶段支持非对称距离计算：float32 query 对 4-bit database code。
8. 已存在短码距离或短码下界计算。
9. 已存在完整 4-bit long-code 距离计算。
10. 当前可选地从索引外部的原始向量文件读取 float32 数据，对候选集执行 rerank。

本任务不修改“float32 负责构图、量化码负责最终存储”的构图原则，也不新建或复制第二张 HNSW 图。

本任务主要修改：

- 节点编码格式；
- 候选状态；
- 底层 HNSW 搜索流程；
- 搜索终止条件；
- 最终 Top-K 输出逻辑。

---

# 3. 核心思想

一个节点的距离不再一次性计算完成，而是分为三个阶段：

```text
Stage 0：Short
Stage 1：Long
Stage 2：Residual
```

候选节点包含当前距离精度状态：

```cpp
enum class DistanceStage : uint8_t {
    Short = 0,
    Long = 1,
    Residual = 2
};
```

处理原则：

1. 新发现的底层邻居，先计算 Short 距离和距离区间。
2. Short 候选可以进入候选队列，但不能直接展开邻居。
3. 当 Short 候选成为当前最值得扩展的节点时，先升级为 Long。
4. Long 距离计算后，重新插入候选队列，不立即展开。
5. 如果 Long 候选仍然是最优候选，或者靠近 Top-K 边界，再升级为 Residual。
6. Residual 距离计算后，再次重新插入候选队列。
7. 只有达到规定精度阶段的节点才能真正展开邻居。
8. 搜索结束前，所有最终 Top-K 节点必须达到 Residual 阶段。
9. 最终直接使用 Residual 距离返回结果。

---

# 4. 三层编码设计

## 4.1 Short Code

Short Code 用于：

- 快速估计距离；
- 计算距离下界；
- 计算距离上界；
- 提前淘汰明显不可能影响搜索的节点。

推荐直接使用 4-bit long code 的最高有效 bit-plane，或者使用旋转后残差向量的符号位。

例如每维 1 bit。对于 128 维向量：

```text
128 bit = 16 bytes
```

Short 阶段必须至少输出：

```cpp
struct DistanceInterval {
    float estimate;
    float lower_bound;
    float upper_bound;
};
```

接口建议：

```cpp
DistanceInterval computeShortDistance(
    const QueryContext& query,
    tableint node_id);
```

如果当前代码只有 lower bound，没有 upper bound，第一版可以先使用：

```text
estimate = short estimated distance
lower_bound = existing lower bound
upper_bound = estimate + calibrated_error
```

其中 `calibrated_error` 可以先使用离线统计得到的高分位误差。后续再替换为严格的理论或节点级误差界。

## 4.2 4-bit Long Code

Long Code 是当前 Extended RaBitQ 的完整 4-bit 编码。

它用于：

- 修正 Short Code 的排序误差；
- 决定底层图搜索的主要扩展顺序；
- 维护 beam 候选；
- 为 Residual 阶段提供基础距离。

接口建议：

```cpp
DistanceInterval computeLongDistance(
    const QueryContext& query,
    tableint node_id);
```

如果当前 long-code 距离没有上下界，可以先返回：

```text
estimate = long distance
lower_bound = long distance - calibrated_long_error
upper_bound = long distance + calibrated_long_error
```

第一版重点是完成搜索状态升级机制。

## 4.3 Residual Correction Code

Long Code 对原始旋转残差向量仍有量化误差。

定义：

```text
z = 旋转后的真实残差向量
z4 = 4-bit long code 的重构向量
e = z - z4
```

对误差向量 `e` 再编码。

第一版优先实现 1-bit residual：

```text
每维保存 residual 的正负号
每个节点保存一个 residual_scale
```

即：

```text
residual_sign[j] ∈ {-1, +1}
residual_hat[j] = residual_scale * residual_sign[j]
```

节点级 residual scale 使用最小二乘解：

```text
residual_scale = dot(e, residual_sign) / D
```

Residual Code 空间：

```text
D bit residual sign + 1 个 float residual_scale
```

对于 128 维约为：

```text
16 bytes sign bits + 4 bytes scale = 20 bytes
```

接口建议：

```cpp
DistanceInterval computeResidualDistance(
    const QueryContext& query,
    tableint node_id,
    float long_distance);
```

Residual 距离不要重新解码完整 float32 向量，应在 long distance 基础上增量修正。

设：

```text
x4 = 4-bit 重构向量
er = residual 重构向量
```

则：

```text
d_final = ||q - x4 - er||²
```

展开为：

```text
d_final = d_long + ||er||² - 2 * <q, er> + 2 * <x4, er>
```

其中：

- `||er||²` 可以离线预计算；
- `<x4, er>` 可以离线预计算；
- 查询阶段主要计算 `<q, er>`；
- residual 是 1-bit 符号码，可复用现有 bitwise/SIMD 内积逻辑。

第一版优先支持当前主要使用的 L2Squared。

---

# 5. 节点存储结构

不要把图、short、long、residual 全部放入一个大结构体。推荐使用 Structure of Arrays。

```cpp
class ProgressiveRaBitQStorage {
public:
    std::vector<NodeHeader> headers_;
    std::vector<uint8_t> short_codes_;
    std::vector<uint8_t> long_codes_;
    std::vector<uint8_t> residual_codes_;
};
```

节点头：

```cpp
struct NodeHeader {
    uint16_t centroid_id;

    float norm_sqr;
    float long_scale;

    float residual_scale;
    float residual_norm_sqr;
    float long_residual_inner_product;

    float short_error;
    float long_error;
    float residual_error;
};
```

如果项目已有 `EncodedHeader`、`ShortCodeFactors` 等结构，优先扩展现有结构，不要复制一套重复元数据。

---

# 6. QueryContext

查询向量保持 float32。

对于每个 centroid，查询需要对应的中心化和旋转结果。

```cpp
struct QueryContext {
    uint16_t centroid_id;
    float query_norm_sqr;
    std::vector<float> rotated_query;
    ShortLookup short_lookup;
    LongLookup long_lookup;
    ResidualLookup residual_lookup;
};
```

如果 `centroid_count > 1`，不要在查询开始时为所有 centroid 生成 QueryContext，使用懒加载：

```cpp
const QueryContext& getOrCreateQueryContext(
    uint16_t centroid_id,
    const float* query);
```

---

# 7. 候选数据结构

```cpp
struct RefinableCandidate {
    tableint id;

    float estimate;
    float lower_bound;
    float upper_bound;

    DistanceStage stage;
    bool expanded;
    uint32_t version;
};
```

由于 `std::priority_queue` 不支持 decrease-key，采用 lazy update。

候选升级后：

```cpp
candidate.version += 1;
push(new_candidate);
```

弹出堆元素时：

```cpp
if (heap_item.version != current_state[node_id].version) {
    continue;
}
```

---

# 8. 搜索队列

## 8.1 Expansion Queue

用于决定下一步处理哪个节点。

优先级：

```text
lower_bound 越小越优先
```

定义最小堆：

```cpp
struct ExpansionItem {
    float lower_bound;
    tableint id;
    uint32_t version;
};
```

## 8.2 Result Queue

用于维护当前 beam 中最有希望的 `efSearch` 个候选，使用最大堆。

不同阶段使用不同排序值：

```text
Short：使用 upper_bound
Long：使用 long estimate
Residual：使用 residual estimate
```

建议接口：

```cpp
float candidateRankingDistance(
    const RefinableCandidate& candidate);
```

---

# 9. 上层 HNSW 搜索

HNSW 的 `level > 0` 节点数量较少。

第一版不要对上层使用 Short，上层直接使用 Long 距离：

```text
upper layers: 4-bit long distance only
```

原因：

- 上层计算量少；
- 上层导航错误会严重影响底层入口；
- 简化第一版实现。

---

# 10. 底层搜索完整流程

## 10.1 初始化

对底层入口节点直接计算 Long 距离：

```text
entry stage = Long
```

插入 expansion queue 和 result queue。

## 10.2 发现新邻居

展开一个节点时，遍历其所有未访问邻居。对每个新邻居只计算 Short：

```cpp
DistanceInterval short_result =
    computeShortDistance(query_ctx, neighbor_id);
```

创建 Short 候选。

## 10.3 Short 剪枝

设当前 beam 阈值为 `tau_beam`。

如果 result queue 已满，且：

```text
candidate.lower_bound > tau_beam + short_margin
```

则跳过。

如果 result queue 未满，不要仅因为距离差就剪枝。

## 10.4 Short 候选入队

通过剪枝的 Short 候选插入：

- expansion queue；
- result queue。

Short 候选此时不能展开邻居。

## 10.5 Short 升级为 Long

每次从 expansion queue 弹出候选。

如果候选阶段是 Short：

1. 计算 Long 距离；
2. 更新 estimate、lower_bound、upper_bound；
3. stage 改为 Long；
4. version 加一；
5. 重新插入 expansion queue；
6. 更新 result queue；
7. 本轮结束，不展开该节点。

```cpp
if (candidate.stage == DistanceStage::Short) {
    DistanceInterval long_result =
        computeLongDistance(query_ctx, candidate.id);

    candidate.estimate = long_result.estimate;
    candidate.lower_bound = long_result.lower_bound;
    candidate.upper_bound = long_result.upper_bound;
    candidate.stage = DistanceStage::Long;
    candidate.version += 1;

    updateCandidate(candidate);
    pushExpansion(candidate);
    updateResultQueue(candidate);
    continue;
}
```

升级后不能立即展开。必须重新入队，让其他候选和它重新比较。

## 10.6 Long 是否升级为 Residual

定义：

```cpp
bool needResidualRefinement(
    const RefinableCandidate& candidate,
    float topk_threshold,
    float beam_threshold,
    size_t current_rank);
```

第一版规则：

1. 候选位于 expansion queue 顶部，且区间较宽；
2. 候选可能进入最终 Top-K；
3. 候选位于当前 beam 的前 `residual_beam` 个位置。

建议：

```text
residual_beam = 2 * k
final_margin = 0
```

## 10.7 Long 升级为 Residual

如果 `needResidualRefinement` 返回 true：

1. 计算 Residual 距离；
2. stage 改为 Residual；
3. 更新 estimate、lower_bound、upper_bound；
4. version 加一；
5. 重新插入 expansion queue；
6. 更新 result queue；
7. 本轮不展开。

## 10.8 何时允许展开节点

第一版建议：

```text
Short：不能展开
Long：通常可以展开
Residual：可以展开
```

如果 Long 候选靠近 Top-K 边界或不确定性高，则先升级 Residual，再展开。

---

# 11. 搜索终止条件

传统 HNSW 终止条件不能直接使用 estimate，必须使用 lower bound。

设 expansion queue 顶部候选的最小下界为：

```text
best_unexpanded_lb
```

设 result queue 当前 beam 最差可靠距离为：

```text
beam_threshold
```

当以下条件成立时，才允许停止图扩展：

```text
best_unexpanded_lb > beam_threshold
```

并且 result queue 中至少有 `efSearch` 个有效候选。

---

# 12. 最终稳定化阶段

图搜索主循环结束后，不读取 float32，执行 residual stabilization。

目标：

```text
所有最终 Top-K 节点必须是 Residual 阶段
```

步骤：

1. 从 result queue 中取出候选；
2. 按当前 estimate 排序；
3. 找到当前 Top-K；
4. 对 Top-K 中所有非 Residual 候选进行 Residual 升级；
5. 重新排序；
6. 检查 Top-K 外部候选是否可能进入 Top-K；
7. 对所有满足下面条件的候选继续 Residual 升级：

```text
candidate.lower_bound <= current_topk_worst_upper_bound
```

8. 重复直到：
   - Top-K 全部为 Residual；
   - 所有 Top-K 外候选的 lower_bound 都大于 Top-K 中最差 upper_bound；
   - 或达到 residual refinement budget。

稳定条件：

```text
min(lower_bound of candidates outside Top-K)
>
max(upper_bound of candidates inside Top-K)
```

---

# 13. Residual Budget

为防止最坏情况下大量 residual 计算，加入预算：

```cpp
size_t max_residual_evaluations;
```

建议默认：

```text
max_residual_evaluations = 4 * k
```

参数测试：

```text
k, 2k, 4k, 8k, unlimited
```

如果达到预算但 Top-K 仍未区间分离，第一版直接返回当前 Residual 距离排序结果，不回退到 float32，并统计：

```text
budget_exhausted_queries
```

---

# 14. 完整伪代码

```cpp
std::vector<ResultItem> searchProgressive(
    const float* query,
    size_t k,
    size_t efSearch) {

    QueryContextCache query_cache(query);

    tableint entry = entrypoint_;

    entry = searchUpperLayersWithLongDistance(
        query_cache,
        entry);

    ExpansionMinHeap expansion_queue;
    ResultMaxHeap result_queue;
    CandidateStateTable states;
    VisitedList visited;

    auto entry_interval = computeLongDistance(
        query_cache.getForNode(entry),
        entry);

    RefinableCandidate entry_candidate;
    entry_candidate.id = entry;
    entry_candidate.estimate = entry_interval.estimate;
    entry_candidate.lower_bound = entry_interval.lower_bound;
    entry_candidate.upper_bound = entry_interval.upper_bound;
    entry_candidate.stage = DistanceStage::Long;
    entry_candidate.expanded = false;
    entry_candidate.version = 0;

    states.set(entry, entry_candidate);
    expansion_queue.push(toExpansionItem(entry_candidate));
    result_queue.insertOrUpdate(entry_candidate);
    visited.mark(entry);

    size_t residual_evaluations = 0;

    while (!expansion_queue.empty()) {

        ExpansionItem item = expansion_queue.top();
        expansion_queue.pop();

        RefinableCandidate candidate = states.get(item.id);

        if (item.version != candidate.version) {
            continue;
        }

        float beam_threshold =
            result_queue.currentWorstDistance(efSearch);

        if (result_queue.size() >= efSearch &&
            candidate.lower_bound > beam_threshold) {
            break;
        }

        if (candidate.stage == DistanceStage::Short) {
            auto long_result = computeLongDistance(
                query_cache.getForNode(candidate.id),
                candidate.id);

            candidate.estimate = long_result.estimate;
            candidate.lower_bound = long_result.lower_bound;
            candidate.upper_bound = long_result.upper_bound;
            candidate.stage = DistanceStage::Long;
            candidate.version += 1;

            states.set(candidate.id, candidate);
            expansion_queue.push(toExpansionItem(candidate));
            result_queue.insertOrUpdate(candidate);
            continue;
        }

        float topk_threshold =
            result_queue.currentTopKWorstDistance(k);

        bool need_residual = needResidualRefinement(
            candidate,
            topk_threshold,
            beam_threshold,
            result_queue.rankOf(candidate.id));

        if (candidate.stage == DistanceStage::Long &&
            need_residual &&
            residual_evaluations < max_residual_evaluations) {

            auto residual_result = computeResidualDistance(
                query_cache.getForNode(candidate.id),
                candidate.id,
                candidate.estimate);

            candidate.estimate = residual_result.estimate;
            candidate.lower_bound = residual_result.lower_bound;
            candidate.upper_bound = residual_result.upper_bound;
            candidate.stage = DistanceStage::Residual;
            candidate.version += 1;

            residual_evaluations += 1;

            states.set(candidate.id, candidate);
            expansion_queue.push(toExpansionItem(candidate));
            result_queue.insertOrUpdate(candidate);
            continue;
        }

        if (candidate.expanded) {
            continue;
        }

        candidate.expanded = true;
        candidate.version += 1;
        states.set(candidate.id, candidate);

        for (tableint neighbor :
             getBottomLayerNeighbors(candidate.id)) {

            if (visited.testAndSet(neighbor)) {
                continue;
            }

            auto short_result = computeShortDistance(
                query_cache.getForNode(neighbor),
                neighbor);

            float current_beam_threshold =
                result_queue.currentWorstDistance(efSearch);

            if (result_queue.size() >= efSearch &&
                short_result.lower_bound >
                    current_beam_threshold + short_margin) {
                continue;
            }

            RefinableCandidate next;
            next.id = neighbor;
            next.estimate = short_result.estimate;
            next.lower_bound = short_result.lower_bound;
            next.upper_bound = short_result.upper_bound;
            next.stage = DistanceStage::Short;
            next.expanded = false;
            next.version = 0;

            states.set(neighbor, next);
            expansion_queue.push(toExpansionItem(next));
            result_queue.insertOrUpdate(next);
            result_queue.trimToSize(efSearch);
        }
    }

    stabilizeFinalTopK(
        query_cache,
        states,
        result_queue,
        k,
        max_residual_evaluations,
        residual_evaluations);

    return result_queue.extractTopKResidualOnly(k);
}
```

---

# 15. 新增或修改的主要接口

新增：

```cpp
DistanceInterval computeShortDistance(
    const QueryContext& query,
    tableint node_id) const;

DistanceInterval computeLongDistance(
    const QueryContext& query,
    tableint node_id) const;

DistanceInterval computeResidualDistance(
    const QueryContext& query,
    tableint node_id,
    float long_distance) const;
```

新增搜索接口：

```cpp
std::vector<std::pair<float, labeltype>>
searchKnnProgressiveRefinement(
    const void* query_data,
    size_t k,
    size_t efSearch);
```

保留当前接口用于对照：

```text
searchKnn
searchKnnWithRawRerank
searchKnnProgressiveRefinement
```

不要直接删除旧实现。

---

# 16. 建议修改的文件

优先检查：

```text
hnswlib/hnswalg.h
hnswlib/rabitq_hnsw.h
hnswlib/space_rabitq.h
hnswlib/rabitq_quantizer.h
```

职责划分：

## `rabitq_quantizer.h`

负责：

- long code 编码；
- short code 提取；
- residual code 编码；
- residual scale 计算；
- 编码元数据生成。

## `space_rabitq.h`

负责：

- QueryContext；
- short distance；
- long distance；
- residual distance；
- 距离区间；
- batch distance API。

## `rabitq_hnsw.h`

负责：

- progressive candidate；
- progressive search；
- result queue；
- expansion queue；
- lazy refinement；
- final stabilization。

## `hnswalg.h`

只做必要的 HNSW 公共结构扩展。

如果 progressive search 是 RaBitQ 专用逻辑，尽量不要把大量专用代码直接塞入通用 `hnswalg.h`。

---

# 17. 批量化要求

当前项目已有批量短码下界和 survivor filtering 时，应继续利用。

邻居处理推荐：

```text
收集一批未访问邻居
    ↓
批量计算 Short estimate / lower bound
    ↓
批量筛选 survivors
    ↓
只为 survivors 创建候选状态
```

建议使用查询级 scratch buffer，避免逐节点动态分配。

Long 和 Residual 第一版可先逐候选计算，后续再加入批量升级。

---

# 18. 参数配置

```cpp
struct ProgressiveSearchConfig {
    size_t efSearch;
    float short_margin;
    size_t residual_beam;
    float residual_uncertainty_threshold;
    float final_margin;
    size_t max_residual_evaluations;
    bool require_residual_before_expand;
    bool enable_interval_stabilization;
};
```

建议默认值：

```text
short_margin = 0
residual_beam = 2 * k
final_margin = 0
max_residual_evaluations = 4 * k
require_residual_before_expand = false
enable_interval_stabilization = true
```

---

# 19. 日志和性能统计

必须增加：

```cpp
struct ProgressiveSearchStats {
    size_t visited_nodes;

    size_t short_distance_evaluations;
    size_t long_distance_evaluations;
    size_t residual_distance_evaluations;

    size_t short_pruned_nodes;

    size_t short_to_long_upgrades;
    size_t long_to_residual_upgrades;

    size_t long_reinsertions;
    size_t residual_reinsertions;

    size_t expanded_long_nodes;
    size_t expanded_residual_nodes;

    size_t ranking_changes_after_long;
    size_t ranking_changes_after_residual;

    size_t stabilization_rounds;
    size_t stabilization_residual_evaluations;

    bool residual_budget_exhausted;
};
```

核心指标：

```text
refinement_rate_long = long_distance_evaluations / short_distance_evaluations
refinement_rate_residual = residual_distance_evaluations / long_distance_evaluations
short_prune_ratio = short_pruned_nodes / short_distance_evaluations
```

---

# 20. 测试计划

## 20.1 编码测试

验证：

```text
4-bit + residual reconstruction error
<
4-bit-only reconstruction error
```

## 20.2 距离准确率测试

比较：

```text
float32 exact distance
short estimate
long estimate
residual estimate
```

输出：

```text
mean absolute error
mean relative error
P50 / P95 / P99 error
pairwise ranking accuracy
```

## 20.3 搜索正确性测试

比较：

```text
Float HNSW
Long-only
Short-filter
Progressive Short→Long
Progressive Short→Long→Residual
Float rerank
```

输出 Recall@1 和 Recall@10。

## 20.4 搜索性能测试

扫描：

```text
efSearch
residual_beam
max_residual_evaluations
```

输出完整 Recall-QPS 曲线，不要只比较固定 ef。

---

# 21. 必须保留的实验基线

```text
Baseline A: float32 HNSW
Baseline B: 4-bit Long-only HNSW
Baseline C: Short lower-bound filtering，通过后立即计算 Long
Baseline D: 当前 Top-L float32 rerank
Method 1: Lazy Short→Long
Method 2: Lazy Short→Long→Residual
Method 3: Lazy Short→Long→Residual + interval stabilization
```

---

# 22. 实现顺序

## 第一步

实现 residual 编码和 residual 距离。

暂时保持原搜索逻辑：

```text
Long search → Top-L → Residual refinement
```

验证 residual 能否代替 float32 rerank。

## 第二步

实现 Candidate 的 Short 和 Long 状态：

```text
Short 入队 → 成为队首 → 升级 Long → 重新入队
```

先不加 Residual 搜索内升级。

## 第三步

加入 Long → Residual，仅对当前 Top-K 和扩展队首且不确定性较高的节点精化。

## 第四步

实现 final stabilization，删除固定 `rerank_L`，改成只精化与 Top-K 边界区间重叠的候选。

## 第五步

优化：

- 批量 Short；
- SIMD residual；
- SoA 存储；
- query scratch buffer；
- 减少动态分配。

---

# 23. 第一版完成标准

1. 不读取原始 float32 向量。
2. 搜索结果只使用 Short、Long、Residual code。
3. Short 候选不能直接展开。
4. Short 升级 Long 后必须重新入队。
5. Long 升级 Residual 后必须重新入队。
6. 最终 Top-K 全部达到 Residual 状态。
7. 保留原有搜索和 float32 rerank 作为基线。
8. 输出完整搜索统计。
9. 所有新增代码有单元测试。
10. 在小数据集上无越界、无重复展开、无死循环。

---

# 24. 非目标

第一版暂时不要做：

```text
DiskANN
SSD 图存储
学习式 refinement policy
GPU 实现
多机检索
在线构图
量化图构建
复杂 residual PQ
```

第一版只解决：

> 在内存 HNSW 中，使用 Short、4-bit Long 和 1-bit Residual 三阶段距离精化，取消原始 float32 rerank。

---

# 25. 最终系统流程

```text
离线阶段
────────────────────────

float32 数据
    ↓
逐个插入节点并使用 float32 距离完成：
    图搜索、候选选择、邻居剪枝和连边决策
    ↓
在同一构图过程中为节点生成：
    4-bit Long Code、Short Code 和 1-bit Residual Code
    ↓
当前 HNSW 图直接写入最终索引
    ↓
最终索引只保存：
    图结构、三层编码和距离恢复辅助参数
    ↓
不复制第二张图
    ↓
不把数据库 float32 向量写入最终索引


在线查询
────────────────────────

float32 query
    ↓
上层 HNSW 使用 Long 距离导航
    ↓
进入底层
    ↓
邻居只计算 Short 距离和区间
    ↓
明显无效候选直接剪枝
    ↓
Short 候选进入 Expansion Queue
    ↓
Short 候选成为队首
    ↓
升级 Long
    ↓
使用 Long 距离重新入队
    ↓
关键 Long 候选升级 Residual
    ↓
使用 Residual 距离重新入队
    ↓
Long 或 Residual 节点展开邻居
    ↓
搜索主循环结束
    ↓
只精化与 Top-K 边界重叠的候选
    ↓
所有 Top-K 节点达到 Residual
    ↓
直接返回
```

---

# 26. Codex 执行要求

在开始修改代码前：

1. 阅读现有 RaBitQ 编码结构。
2. 找到当前 Short distance、lower bound、Long distance 的实现。
3. 找到当前 HNSW 底层搜索主循环。
4. 找到当前 float32 rerank 入口。
5. 先输出代码结构分析和修改计划。
6. 不要直接删除旧代码。
7. 每完成一个阶段立即编译和运行现有测试。
8. 新增最小单元测试。
9. 不要同时重构无关代码。
10. 保持旧索引格式兼容，或者明确增加索引版本号。

实现完成后输出：

```text
修改文件列表
新增数据结构
新增接口
搜索流程变化
索引格式变化
测试结果
当前限制
后续优化建议
```
