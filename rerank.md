# 目标

在查询阶段实现 rerank，提高最终 recall。

当前 `hnswlib` 内部搜索只使用 RaBitQ 估计距离返回 top-k，没有用原始 float 向量做二次精排。低维数据集，例如 deep1B 的 96 维，在 4-bit 量化后距离排序容易被量化误差扰动，因此不 rerank 时 recall 可能偏低。

# 当前状态

`/home/kai3/coco/hnsw_rabitq/hnswlib` 里没有真正的 rerank。

关键逻辑如下：

```cpp
// hnswlib/rabitq_hnsw.h
std::priority_queue<std::pair<float, labeltype>>
searchKnn(const float *raw_query, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
    return index_.searchKnn(raw_query, k, isIdAllowed);
}
```

```cpp
// hnswlib/space_rabitq.h
float result_distance(const void *prepared_query, const void *data_point) override {
    return query_distance(prepared_query, data_point);
}
```

也就是说最终结果仍然按 RaBitQ 估计距离排序，没有重新读取原始 float 向量计算真实 L2/cosine 距离。

# 推荐方法

推荐先做外部 rerank，不改索引存储格式。

流程：

1. HNSW/RaBitQ 先返回 `candidate_count` 个候选。
2. 根据候选 id 从原始 `.fvecs` 文件中读取对应原始向量。
3. 用原始 query 和原始 base 向量计算真实距离。
4. 按真实距离重新排序。
5. 取前 `k` 个作为最终结果。

这个方案的优点：

- 不增加索引文件大小。
- 不需要在 HNSW index 里保存原始 float 向量。
- 容易和“不 rerank”结果对比。
- `candidate_count` 可调，方便测 recall/latency tradeoff。

缺点：

- 查询时间会增加。
- 如果每次从磁盘随机读 `.fvecs`，rerank 时间可能比较高。
- 如果把原始 base 全量加载到内存，查询更快，但会明显增加内存。

# 当前测试代码已有的外部 rerank

`tests/cpp/sift_1b.cpp` 里的 `test_approx(...)` 已经有一版外部 rerank 逻辑：

```cpp
const size_t candidate_count =
    std::max(k, std::min(rerank_candidates, std::max(k, search_ef)));

vector<pair<float, labeltype>> results =
    appr_alg.searchKnnCloserFirst(massQ + vecdim * i, candidate_count);

if (candidate_count > k) {
    vector<pair<float, labeltype>> reranked;
    const float *query = massQ + vecdim * i;
    for (const auto &entry : results) {
        raw_reader.readVector(static_cast<size_t>(entry.second), raw_candidate.data());
        reranked.emplace_back(raw_l2_float(query, raw_candidate.data(), vecdim), entry.second);
    }
    std::sort(reranked.begin(), reranked.end());
    reranked.resize(k);
    results.swap(reranked);
}
```

所以如果要打开 rerank，核心是设置：

```cpp
const int rerank_candidates = 100;
```

如果要测试不 rerank，设置：

```cpp
const int rerank_candidates = 1;
```

或保证 `candidate_count == k`。

# 参数建议

假设最终评估是 Recall@1：

```cpp
const size_t k = 1;
```

可以按下面顺序测试：

```cpp
rerank_candidates = 1;    // 不 rerank
rerank_candidates = 5;
rerank_candidates = 10;
rerank_candidates = 20;
rerank_candidates = 50;
rerank_candidates = 100;
```

建议同时记录：

- `recall`
- `hnsw_search_us_per_query`
- `redundant_rerank_us_per_query`
- `total_us_per_query`

这样可以看清楚 rerank 带来的 recall 提升和查询时间增加。

# 低维数据的注意点

deep1B 这里是 96 维，而且向量已经归一化。

对单位向量：

```text
L2^2(x, y) = 2 - 2 * cos(x, y)
```

因此 deep1B 使用 raw L2 和 cosine/angular 排序是等价的。recall 低更可能来自 4-bit 量化误差，而不是没有归一化。

低维数据上，4-bit 量化误差对距离排序的扰动会更明显，所以 rerank 通常会比高维数据更有帮助。

# 两种实现路线

## 路线 A：外部文件 rerank

使用当前 `FVecRandomReader` 按候选 id 从 `.fvecs` 文件中读取原始向量。

适合：

- 评测实验。
- 不想增加常驻内存。
- 想快速验证 rerank 对 recall 的影响。

代价：

- 随机读文件可能慢。
- `rerank_candidates` 越大，查询时间越高。

## 路线 B：内存 rerank

启动时把原始 base 向量加载到内存：

```text
base_vectors[label * dim : (label + 1) * dim]
```

查询时直接根据 label 访问内存中的原始向量并计算距离。

适合：

- 追求更低 rerank latency。
- 机器内存足够。

代价：

- 内存增加约为 `base_count * dim * sizeof(float)`。

例如：

```text
deep1B:  1B * 96   * 4 bytes 约 384 GB
dbpedia: 990K * 1536 * 4 bytes 约 6.1 GB
sift10m: 10M * 128  * 4 bytes 约 5.1 GB
```

# 和 hnswlib 内部 rerank 的关系

如果要把 rerank 做进 `hnswlib` 内部，需要解决两个问题：

1. HNSW index 当前只保存 RaBitQ 编码 payload，没有保存原始 float。
2. `searchKnn()` 当前只返回 top-k，无法在内部拿 top-N 后再按原始距离精排，除非新增接口。

可新增接口：

```cpp
searchKnnWithRerank(raw_query, k, candidate_count, raw_vector_reader)
```

或者：

```cpp
searchCandidateIds(raw_query, candidate_count)
```

再由外部调用方自己 rerank。当前代码已经有 `searchCandidateIds(...)`，所以更推荐先走外部 rerank。

# 验证步骤

1. 设置数据集参数：

```cpp
// deep1B
vecdim = 96;
gt_width = 100;

// dbpedia
vecdim = 1536;
gt_width = 100;

// sift10m
vecdim = 128;
gt_width = 1000;
```

2. 设置 rerank 候选数：

```cpp
const int rerank_candidates = 100;
```

3. 重新编译：

```bash
cd /home/kai3/coco/hnsw_rabitq/build
cmake --build . --target main -j
```

4. 运行并保存日志：

```bash
nohup ./main > test_deep1B_ef_40_M_16_测试rerank100效果.log 2>&1 &
```

5. 对比不 rerank 日志：

```text
test_deep1B_ef_40_M_16_测试不rerank效果.log
test_deep1B_ef_40_M_16_测试rerank100效果.log
```

# 预期结果

开启 rerank 后：

- recall 应该上升。
- `redundant_rerank_us_per_query` 会大于 0。
- `total_us_per_query` 会增加。

如果 recall 没有明显上升，需要检查：

- `candidate_count` 是否真的大于 `k`。
- `rerank_candidates` 是否被 `search_ef` 限制住。
- 原始 `.fvecs` 路径是否和建索引使用的 base 文件一致。
- label 是否等于原始 base 文件里的行号。
- deep1B/dbpedia/sift10m 的 `vecdim` 和 `gt_width` 是否正确。
