# 直接将现有 4-bit ExRaBitQ-HNSW 改为 4-bit ExRaBitQ-Vamana/DiskANN

## 最终目标
直接修改当前工程：

`/home/kai3/coco/hnsw_rabitq`

不保留 HNSW runtime path，改成：

`训练 ExRaBitQ -> 所有 base vectors 编码为 4-bit -> 使用 4-bit ExRaBitQ 对称距离直接构建 Vamana 图 -> 保存 Vamana graph + 4-bit payload -> Float32 query -> ExRaBitQ 非对称距离 -> DiskANN-style beam search -> TopK`

注意：**不是 Float32 构图。构图从头到尾直接使用 4-bit ExRaBitQ 对称量化距离。**

## 最重要的约束
现有工程中“4-bit 对称量化构图”的 ExRaBitQ 方法必须保持不变，包括：
- quantizer training
- centroid / rotation / transform
- 4-bit code layout
- metadata
- 4-bit data-to-data symmetric distance
- query preprocessing
- Float32 query-to-4bit asymmetric distance
- single/batch distance kernel

Codex 首先必须找到当前 HNSW + ExRaBitQ 对称构图实际使用的函数，不重新设计 ExRaBitQ 数学公式。

本次修改只改变：

`HNSW graph construction/search`
-> `Vamana graph construction/search`

量化方法不变。

## DiskANN 参考代码
参考文件均在：

`/home/kai3/coco/hnsw_rabitq/diskann`

重点：
- `index.rs`：insert、search_internal、robust_prune、occlude_list、backedge
- `knn_search.rs`：L_search、beam_width
- `glue.rs`：SearchAccessor / InsertStrategy / PruneAccessor 的职责
- `prune.rs`：RobustPrune scratch/state
- `backedge.rs`：反向边处理
- `vectors.rs`：4-bit Data↔Data 对称 compensated distance，以及 FullQuery↔Data 查询距离
- `quantizer.rs`：4-bit 数据压缩流程
- `iface.rs`：SameAsData distance computer 与 FullPrecision query computer

Rust 文件只作为算法参考，不直接编译进当前 C++ 工程。

## DiskANN 文档中必须对应的两条距离路径

### 构图距离：4-bit ↔ 4-bit
参考 `vectors.rs`：

`CompensatedSquaredL2<DataRef<4>, DataRef<4>>`

其语义是：

`quantized database vector`
↔
`quantized database vector`

这条路径用于：
- 新节点构图搜索
- candidate ranking
- RobustPrune
- candidate-candidate occlusion
- backedge overflow 后再次 prune

因此 Vamana 构图不能偷偷使用 Float32。

### 查询距离：Float32 ↔ 4-bit
参考 `vectors.rs`：

`CompensatedSquaredL2<FullQueryRef, DataRef<4>>`

其语义是：

`Float32 query`
↔
`quantized database vector`

这条路径用于在线 ANN 查询。

也就是说最终必须明确区分：

`build_distance = exrabitq4_symmetric`

`query_distance = float32_query_to_exrabitq4`

## Step 1：先扫描当前代码
先不要重写。

找到当前工程中已经存在的：
- 4-bit ExRaBitQ trainer
- base vector encode
- 4-bit code storage
- 4bit↔4bit symmetric distance
- Float32 query preprocessing
- Float32 query↔4bit distance
- batch distance
- 当前 HNSW 对称量化构图入口
- 当前 HNSW query 入口
- dataset loader
- GT / Recall / QPS
- save/load
- index size
- build time

输出一个极简映射：

`现有 HNSW graph/search 函数 -> 新 Vamana 函数`
`现有 ExRaBitQ 函数 -> 原样复用`

尤其先确认“当前 4-bit HNSW 对称构图到底调用哪个 distance function”。

不要假设，直接从代码确认。

## Step 2：量化必须在构图前完成
正确顺序：

1. load Float32 base vectors
2. train 当前 ExRaBitQ quantizer
3. encode 所有 base vectors 为 4-bit ExRaBitQ
4. 准备每个 4-bit code 所需 metadata
5. 后续 Vamana 构图只通过 4-bit code 计算距离

除 quantizer training / encode 输入外，Vamana graph construction 不调用 Float32 base-to-base L2。

若当前对称 HNSW 构图已经是上述流程，直接复用其 quantized storage 和 symmetric distance。

## Step 3：将 HNSW 图结构替换为 Vamana
新增或重构：

`vamana_index.h`
`vamana_search.h`
`vamana_prune.h`

最终索引主体不再使用 HNSW hierarchy。

删除/停用：
- level
- maxlevel
- upper layers
- HNSW enterpoint hierarchy
- M/M0 的 HNSW 语义
- efConstruction
- efSearch
- HNSW neighbor-selection heuristic

Vamana 使用：
- `R`
- `L_build`
- `alpha`
- `L_search`
- `beam_width`
- `start_node`

第一版可先用：
`R=32`
`L_build=400`
`alpha=1.2`
`beam_width=1`

参数后续可 sweep。

## Step 4：实现统一 Vamana search controller
实现：

`search_vamana(...)`

搜索控制器不能绑定某个具体 distance。

但本项目只需要两种 evaluator：

### BuildEvaluator
输入：
`node_id / encoded ExRaBitQ4`

计算：
`4bit code ↔ 4bit code symmetric distance`

用于构图。

### QueryEvaluator
固定：
`preprocessed Float32 query`

输入：
`candidate ExRaBitQ4 code`

计算：
`Float32 query ↔ 4bit code asymmetric distance`

用于查询。

除 evaluator 外，下列逻辑完全共享：
- start node
- candidate queue
- visited set
- beam frontier
- adjacency expansion
- termination

## Step 5：Vamana 构图搜索必须使用 4-bit 对称距离
插入节点 `p` 时：

1. `p` 已经编码为 ExRaBitQ 4-bit。
2. 从 Vamana `start_node` 开始图搜索。
3. 当前 candidate `u` 的评分：
   `d_sym4(p_code, u_code)`
4. 展开 `u` 的邻居 `v`。
5. 新邻居评分：
   `d_sym4(p_code, v_code)`
6. 维护构图 candidate pool，窗口=`L_build`。
7. 将 visited/candidate 交给 RobustPrune。

禁止：
`p float32 -> candidate float32`

构图搜索必须与当前 4-bit HNSW 对称构图保持相同量化距离语义。

## Step 6：RobustPrune 也必须使用 4-bit 对称距离
实现：

`robust_prune_4bit(p, candidates, alpha, R)`

所有距离都用同一个现有 symmetric ExRaBitQ distance：

`d(p,candidate) = d_sym4(p_code, candidate_code)`

以及：

`d(candidate_i,candidate_j) = d_sym4(code_i, code_j)`

流程参考 DiskANN：
- candidate pool
- distance ordering
- occlusion factor
- alpha
- pruned neighbors
- optional saturation

重点参考：
`diskann/index.rs::robust_prune`
`robust_prune_list`
`robust_prune_with`
`occlude_list`
`diskann/prune.rs`

不要自己简化成：
`sort -> Top-R`

也不要在 candidate-candidate distance 时回退 Float32。

DiskANN 的 `PruneAccessor::fill()` 本质就是提供一批同类型 database elements 和 element-to-element distance computer；在当前项目中这正对应：
`ExRaBitQ4 code view + symmetric ExRaBitQ distance`。

## Step 7：backedge
新节点 `p` 完成 prune 后：

`graph[p] = selected_neighbors`

然后对每个 `u in selected_neighbors` 添加：

`u -> p`

如果 `graph[u]` 超过 degree/slack 限制：

对：

`u + graph[u] candidates`

重新执行：

`robust_prune_4bit()`

仍然全部使用 4-bit 对称距离。

参考：
`index.rs::insert`
`add_edge_and_prune`
`backedge.rs`

不需要照搬 Rust `BackedgeBuffer`，只保留算法语义。

## Step 8：start node
start node 必须和“直接 4-bit 构图”保持一致。

优先规则：
1. 先检查当前对称量化 HNSW/已有实现怎样确定 entry/center。
2. 如果已有适配 ExRaBitQ 的中心/入口选法，继续复用。
3. 如果没有，再实现一个明确的 start node 方法。

不要为了方便默认使用 Float32 medoid 而改变当前对称量化方法。

如果需要从 centroid 选离中心最近的数据库节点，节点比较优先使用当前 ExRaBitQ 构图设定所允许的距离方式，并把实现写清楚。

## Step 9：在线查询
构图完成后 graph 和 payload 都已经是：

`Vamana graph + ExRaBitQ4`

查询：

1. load Float32 query
2. 对 query 只做一次现有 ExRaBitQ query preprocessing
3. 从 `start_node` 开始 Vamana beam search
4. 对访问到的数据库节点：
   `d_query = asymmetric_distance(preprocessed_float_query, exrabitq4_code)`
5. candidate queue 使用该距离
6. 返回 TopK

禁止：
- query 再量化成 4-bit，除非当前方法原本就是这么做
- decode 4-bit database vector 后算 Float32 L2
- 每访问一个节点重复 query preprocessing

查询方式保持当前 HNSW + ExRaBitQ 的方式不变，只把 graph traversal 改成 Vamana。

## Step 10：beam search
参考：
`knn_search.rs`
`index.rs::search_internal`
`glue.rs::SearchAccessor::expand_beam`

支持：
`beam_width = 1,2,4,8`

每轮：
1. 取最多 beam_width 个最优未展开 candidate
2. 获取这些节点 adjacency
3. 合并邻居
4. visited 去重
5. 对新候选调用 QueryEvaluator
6. 更新 candidate pool

如果已有 ExRaBitQ batch distance：
优先将去重后的多个 neighbor codes 一次交给 batch kernel。

构图阶段第一版可以 `beam_width=1`；若后续需要，可单独支持 build beam。

## Step 11：索引存储
最终 serialized index 保存：

- magic/version
- N
- dim
- R
- L_build
- alpha
- start_node
- Vamana adjacency lists
- ExRaBitQ quantizer metadata
- ExRaBitQ 4-bit codes

正式 index 不保存 HNSW：
- levels
- maxlevel
- upper links
- HNSW entry hierarchy

正式 index size 只统计真实 Vamana+ExRaBitQ 内容。

## Step 12：日志
最终至少输出：

`graph=Vamana`
`payload=ExRaBitQ4`
`build_distance=ExRaBitQ4_symmetric`
`query_distance=Float32_to_ExRaBitQ4`
`R`
`L_build`
`alpha`
`L_search`
`beam_width`
`build_time_ms`
`graph_size_MB`
`payload_size_MB`
`total_index_size_MB`
`recall@10`
`QPS`
`avg_visited_nodes`
`avg_distance_computations`
`avg_hops`
`max_degree`
`avg_degree`

构图时间统一使用 ms。

## Step 13：正确性测试
必须先做距离测试，再测完整图。

### Test A：4-bit symmetric distance
从当前已有对称 HNSW 实现随机抽取大量 `(i,j)`：

比较：
`旧 symmetric build distance(i,j)`
vs
`新 Vamana BuildEvaluator(i,j)`

要求一致，或仅有可解释的浮点误差。

### Test B：query asymmetric distance
比较：
`旧 HNSW query distance(q,i)`
vs
`新 Vamana QueryEvaluator(q,i)`

要求一致。

### Test C：RobustPrune
小数据集检查：
- 无 self-loop
- 无 duplicate
- degree 合法
- backedge overflow 后确实重新 prune

### Test D：完整 Vamana
扫：
`L_search=20,40,60,80,100,150,200,300`

检查 Recall 整体随搜索预算提高。

## Step 14：最终改造原则
当前工程最终直接成为：

`Vamana + ExRaBitQ`

不再要求保留 HNSW runtime path。

但在改造完成前，不要过早删除旧 HNSW 对称构图代码，因为需要它做：
- symmetric distance 对照
- query distance 对照
- 量化流程参考

等 Test A/B 通过后再删除 HNSW graph/search path。

注意：
“暂时留代码用于验证”
不等于
“最终保留 HNSW baseline”。

最终 runtime 只需要 Vamana。

## 禁止事项
- 禁止 Float32 构图
- 禁止 Float32 RobustPrune
- 禁止构图后才量化
- 禁止 Top-R 替代 RobustPrune
- 禁止改变现有 ExRaBitQ 编码数学
- 禁止使用 DiskANN spherical quantizer 代替 ExRaBitQ
- 禁止 SSD / async I/O
- 禁止 PQ
- 禁止 progressive
- 禁止 rerank
- 禁止 dynamic delete/insert 优化
- 禁止把旧 HNSW level0 adjacency 直接当作 Vamana

## Codex 执行顺序
严格按以下顺序执行：

1. 扫描当前仓库。
2. 明确当前 4-bit 对称 HNSW 构图使用的 ExRaBitQ encode 和 symmetric distance 函数。
3. 明确当前 Float32 query -> 4bit distance 函数。
4. 阅读 `diskann/` 下参考实现。
5. 输出简短函数映射。
6. 先写 BuildEvaluator 和 QueryEvaluator，对现有距离做一致性测试。
7. 实现 Vamana search controller。
8. 实现 4-bit RobustPrune。
9. 实现 Vamana insert + backedge。
10. 构建 4-bit Vamana 图。
11. 接入在线 Float32 query search。
12. 编译并做小数据 smoke test。
13. 跑真实数据 Recall/QPS。
14. 最后删除/停用旧 HNSW runtime path。

不要先实现 Float32 Vamana baseline。
不要先实现 Float32 构图。
本任务从第一版开始就是：
`4-bit symmetric ExRaBitQ Vamana build + Float32-to-4bit ExRaBitQ query search`。
