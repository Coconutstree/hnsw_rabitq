# DBpedia HNSW vs RaBitQ Case C 对比

## 对比方式

- 构图时间、总构建时间、有效查询存储占用都是单值指标，使用柱状图对比。
- 查询性能以 Recall@1 vs QPS 为主图，因为它能在相近准确率下比较吞吐；图中只展示 recall >= 0.9 的点，横坐标固定为 0.9-1.0。
- efSearch 曲线作为辅助诊断图，用来观察 search effort 增加时 recall 和 QPS 的变化；QPS 子图固定为 0-3000，超过该范围的点不画。
- QPS 优先读取日志中的 `qps=`；原始 HNSW 日志没有该字段时，按 `1e6 / total_us_per_query` 换算。
- 日志中没有运行时 RSS/峰值内存字段，因此这里的“内存占用”采用 `Index storage size` 表示有效查询存储占用。

## 摘要

| 方法 | 构图时间 s | 总构建 s | 存储占用 MB | 最高 recall | 最高 recall ef | 最高 recall QPS |
|---|---:|---:|---:|---:|---:|---:|
| hnsw | 416.680 | 427.211 | 6356.12 | 0.99770 | 460 | 109.00 |
| rabitq_hnsw | 531.043 | 557.319 | 2579.28 | 0.99546 | 460 | 162.42 |

## 相对结果

- `rabitq_hnsw` 的构图时间是 `hnsw` 的 `1.274x`。
- `rabitq_hnsw` 的总构建时间是 `hnsw` 的 `1.305x`。
- `rabitq_hnsw` 的有效查询存储占用是 `hnsw` 的 `0.406x`，降低 `59.4%`。

## 生成文件

- `summary.csv`: 每个方法一行的总体指标。
- `eval_points.csv`: 解析出的全部 ef/recall/latency/QPS 明细。
- `common_ef_comparison.csv`: 相同 ef 下的 recall 差值和 QPS 比值。
- `build_storage_bars.png/pdf`: 构图时间、总构建时间、存储占用柱状图。
- `recall_vs_qps.png/pdf`: 主要的准确率-吞吐对比图。
- `ef_curves.png/pdf`: efSearch 辅助诊断曲线。
