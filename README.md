# Vamana/DiskANN + Extended RaBitQ4

当前主实验入口是 `./build/main`，运行的是 DiskANN-style Vamana 图加 ExRaBitQ 4-bit payload：

```text
Float32 base vectors
-> train ExRaBitQ centroid
-> encode all base vectors to ExRaBitQ4
-> build Vamana graph with ExRaBitQ4 <-> ExRaBitQ4 symmetric distance
-> search with Float32 query -> ExRaBitQ4 asymmetric distance
-> optional paper-prune during traversal
-> optional 4-bit residual rerank
```

注意：当前 Vamana 构图不是 Float32 构图。构图搜索、RobustPrune、backedge overflow prune 和 refine pass 都使用现有 ExRaBitQ4 symmetric distance kernel。

## 当前配置

DBpedia 复现实验使用：

```text
dataset=dbpedia_openai1536
base_count=990000
query_count=10000
dimension=1536
GT width=100
Recall@10

graph=Vamana
payload=ExRaBitQ4
build_distance=ExRaBitQ4_symmetric
query_distance=Float32_to_ExRaBitQ4

R=32
L_build=400
alpha=1.2
beam_width=1

quantizer=4-bit ExRaBitQ
centroid_count=1
random_seed=100
code_layout=sequential

query_mode=residual4
residual_bits=4
residual_block_size=16
residual_scale_mode=mse
residual_scale_storage=fp16
rerank_candidates=100

paper_prune_active=1
paper_epsilon0=1.9
```

默认索引目录：

```text
build/vamana_exrabitq4
```

默认数据目录：

```text
/home/kai3/coco/data/dbpedia_openai1536
```

当前 `./build/main` 会直接进入 Vamana 路径。旧 HNSW A/B/C/D 代码只作为历史对照保留，不是当前 runtime path。

## 数据准备

DBpedia 官方实验包：

```text
https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz
```

### 下载和解压

```bash
cd /home/kai3/coco/hnsw_rabitq
mkdir -p downloads dbpedia_1M logs/dbpedia/vamana

wget -c \
  https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz \
  -O downloads/dbpedia_openai_1M.tgz

echo "dc5fecd77592b669643a5e1ea0541887  downloads/dbpedia_openai_1M.tgz" \
  | md5sum -c -

tar -xzf downloads/dbpedia_openai_1M.tgz -C downloads
mv downloads/dbpedia_openai_1M/vectors.npy dbpedia_1M/
mv downloads/dbpedia_openai_1M/tests.jsonl dbpedia_1M/
```

### 转换为 fvecs/ivecs

```bash
cd /home/kai3/coco/hnsw_rabitq
python3 -m pip install numpy

python3 dbpedia_1M/convert_dbpedia.py \
  --data-dir dbpedia_1M \
  --batch-size 10000
```

转换后会生成：

```text
dbpedia_1M/dbpedia_openai1536_base.fvecs
dbpedia_1M/dbpedia_openai1536_query.fvecs
dbpedia_1M/dbpedia_openai1536_groundtruth.ivecs
```

如果要使用默认路径，放到：

```bash
mkdir -p /home/kai3/coco/data/dbpedia_openai1536
cp dbpedia_1M/dbpedia_openai1536_base.fvecs /home/kai3/coco/data/dbpedia_openai1536/
cp dbpedia_1M/dbpedia_openai1536_query.fvecs /home/kai3/coco/data/dbpedia_openai1536/
cp dbpedia_1M/dbpedia_openai1536_groundtruth.ivecs /home/kai3/coco/data/dbpedia_openai1536/
```

也可以不复制，运行时显式指定：

```bash
export RABITQ_BASE_PATH=/path/to/dbpedia_openai1536_base.fvecs
export RABITQ_QUERY_PATH=/path/to/dbpedia_openai1536_query.fvecs
export RABITQ_GT_PATH=/path/to/dbpedia_openai1536_groundtruth.ivecs
```

## 编译验证

```bash
cd /home/kai3/coco/hnsw_rabitq
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target main rabitq_hnsw_smoke_test distance_kernel_benchmark -j 64
ctest --test-dir build -R '^rabitq_hnsw_smoke$' --output-on-failure
```

## 复现当前 Vamana baseline

这组对应当前主要对比日志：

```text
logs/dbpedia/vamana/dbpedia_vamana_exrabitq4_Lbuild400_refine2_D_aligned.log
```

命令：

```bash
cd /home/kai3/coco/hnsw_rabitq
mkdir -p logs/dbpedia/vamana

OMP_NUM_THREADS=64 OMP_DYNAMIC=false \
RABITQ_FORCE_REBUILD=1 \
RABITQ_QUERY_MODE=residual4 \
RABITQ_RERANK_CANDIDATES=100 \
RABITQ_RECALL_TARGET=0.995 \
RABITQ_VAMANA_REFINE_PASSES=2 \
RABITQ_VAMANA_PAPER_PRUNE=1 \
RABITQ_VAMANA_PAPER_EPSILON0=1.9 \
./build/main 2>&1 | tee logs/dbpedia/vamana/dbpedia_vamana_exrabitq4_Lbuild400_refine2_D_aligned.log
```

当前这组日志的关键结果：

```text
build_time_seconds=318.615 seconds
Index storage size: 2444.21 MB (index=1169.07 MB, auxiliary=0.014473 MB, residual=1275.12 MB, total_bytes=2444206389)
L_search=580 recall@10=0.99501 QPS=121.936 total_us_per_query_us=8201.03
L_search_sweep_stop reason=recall_target_reached recall_target=0.995 L_search=580 recall@10=0.99501
```

## 复现 refine_pass=1

```bash
cd /home/kai3/coco/hnsw_rabitq
mkdir -p logs/dbpedia/vamana

OMP_NUM_THREADS=64 OMP_DYNAMIC=false \
RABITQ_FORCE_REBUILD=1 \
RABITQ_QUERY_MODE=residual4 \
RABITQ_RERANK_CANDIDATES=100 \
RABITQ_RECALL_TARGET=0.995 \
RABITQ_VAMANA_REFINE_PASSES=1 \
RABITQ_VAMANA_PAPER_PRUNE=1 \
RABITQ_VAMANA_PAPER_EPSILON0=1.9 \
./build/main 2>&1 | tee logs/dbpedia/vamana/dbpedia_vamana_Lbuild_400_M_32_refine_1.log
```

## 复现 refine_pass=0

```bash
cd /home/kai3/coco/hnsw_rabitq
mkdir -p logs/dbpedia/vamana

OMP_NUM_THREADS=64 OMP_DYNAMIC=false \
RABITQ_FORCE_REBUILD=1 \
RABITQ_QUERY_MODE=residual4 \
RABITQ_RERANK_CANDIDATES=100 \
RABITQ_RECALL_TARGET=0.995 \
RABITQ_VAMANA_REFINE_PASSES=0 \
RABITQ_VAMANA_PAPER_PRUNE=1 \
RABITQ_VAMANA_PAPER_EPSILON0=1.9 \
./build/main 2>&1 | tee logs/dbpedia/vamana/dbpedia_vamana_Lbuild_400_M_32_refine_0.log
```

## 常用运行开关

```text
RABITQ_FORCE_REBUILD=1
  忽略已有 Vamana index/quantizer/residual sidecar，重新构建。

RABITQ_QUERY_MODE=primary4_only|residual4|residual8
  primary4_only 只用 ExRaBitQ4 traversal 结果。
  residual4 使用 block residual4 对候选 rerank。
  residual8 使用 residual8 rerank。

RABITQ_RERANK_CANDIDATES=100
  residual rerank 的候选上限。

RABITQ_RECALL_TARGET=0.995
  L_search 自适应扫描的停止 recall。

RABITQ_LSEARCH_MAX=<N>
  限制最大 L_search。

RABITQ_VAMANA_REFINE_PASSES=0|1|2|...
  构图后的 Vamana full-graph refine pass 数。

RABITQ_VAMANA_PAPER_PRUNE=0|1
  是否在 Vamana traversal 中启用 paper-prune。

RABITQ_VAMANA_PAPER_EPSILON0=1.9
  paper-prune epsilon0。

RABITQ_VAMANA_BUILD_BATCH=4096
  segmented parallel graph build 的 batch size。

RABITQ_BUILD_REPORT_EVERY=65536
  构图/refine 进度打印间隔。设为 0 可关闭中间进度。

RABITQ_BASE_LIMIT=<N>
  只使用前 N 个 base vectors，适合 smoke 或局部性能调试。

RABITQ_INDEX_DIR=<path>
  修改索引输出目录。
```

当前代码没有启用 BFS reorder / multi-entry runtime 开关；如果看到旧实验日志里的 `layout=bfs` 或 `entry_count=8`，那是已撤销分支产生的实验结果，不代表当前 `./build/main` 可用参数。

## 搜索扫描

当前 Vamana 搜索扫描为：

```text
L_search=10..30
L_search=40,50,60,70,80,90,100
L_search=140 起每次 +40，直到 recall@10 >= RABITQ_RECALL_TARGET
```

默认目标是：

```text
RABITQ_RECALL_TARGET=0.995
```

## 日志字段

重点看这些字段：

```text
graph=Vamana
payload=ExRaBitQ4
build_distance=ExRaBitQ4_symmetric
query_distance=Float32_to_ExRaBitQ4
R=32
L_build=400
alpha=1.2
beam_width=1
build_stage=vamana_graph_build seconds=...
build_stage=vamana_graph_refine seconds=...
build_time_seconds=...
Index storage size: ... MB (...)
L_search=...
recall@10=...
QPS=...
total_us_per_query_us=...
traversal_us_per_query_us=...
residual_rerank_us_per_query_us=...
avg_visited_nodes=...
avg_distance_computations=...
avg_hops=...
paper_prune_ratio=...
paper_saved_ratio=...
max_degree=...
avg_degree=...
```

`total_us_per_query_us` 是单 query 的端到端平均耗时，单位是微秒，包括 Vamana traversal 和 residual rerank。`QPS` 按同一个端到端耗时换算：

```text
QPS = 1e6 / total_us_per_query_us
```

## 和旧 HNSW D 对比

旧 HNSW D 对比日志通常是：

```text
logs/dbpedia/dbpedia_D_ef_400_M_32.log
```

公平对比时保持：

```text
线程数相同
DBpedia 数据相同
R/M 都为 32
L_build/efConstruction 都为 400
query_mode=residual4
residual_block_size=16
rerank_candidates=100
paper-prune active
recall@10
```

不要把 Vamana 的 `L_search` 直接当作 HNSW 的 `efSearch` 完全等价解释。它们都是候选池宽度，但图结构、入口、剪枝语义和访问路径不同。
