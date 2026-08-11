# HNSW + Extended RaBitQ

本项目使用Extended RaBitQ 4-bit主码构建和搜索HNSW，并对候选进行概率剪枝和可选的4-bit residual精排。

## 方法

固定实验配置：

```text
DBpedia OpenAI-1536，975000 base，5000 query
K=1，Sequential 4-bit
M=16，efConstruction=200
Float32 query -> 4-bit code 非对称构图
paper active pruning，epsilon0=1.9
batch prefetch，prefetch_distance=2
Recall@10，rerank上限R=100
```

构图和搜索只保存4-bit主码；原始Float32向量只在构图时通过mmap读取。Residual不参与构图，只用于最终精排。

### A/B/C/D

| 实验 | 构图距离 | 精排方式 |
|---|---|---|
| A | Float32 query对4-bit主码（非对称） | 不使用residual，直接返回主4-bit结果 |
| B | Float32 query对4-bit主码（非对称） | 1536维共用一个4-bit residual尺度 |
| C | Float32 query对4-bit主码（非对称） | residual每16维一块，每块使用独立4-bit尺度 |
| D | 4-bit主码对4-bit主码（对称） | 与C相同的block16 4-bit residual精排 |

A、B、C共享相同的非对称primary4图；D使用独立的对称primary4图。D的构图内积估计为：

```text
estimated_ip = <o_bar,q_bar> / (<o_bar,o> * <q_bar,q>)
```

估计值截断到`[-1,1]`后转换为平方L2距离。Residual不参与D的构图。

## 数据准备

DBpedia官方实验包地址：

```text
https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz
```

压缩包约5.10GB，包含`vectors.npy`和`tests.jsonl`。从空目录开始执行以下步骤。

### 1. 进入仓库并创建文件夹

```bash
cd /home/kai3/coco/hnsw_rabitq
mkdir -p downloads
mkdir -p dbpedia_1M
mkdir -p logs/dbpedia
```

### 2. 下载DBpedia

```bash
wget -c \
  https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz \
  -O downloads/dbpedia_openai_1M.tgz
```

`-c`支持中断后继续下载。下载完成后校验MD5：

```bash
echo "dc5fecd77592b669643a5e1ea0541887  downloads/dbpedia_openai_1M.tgz" \
  | md5sum -c -
```

正确输出：

```text
downloads/dbpedia_openai_1M.tgz: OK
```

### 3. 解压并移动原始文件

```bash
tar -xzf downloads/dbpedia_openai_1M.tgz -C downloads

mv downloads/dbpedia_openai_1M/vectors.npy dbpedia_1M/
mv downloads/dbpedia_openai_1M/tests.jsonl dbpedia_1M/
```

检查原始文件：

```bash
ls -lh dbpedia_1M/vectors.npy dbpedia_1M/tests.jsonl
```

### 4. 运行格式转换脚本

转换脚本为：[dbpedia_1M/convert_dbpedia.py](dbpedia_1M/convert_dbpedia.py)。

```bash
python3 -m pip install numpy

python3 dbpedia_1M/convert_dbpedia.py \
  --data-dir dbpedia_1M \
  --batch-size 10000
```

脚本将生成HNSW实验需要的三个文件：

```text
dbpedia_1M/dbpedia_openai1536_base.fvecs
dbpedia_1M/dbpedia_openai1536_query.fvecs
dbpedia_1M/dbpedia_openai1536_groundtruth.ivecs
```

对应数据规模：

```text
base:    975000 × 1536
query:     5000 × 1536
GT:        5000 × 10
```

### 5. 检查转换结果

```bash
ls -lh \
  dbpedia_1M/dbpedia_openai1536_base.fvecs \
  dbpedia_1M/dbpedia_openai1536_query.fvecs \
  dbpedia_1M/dbpedia_openai1536_groundtruth.ivecs
```

默认数据目录为：

```text
/home/kai3/coco/data/dbpedia_openai1536
```

若将转换结果放在其他目录，运行实验前设置：

```bash
export RABITQ_BASE_PATH=/path/to/dbpedia_openai1536_base.fvecs
export RABITQ_QUERY_PATH=/path/to/dbpedia_openai1536_query.fvecs
export RABITQ_GT_PATH=/path/to/dbpedia_openai1536_groundtruth.ivecs
```

## 编译

```bash
cd /home/kai3/coco/hnsw_rabitq
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target main rabitq_hnsw_smoke_test distance_kernel_benchmark -j 64
ctest --test-dir build -R '^rabitq_hnsw_smoke$' --output-on-failure
```

## 运行

为公平比较构图时间，四组实验都设置`RABITQ_FORCE_REBUILD=1`。该开关会忽略已有最终索引、共享图和质心缓存，确保日志记录的是本次真实构建时间。四组命令应在同一台机器、相同线程和系统负载下依次运行。

### 当前D实验：M=32，efConstruction=400

D使用`4-bit primary -> 4-bit primary`对称距离构图，block16 residual4只用于最终rerank。下面命令对应当前DBpedia实验日志：

```text
logs/dbpedia/dbpedia_D_ef_400_M_32.log
```

运行命令：

```bash
cd /home/kai3/coco/hnsw_rabitq

mkdir -p logs/dbpedia

RABITQ_FORCE_REBUILD=1 \
RABITQ_EF_CONSTRUCTION=400 \
RABITQ_M=32 \
RABITQ_BUILD_DISTANCE=symmetric4 \
RABITQ_ABC_ABLATION=D \
RABITQ_PAPER_PRUNE_COMPARE=active \
RABITQ_PAPER_EPSILON0=1.9 \
./build/main > logs/dbpedia/dbpedia_D_ef_400_M_32.log 2>&1
```

如果希望构图阶段也持续输出进度，在运行命令中额外加入：

```bash
RABITQ_BUILD_REPORT_EVERY=10000
```

没有这项时默认是`0`，`symmetric graph addpoint loop`中间不会打印过程；看日志里的`reuse_graph=0`可确认没有复用共享图。

如果需要严格复现`dbpedia_D_ef_400_M_32.log`里的64线程运行环境，在命令前额外加上`OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores`即可。`RABITQ_DATASET`、`RABITQ_INDEX_DIR`和三个DBpedia路径可以省略，因为当前程序默认使用`/home/kai3/coco/data/dbpedia_openai1536`和`build/indexes/dbpedia_ablation`。



### A

```bash
RABITQ_FORCE_REBUILD=1 \
RABITQ_BUILD_DISTANCE=asymmetric4 \
RABITQ_ABC_ABLATION=A \
RABITQ_PAPER_PRUNE_COMPARE=active \
RABITQ_PAPER_EPSILON0=1.9 \
./build/main > logs/dbpedia/dbpedia_A_ef_200_M_16.log 2>&1
```

### B

```bash
RABITQ_FORCE_REBUILD=1 \
RABITQ_BUILD_DISTANCE=asymmetric4 \
RABITQ_ABC_ABLATION=B \
RABITQ_PAPER_PRUNE_COMPARE=active \
RABITQ_PAPER_EPSILON0=1.9 \
./build/main > logs/dbpedia/dbpedia_B_ef_200_M_16.log 2>&1
```

### C

```bash
RABITQ_FORCE_REBUILD=1 \
RABITQ_BUILD_DISTANCE=asymmetric4 \
RABITQ_ABC_ABLATION=C \
RABITQ_PAPER_PRUNE_COMPARE=active \
RABITQ_PAPER_EPSILON0=1.9 \
./build/main > logs/dbpedia/dbpedia_C_ef_200_M_16.log 2>&1
```

### D

```bash
RABITQ_FORCE_REBUILD=1 \
RABITQ_BUILD_DISTANCE=symmetric4 \
RABITQ_ABC_ABLATION=D \
RABITQ_PAPER_PRUNE_COMPARE=active \
RABITQ_PAPER_EPSILON0=1.9 \
./build/main > logs/dbpedia/dbpedia_D_ef_200_M_16.log 2>&1
```

四组实验扫描搜索`ef=10–30, 40–100, 140–460`。文件名中的`ef_200`表示`efConstruction=200`。

构图时间比较使用日志中的：

```text
build_stage=asymmetric4_graph_build us=...
build_stage=symmetric4_graph_build us=...
Graph construction time: ...
Build time: ...
```

其中`Graph construction time`用于比较纯HNSW构图；`Build time`还包含质心训练、主码编码、residual sidecar生成和索引保存。

## 日志检查

日志包含：

```text
Graph construction time
Build time
Recall@10
hnsw_search_us_per_query
residual_rerank_us_per_query
total_us_per_query
QPS
P95
visited_nodes
distance_computations
paper_prune_ratio
paper_saved_ratio
paper_false_prune
Index storage size
```

验收要求：

- A、B、C的`graph_fingerprint`相同，D使用独立图；
- B的`residual_block_size=2048`，C为`16`；
- 非对称构图的`encoded_distance_calls=0`；
- D构图的`asymmetric_distance_calls=0`且`encoded_distance_calls>0`；
- 不修改M、efConstruction、停止条件或rerank上限。

> 当前B的共享图residual映射问题修复前，旧B日志无效，必须重新生成B索引和日志。
