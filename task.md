# Extended RaBitQ-HNSW 实验命令

## 编译

当前 `build/` 目录已配置为 Unix Makefiles，可以直接：

```bash
cd /home/kai3/coco/hnsw_rabitq
make -C build -j
```

只编译主实验程序：

```bash
make -C build main -j
```

## 统一实验配置

本工程主码固定为 4-bit Extended RaBitQ，只比较 residual refinement 位宽：

```text
RABITQ_RESIDUAL_BITS=1/2/4/8/16
```

推荐统一使用 disk/disk，保证小数据和大数据走同一条路径：

```text
RABITQ_RESIDUAL_STORAGE=disk
RABITQ_PAYLOAD_MODE=disk
```

索引文件默认写到 `build/` 目录：

```text
RABITQ_INDEX_DIR=build
```

从项目根目录运行 `./build/main` 时不需要显式设置。若从 `build/` 目录内运行 `./main`，请设置：

```text
RABITQ_INDEX_DIR=.
```

说明：

```text
RABITQ_RESIDUAL_STORAGE=disk  最终 residual 存到 .residual sidecar，并通过 mmap 查询
RABITQ_PAYLOAD_MODE=disk      构建阶段临时 payload 写磁盘，降低内存压力
```

默认参数：

```text
RABITQ_FAST_RESIDUAL_CANDIDATES=100
RABITQ_FAST_SEARCH_EF_MULTIPLIER=1
RABITQ_RESIDUAL_BLEND=1
```

这些默认值不必显式写在命令里。

## 公平对比原则

不要混用旧日志和新日志直接比较。

推荐统一使用新命名：

```text
test_run{N}_{dataset}_residual{bits}_disk.log
```

同一轮实验应固定：

```text
dataset
M=16
efConstruction=40
random_seed=100
RABITQ_RESIDUAL_STORAGE=disk
RABITQ_PAYLOAD_MODE=disk
search 参数默认值
```

只改变：

```text
RABITQ_RESIDUAL_BITS
```

大数据实验不要同时跑多个 bit，建议排队串行运行，避免 CPU、内存和磁盘 IO 互相干扰。

查看是否仍有实验在跑：

```bash
ps -ef | grep './build/main' | grep -v grep
```

## sift10m

```bash
cd /home/kai3/coco/hnsw_rabitq
mkdir -p logs/sift10m

nohup bash -c '
RABITQ_DATASET=sift10m RABITQ_RESIDUAL_BITS=1  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/sift10m/test_run1_sift10m_residual1_disk.log 2>&1 &&
RABITQ_DATASET=sift10m RABITQ_RESIDUAL_BITS=2  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/sift10m/test_run1_sift10m_residual2_disk.log 2>&1 &&
RABITQ_DATASET=sift10m RABITQ_RESIDUAL_BITS=4  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/sift10m/test_run1_sift10m_residual4_disk.log 2>&1 &&
RABITQ_DATASET=sift10m RABITQ_RESIDUAL_BITS=8  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/sift10m/test_run1_sift10m_residual8_disk.log 2>&1 &&
RABITQ_DATASET=sift10m RABITQ_RESIDUAL_BITS=16 RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/sift10m/test_run1_sift10m_residual16_disk.log 2>&1
' > logs/sift10m/test_run1_sift10m_queue.log 2>&1 &
```

查看队列日志：

```bash
tail -f logs/sift10m/test_run1_sift10m_queue.log
```

查看单个实验：

```bash
tail -f logs/sift10m/test_run1_sift10m_residual8_disk.log
```

## deep1B

```bash
cd /home/kai3/coco/hnsw_rabitq
mkdir -p logs/deep1B

nohup bash -c '
RABITQ_DATASET=deep1B RABITQ_RESIDUAL_BITS=1  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/deep1B/test_run1_deep1B_residual1_disk.log 2>&1 &&
RABITQ_DATASET=deep1B RABITQ_RESIDUAL_BITS=2  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/deep1B/test_run1_deep1B_residual2_disk.log 2>&1 &&
RABITQ_DATASET=deep1B RABITQ_RESIDUAL_BITS=4  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/deep1B/test_run1_deep1B_residual4_disk.log 2>&1 &&
RABITQ_DATASET=deep1B RABITQ_RESIDUAL_BITS=8  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/deep1B/test_run1_deep1B_residual8_disk.log 2>&1 &&
RABITQ_DATASET=deep1B RABITQ_RESIDUAL_BITS=16 RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/deep1B/test_run1_deep1B_residual16_disk.log 2>&1
' > logs/deep1B/test_run1_deep1B_queue.log 2>&1 &
```

## dbpedia_openai1536

程序接受的数据集名为 `dbpedia` 或 `dbpedia-openai1536`。

```bash
cd /home/kai3/coco/hnsw_rabitq
mkdir -p logs/dbpedia_openai1536

nohup bash -c '
RABITQ_DATASET=dbpedia RABITQ_RESIDUAL_BITS=1  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/dbpedia_openai1536/test_run1_dbpedia_openai1536_residual1_disk.log 2>&1 &&
RABITQ_DATASET=dbpedia RABITQ_RESIDUAL_BITS=2  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/dbpedia_openai1536/test_run1_dbpedia_openai1536_residual2_disk.log 2>&1 &&
RABITQ_DATASET=dbpedia RABITQ_RESIDUAL_BITS=4  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/dbpedia_openai1536/test_run1_dbpedia_openai1536_residual4_disk.log 2>&1 &&
RABITQ_DATASET=dbpedia RABITQ_RESIDUAL_BITS=8  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/dbpedia_openai1536/test_run1_dbpedia_openai1536_residual8_disk.log 2>&1 &&
RABITQ_DATASET=dbpedia RABITQ_RESIDUAL_BITS=16 RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/dbpedia_openai1536/test_run1_dbpedia_openai1536_residual16_disk.log 2>&1
' > logs/dbpedia_openai1536/test_run1_dbpedia_openai1536_queue.log 2>&1 &
```

## gist

```bash
cd /home/kai3/coco/hnsw_rabitq
mkdir -p logs/gist

nohup bash -c '
RABITQ_DATASET=gist RABITQ_RESIDUAL_BITS=1  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/gist/test_run1_gist_residual1_disk.log 2>&1 &&
RABITQ_DATASET=gist RABITQ_RESIDUAL_BITS=2  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/gist/test_run1_gist_residual2_disk.log 2>&1 &&
RABITQ_DATASET=gist RABITQ_RESIDUAL_BITS=4  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/gist/test_run1_gist_residual4_disk.log 2>&1 &&
RABITQ_DATASET=gist RABITQ_RESIDUAL_BITS=8  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/gist/test_run1_gist_residual8_disk.log 2>&1 &&
RABITQ_DATASET=gist RABITQ_RESIDUAL_BITS=16 RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/gist/test_run1_gist_residual16_disk.log 2>&1
' > logs/gist/test_run1_gist_queue.log 2>&1 &
```

## glove

```bash
cd /home/kai3/coco/hnsw_rabitq
mkdir -p logs/glove

nohup bash -c '
RABITQ_DATASET=glove RABITQ_RESIDUAL_BITS=1  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/glove/test_run1_glove_residual1_disk.log 2>&1 &&
RABITQ_DATASET=glove RABITQ_RESIDUAL_BITS=2  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/glove/test_run1_glove_residual2_disk.log 2>&1 &&
RABITQ_DATASET=glove RABITQ_RESIDUAL_BITS=4  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/glove/test_run1_glove_residual4_disk.log 2>&1 &&
RABITQ_DATASET=glove RABITQ_RESIDUAL_BITS=8  RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/glove/test_run1_glove_residual8_disk.log 2>&1 &&
RABITQ_DATASET=glove RABITQ_RESIDUAL_BITS=16 RABITQ_RESIDUAL_STORAGE=disk RABITQ_PAYLOAD_MODE=disk ./build/main > logs/glove/test_run1_glove_residual16_disk.log 2>&1
' > logs/glove/test_run1_glove_queue.log 2>&1 &
```
