# Codex任务：HNSW图感知节点重排

目标：在不改变图结构、Recall和量化方式的前提下，通过重排节点存储顺序，提高缓存局部性和QPS。

## 要做什么

1. 固定当前最佳配置：

```text
K=1
Sequential 4-bit RaBitQ
batch_prefetch
prefetch_distance=2
```

2. Float32 HNSW构图完成后，使用level 0图从entry point做BFS，得到新的节点顺序：

```text
old_to_new
new_to_old
```

3. 按新ID重新生成量化索引：

- 每个节点的RaBitQ payload按新ID重新排列；
- 所有层邻接表中的旧ID替换为新ID；
- entry point替换为新ID；
- label保持不变；
- 每个节点的邻居顺序保持不变；
- 图拓扑、节点层级和量化数据都不能改变。

4. 不要原地修改旧索引，创建一个新的重排索引。

5. 不做以下内容：

- 不做route排序；
- 不做short filter；
- 不做1→2→4 bit；
- 不复制邻居量化码；
- 不修改HNSW搜索逻辑；
- 不实现复杂图分区算法。

## 必须验证

重排前后检查：

- graph fingerprint一致；
- 每个label的层级和邻居label集合一致；
- 每个label对应的RaBitQ payload一致；
- Recall差异不超过0.001；
- visited nodes和distance computations基本一致。

## 实验对比

只比较两组：

```text
原始布局 + batch_prefetch(pf=2)
BFS重排布局 + batch_prefetch(pf=2)
```

测试：

```text
ef = 64, 96, 128, 192, 256, 460
```

输出：

- Recall
- QPS
- 平均延迟和P95
- visited nodes
- distance computations
- cache miss
- dTLB miss
- 重排前后的平均邻居ID距离

## 判断标准

保留BFS重排的条件：

- Recall基本不变；
- visited nodes不增加；
- QPS提升至少3%；
- cache miss或dTLB miss下降。

如果QPS提升不足2%，停止继续做更复杂的图重排。
