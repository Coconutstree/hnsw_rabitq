# 基于Extended RaBitQ渐进距离计算的量化HNSW

## 1. 方法概述

本方法面向高维向量近似最近邻搜索，将HNSW与Extended RaBitQ结合。核心目标是在不保存原始Float32数据库向量的情况下，使用4-bit主码完成图搜索，并通过低成本的MSB概率判断跳过一部分完整4-bit距离计算，最后使用4-bit residual对少量候选进行精排。

当前DBpedia实验使用975000条1536维数据库向量和5000条查询向量，距离为平方L2，HNSW参数为M=16、efConstruction=200，查询返回Recall@10，最多精排100个候选。

## 2. 数据编码

### 2.1 主4-bit编码

构建索引前，数据库向量先减去全局质心，再进行随机符号变换和Hadamard旋转。旋转后的单位方向使用Extended RaBitQ量化为每维4-bit主码。

每一维用4个二进制位保存，可以拆成：

```latex
\text{4-bit主码}=\text{1-bit MSB}+\text{remaining 3-bit}
```

MSB是Most Significant Bit，即4个bit中权重最高的第一位。它可以快速给出向量方向的粗略估计；剩余3位继续补充幅值，使距离估计更准确。节点还保存向量范数、量化尺度和误差界所需的校正参数。图搜索完成后不需要保留数据库Float32向量。

### 2.2 Residual编码

主4-bit近似值与原始向量之间的误差定义为residual。Residual同样量化为4-bit，但只在最终候选精排时读取，不参与HNSW构图和图搜索。

当前有两种residual尺度方案：

- Uniform residual4：1536维共用一个量化尺度。
- Block16 residual4：每16维为一块，每块保存独立量化尺度，共96块。

Block16可以适应不同维度区间的误差幅度，减少统一尺度造成的量化误差，提高精排距离的准确性。

## 3. HNSW构图

### 3.1 非对称4-bit构图

A、B、C采用非对称构图。插入节点时，查询侧使用本次插入的原始Float32向量，数据库侧使用已经保存的4-bit主码：

```latex
\text{Float32 query}\longrightarrow\text{Extended RaBitQ 4-bit database code}
```

查询向量只准备一次旋转和质心上下文，插入导航、候选扩展、邻居多样性裁剪和反向邻接表裁剪均使用同一套非对称平方L2距离。Residual不参与构图。

### 3.2 对称4-bit构图

D采用4-bit对4-bit对称构图，构图两侧都不读取Float32向量。内积用来衡量两个向量方向的相似程度。两个4-bit码的内积经过双边校正后估计原向量内积：

```latex
\widehat{\langle o,q\rangle}
=
\frac{\langle\bar{o},\bar{q}\rangle}
{\langle\bar{o},o\rangle\langle\bar{q},q\rangle}
```

其中，o和q表示原向量方向，带横线的o和q表示对应的4-bit量化方向。分母用于分别修正两侧的量化误差。所需校正量已经在编码时保存在节点中，不需要保留Float32向量。估计内积限制在[-1,1]后转换为平方L2：

```latex
d(o,q)
=
\lVert o\rVert^2+\lVert q\rVert^2
-2\lVert o\rVert\lVert q\rVert
\widehat{\langle o,q\rangle}
```

## 4. 图搜索中的渐进距离计算

查询侧始终使用原始Float32查询，数据库侧使用4-bit主码。Level 0启用batch prefetch，预取距离为2，并保持原始邻接表顺序、候选队列语义、停止条件和ef不变。

### 4.1 先用最高位快速判断

当候选队列已经达到ef时，先只读取每一维4-bit码的最高位MSB。MSB计算量较小，可以快速估计查询向量q与数据库向量x的内积。下标1表示该估计只使用了1个bit：

```latex
\widehat{\langle x,q\rangle}_1
```

由于只看1个bit会产生误差，因此再增加一个概率误差范围：

```latex
e
=
\sqrt{\frac{1-\alpha^2}{\alpha^2}}
\frac{\epsilon_0}{\sqrt{D-1}}
```

其中，alpha表示1-bit量化方向与原向量方向的相关程度，D是向量维度，当前使用epsilon0=1.9控制概率误差范围。

将1-bit内积估计与误差范围相加，得到可能的最大内积：

```latex
\langle x,q\rangle_{\mathrm{upper}}
=
\operatorname{clamp}
\left(\widehat{\langle x,q\rangle}_1+e,-1,1\right)
```

内积越大，L2距离越小。因此用最大可能内积转换出最小可能距离，也就是距离下界：

```latex
d_{\mathrm{lower}}
=
\lVert x\rVert^2+\lVert q\rVert^2
-2\lVert x\rVert\lVert q\rVert
\langle x,q\rangle_{\mathrm{upper}}
```

### 4.2 只有必要时才读取剩余3位

在节点影响候选队列前保存当前候选队列中的最差距离。如果节点的最小可能距离仍然更差：

```latex
d_{\mathrm{lower}}>d_{\mathrm{saved\_lowerBound}}
```

则说明该节点即使取最有利的误差结果，也无法进入候选队列，因此直接跳过剩余3位计算。

如果不能排除该节点，才读取剩余3位，并与已经算出的MSB结果合并成完整4-bit距离。MSB不会重复计算。由于使用的是概率误差界，这种剪枝不是严格零错误剪枝，因此实验需要同时报告Recall变化和误剪数量。

## 5. Residual精排

HNSW搜索先使用主4-bit距离产生候选集合。随后最多选取100个候选，读取其residual sidecar并计算更精确的近似距离，再重新排序得到最终Top-10。

Residual精排不会改变HNSW图的遍历过程，只改善最终候选次序。Residual独立存放在mmap sidecar中，图搜索阶段不会读取。

## 6. A/B/C/D消融实验

| 实验 | 构图方式 | 最终精排 |
|---|---|---|
| A | Float32对4-bit非对称构图 | 不使用residual |
| B | Float32对4-bit非对称构图 | 全维统一尺度residual4 |
| C | Float32对4-bit非对称构图 | Block16独立尺度residual4 |
| D | 4-bit对4-bit对称构图 | Block16独立尺度residual4 |

A、B、C用于比较精排策略；D用于考察完全量化构图能否降低构图成本。为了公平测量构图时间，四组实验均使用`RABITQ_FORCE_REBUILD=1`，禁止复用已有索引、共享图和质心缓存。

## 7. 方法特点

1. 数据库向量以4-bit主码保存，显著降低图节点payload和距离计算的数据带宽。
2. 将4-bit码拆成MSB与remaining 3-bit，先用MSB概率界筛选，再按需计算剩余位。
3. 未剪枝节点复用MSB贡献，避免低bit判断和完整距离之间的重复计算。
4. 将MSB物理独立保存，并对邻居批量处理，降低逐节点调度和无效缓存访问。
5. 使用Block16 residual4精排，在较小附加存储下改善最终距离排序。
6. 同时支持Float32对4-bit非对称构图和4-bit对4-bit对称构图，用于权衡构图速度与图质量。

## 8. 评价指标

实验统一报告：

- Recall@10；
- QPS、平均查询延迟和P95延迟；
- HNSW访问节点数和完整距离计算次数；
- MSB概率剪枝比例及完整4-bit节省比例；
- 概率误剪数量；
- residual精排耗时；
- 纯图构建时间、总构建时间和索引存储量。

构图时间重点区分：

```text
Graph construction time：只统计HNSW图构建
Build time：还包含质心训练、编码、residual生成和索引保存
```
