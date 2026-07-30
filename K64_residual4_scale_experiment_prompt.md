# Codex Prompt：K64 Residual4 Scale 模式对比与 QPS/Recall 优化

## 任务目标

在现有 SIFT10M、K64 聚类、Residual4、Float32 构图的 RaBitQ-HNSW 实现上，只修改 residual4 的 block scale 生成方式和 scale 存储精度，比较以下四种模式：

1. `max_abs_fp32`
2. `mse_opt_fp32`
3. `mse_opt_fp16`
4. `centroid_block_clip_fp16`

本轮禁止修改：

- HNSW 图结构、邻接边、入口点和层级；
- K64 聚类中心及每个向量的 centroid assignment；
- primary4 编码方式；
- residual4 的有符号量化范围 `[-7, 7]`；
- 搜索流程、候选队列、ef、rerank 数量和停止条件；
- graph repair、residual1、cross-batch 等其他实验逻辑。

目标是回答两个问题：

1. 更合理的 scale 是否能降低 residual4 量化误差并提高 Recall；
2. FP16 scale 是否能减少 scale 读取和存储成本，并在固定 Recall 下提升 QPS。

---

## 一、术语和模式定义

设数据库向量属于 centroid `c`，旋转后的真实残差为：

\[
r = R(x-c)
\]

primary4 解码结果为：

\[
\hat r_p = decode\_primary4(r)
\]

Residual4 实际量化的误差为：

\[
e = r-\hat r_p
\]

对每个 residual block 独立量化。Residual4 的有符号整数范围固定为：

\[
q_i\in[-7,7]
\]

统一解码公式为：

\[
\hat e_i=q_i\cdot s
\]

其中 `s` 是当前 block 的 scale。

### 1. `max_abs_fp32`

`abs` 表示 absolute value，即绝对值；`max_abs` 表示 block 内误差绝对值的最大值。

计算：

\[
m=\max_i |e_i|
\]

\[
s=\frac{m}{7}
\]

\[
q_i=clip(round(e_i/s),-7,7)
\]

scale 使用 FP32 存储。

特殊情况：

- 若 `m=0`，令该 block 的所有 `q_i=0`；
- scale 可保存为 `0`；
- 解码时 scale 为 `0` 直接输出全零 residual。

该模式是当前基线。它保证不因动态范围不足而产生明显溢出，但少量 outlier 会把 scale 拉大，使多数普通维度的量化步长变粗。

### 2. `mse_opt_fp32`

`MSE` 表示 Mean Squared Error，即均方误差；`opt` 表示 optimized，即通过优化得到的 scale。

目标：

\[
s^*=\arg\min_{s>0}\sum_i(e_i-sq_i)^2
\]

其中：

\[
q_i=clip(round(e_i/s),-7,7)
\]

使用以下离线迭代求近似最优 scale：

1. 初始化：

\[
s_0=\frac{\max_i|e_i|}{7}
\]

2. 重复 3 次：

\[
q_i=clip(round(e_i/s),-7,7)
\]

\[
s=\frac{\sum_i e_iq_i}{\sum_iq_i^2}
\]

3. 若分母为 0，则该 block 的 `q_i=0`、`s=0`；
4. 每次更新后保证 `s>=epsilon`，建议 `epsilon=1e-12`；
5. 最后再使用最终 `s` 重新生成一次 `q_i`；
6. scale 使用 FP32 存储。

该模式允许少量极端值被截断，从而提高多数维度的表示精度。编码阶段可以稍慢，但在线查询公式不变。

### 3. `mse_opt_fp16`

scale 的求解方式与 `mse_opt_fp32` 完全相同，但最终保存到索引时转换为 IEEE FP16。

要求：

- 编码阶段使用 FP32 完成 scale 优化；
- 保存前将 scale 转换为 FP16；
- 查询阶段加载 FP16 scale，再转换为 FP32 参与距离计算；
- residual4 code 不变；
- 不允许在查询热路径中调用高开销通用转换函数；
- 若 CPU 支持 F16C，使用向量化 FP16→FP32 转换；
- 不支持 F16C 时提供正确的 fallback，并在日志中标记。

该模式用于区分：

- MSE scale 带来的精度收益；
- FP16 scale 带来的存储和读取收益。

### 4. `centroid_block_clip_fp16`

该模式为每个 `(centroid_id, block_id)` 离线学习一个 clipping coefficient：

\[
\alpha_{c,b}\in(0,1]
\]

对属于 centroid `c` 的向量，在 block `b` 上计算：

\[
m=\max_i|e_i|
\]

\[
s=\frac{\alpha_{c,b}\cdot m}{7}
\]

然后：

\[
q_i=clip(round(e_i/s),-7,7)
\]

scale 最终以 FP16 保存。

本模式中：

- `centroid` 表示 K64 中的聚类中心；
- `block` 表示 residual4 的量化块；
- `clip` 表示主动缩小动态范围，允许少数 outlier 饱和；
- `fp16` 表示每个向量每个 block 的最终 scale 仍以 FP16 保存。

`alpha_{c,b}` 的学习要求：

- 只能使用固定的 base training subset；
- 禁止使用 query 或 ground-truth；
- 每个 centroid 至少采样 2048 个向量，不足时使用该 centroid 的全部样本；
- alpha 搜索网格默认：

\[
\{0.50,0.55,0.60,\ldots,0.95,1.00\}
\]

- 对每个 `(c,b)` 选择使平均 block reconstruction MSE 最小的 alpha；
- alpha table 单独保存到索引元数据；
- alpha table 使用 FP16 或 FP32 均可，因为表很小，但日志中必须记录；
- 查询阶段不读取 alpha table，只读取已经保存好的每向量 block scale。

注意：该模式仍保留每向量每 block scale，不是 shared-scale 模式。本轮不改变距离解码公式。

---

## 二、实现要求

### 1. 增加统一配置

新增枚举或等价配置：

- `MAX_ABS_FP32`
- `MSE_OPT_FP32`
- `MSE_OPT_FP16`
- `CENTROID_BLOCK_CLIP_FP16`

提供命令行参数：

`--residual-scale-mode=max_abs_fp32|mse_opt_fp32|mse_opt_fp16|centroid_block_clip_fp16`

索引保存时写入：

- scale mode；
- scale storage dtype；
- centroid count；
- residual bits；
- block count；
- block size；
- format version。

加载时必须校验格式，禁止把 FP16 scale 当成 FP32 读取。

### 2. 保持实验隔离

四种模式必须：

- 使用同一套 K64 centroids；
- 使用相同 centroid assignment；
- 使用同一张 HNSW 图；
- 使用同一 primary4 code；
- 只重建 residual4 code 和 scale；
- 输出不同的新 index；
- 不覆盖原 index。

建议文件名：

- `..._K64_residual4_scale_max_abs_fp32.bin`
- `..._K64_residual4_scale_mse_opt_fp32.bin`
- `..._K64_residual4_scale_mse_opt_fp16.bin`
- `..._K64_residual4_scale_centroid_block_clip_fp16.bin`

### 3. 查询代码

查询阶段保持原有搜索语义，只根据索引元数据选择 scale 加载路径。

要求：

- FP32 scale 走现有路径；
- FP16 scale 使用单独、可向量化的加载和转换路径；
- 不修改 candidate queue、visited、heap、ef 和 rerank 规则；
- 不开启 diagnostics、repair 或 debug instrumentation；
- release 编译；
- 查询热路径禁止动态内存分配。

---

## 三、正确性检查

为每种模式增加以下检查：

1. 编码后所有 `q_i` 均在 `[-7,7]`；
2. scale 不得为 NaN、Inf 或负数；
3. `scale=0` 时该 block 的所有 code 必须为 0；
4. 保存、加载后 scale 和 code 可正确恢复；
5. FP16 模式加载后距离结果与独立 scalar reference 一致；
6. SIMD、scalar、reference 的 top-k 排名一致或在允许的浮点误差内一致；
7. 图结构、degree、label 和 centroid assignment 与基线完全一致。

---

## 四、离线量化误差指标

在固定的 base evaluation subset 上，建议至少采样 100000 个向量。四种模式使用完全相同的样本。

记录：

### 1. Residual reconstruction MAE

\[
MAE_e=\frac{1}{N}\sum_i|e_i-\hat e_i|
\]

### 2. Residual reconstruction RMSE

\[
RMSE_e=\sqrt{\frac{1}{N}\sum_i(e_i-\hat e_i)^2}
\]

### 3. Residual distance MAE

对固定 query-vector pair，比较 residual4 距离和 exact residual 距离：

\[
MAE_d=\frac{1}{P}\sum_j|\hat d_j-d_j^{exact}|
\]

### 4. Residual distance max error

\[
max\_abs_d=\max_j|\hat d_j-d_j^{exact}|
\]

### 5. Clip ratio

先计算未截断整数：

\[
q_i^{raw}=round(e_i/s)
\]

定义：

\[
clip\_ratio=
\frac{\#\{|q_i^{raw}|>7\}}
{\#\{q_i^{raw}\}}
\]

不要仅统计最终 `q_i=±7`，因为合法值也可能自然落在边界。

同时输出：

- global clip ratio；
- per-block clip ratio；
- p50/p95/p99 block clip ratio；
- 每个 centroid 的 clip ratio。

### 6. Scale 分布

输出：

- min；
- p50；
- p95；
- p99；
- max；
- zero-scale ratio。

---

## 五、Recall 与性能实验

保持：

- 数据集：SIFT10M；
- K=64；
- 图结构不变；
- `k=10`；
- query 集合固定；
- 单线程；
- 固定 CPU core；
- 相同 NUMA 配置；
- 相同编译参数；
- 相同预热流程。

测试：

\[
ef\in\{50,100,150,200,300,460\}
\]

每个模式、每个 ef：

- 预热至少 100 次 query；
- 正式运行完整 query 集；
- 重复至少 10 轮；
- before/after 交替运行，避免温度和频率偏差；
- 输出 median 和 p95，不以单次结果作为结论。

记录：

- Recall@10；
- candidate coverage@10；
- primary Recall@10；
- residual4 rerank Recall@10；
- expanded nodes；
- edges scanned；
- unique nodes touched；
- distance count；
- rerank candidate count；
- query setup latency；
- graph search latency；
- rerank latency；
- total latency；
- QPS；
- index size；
- residual code bytes；
- scale bytes。

若硬件计数器不可用，输出：

- `cycles=unavailable`
- `instructions=unavailable`
- `llc_misses=unavailable`
- `branch_misses=unavailable`

不得因此中止 benchmark。

---

## 六、固定 Recall 对比

从 ef sweep 中插值或直接选择达到下列目标 Recall 的最小 ef：

- 0.97
- 0.98
- 0.99
- 0.995

对每个 scale mode 输出：

- 达到目标 Recall 的最小 ef；
- total latency；
- QPS；
- rerank latency；
- distance count；
- unique nodes touched。

最终必须生成汇总表，重点比较固定 Recall 下的总延迟，而不是只比较固定 ef。

---

## 七、判定标准

### `mse_opt_fp32` 是否有效

满足以下任一项即可认为值得保留：

- residual reconstruction MAE/RMSE 明显低于 `max_abs_fp32`；
- residual distance MAE 降低至少 10%；
- 固定 ef 下 Recall@10 提升至少 0.2 个百分点；
- 固定 Recall 下 total latency 下降至少 3%。

### `mse_opt_fp16` 是否有效

相对 `mse_opt_fp32`：

- Recall 下降不超过 0.1 个百分点；
- residual distance MAE 增幅可接受；
- scale 存储减少约一半；
- 固定 Recall 下 total latency 或 QPS 有可测收益。

### `centroid_block_clip_fp16` 是否有效

相对 `mse_opt_fp16`：

- residual distance MAE 进一步下降，或；
- 固定 ef 下 Recall 提升，或；
- 固定 Recall 下所需 ef 降低；
- clip ratio 不得异常升高；
- 不得出现少数 centroid 精度严重恶化。

如果该模式只改善训练样本而不改善独立 query 结果，则判定为过拟合，不保留。

---

## 八、最终输出

输出一份总日志和一份 Markdown/CSV 汇总，至少包含：

| scale mode | scale dtype | residual MAE | distance MAE | max error | clip ratio | ef | Recall@10 | rerank us | total us | QPS | index MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

额外输出固定 Recall 表：

| target Recall | scale mode | min ef | total us | QPS | distance count | unique touched |
|---:|---|---:|---:|---:|---:|---:|

最终结论必须明确回答：

1. `max_abs` 的 outlier 问题是否真实存在；
2. `mse_opt` 是否改善 residual4 的排序精度；
3. FP16 scale 是否引入明显精度损失；
4. centroid-aware clipping 是否优于逐向量 MSE scale；
5. 哪个模式在 Recall–QPS 曲线上占优；
6. 是否值得将该模式设为 K64 residual4 的默认实现。
