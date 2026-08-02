# 1+1-bit Full2Bit 实验总结

日期：2026-08-02

## 一句话结论

今天的实验说明：`1-bit primary + 1-bit residual = full 2-bit` 这条路线本身可以跑通，但不适合作为当前主方法。

最值得保留的主路径仍然是：

```text
K256 + primary4 + residual4 + block16 + mse_opt_fp16
```

`1+1-bit` 的问题不在代码算错，而在距离质量太弱：完整 2-bit 的排序误差很大，primary 1-bit 更差。用 progressive 或 blockwise 想省计算，目前都会变成“Recall 不涨，速度还变慢”。

## 实验主线

今天围绕 `1+1-bit` 做了这些尝试：

1. 干净 full2bit baseline
2. full2bit 距离函数正确性验证
3. full2bit 导航 vs 最终排序诊断
4. primary1bit 单独导航
5. progressive 1+1-bit
6. full2bit LUT 加速
7. full2bit upper primary4 尝试
8. primary oracle / gap threshold 诊断
9. blockwise full2bit early termination

这些实验的共同目标是回答三个问题：

```text
1. 1+1-bit 的距离函数是不是正确？
2. Recall 低是导航丢候选，还是最终 2-bit 排序错？
3. 能不能只读 primary 1-bit，必要时再读 secondary 1-bit，从而加速？
```

## 1. Full2Bit 距离函数验证

日志：

```text
logs/sift10m/test/full2bit_distance_validation.log
```

验证了三种距离：

```text
直接解码完整 2-bit code
拆成 primary 1-bit + residual 1-bit 的标量实现
当前 optimized full2bit kernel
```

关键结果：

```text
u_equals_2p_plus_r_mismatches = 0
direct_vs_split_mae = 0.007944531
direct_vs_optimized_mae = 0.006958594
optimized_exact_order_match = 1.0
optimized_top10_overlap_with_direct = 1.0
optimized_top100_overlap_with_direct = 1.0
```

结论：

```text
full2bit 距离函数基本正确。
u_i = 2 * primary_i + residual_i 没问题。
optimized kernel 和 reference 排序一致。
```

所以后面 Recall 低，不是因为 full2bit 距离内核写错。

## 2. Full2Bit 导航 vs 最终排序诊断

日志：

```text
logs/sift10m/test/full2bit_navigation_oracle.log
```

做了三组对比：

```text
FULL2BIT_NAV_FULL2BIT_RETURN
完整 2-bit 导航 + 完整 2-bit 返回

FULL2BIT_NAV_FLOAT32_RERANK_TOP100
完整 2-bit 导航 + Float32 rerank Top100

FULL2BIT_NAV_FLOAT32_RERANK_TOP500
完整 2-bit 导航 + Float32 rerank Top500
```

ef=460 的结果：

```text
full2bit return Recall@10       = 0.678
float32 rerank Top100 Recall@10 = 0.998
float32 rerank Top500 Recall@10 = 0.999
CandidateCoverage@10            = 0.999
```

结论非常直接：

```text
full2bit 导航其实能找到真邻居候选。
Recall 低主要是最终 full2bit 距离排序错了。
```

也就是说，full2bit 的候选覆盖很好，但它自己给 Top100 排序时把好候选排错了。

## 3. Full2Bit baseline

日志：

```text
logs/sift10m/test_ef_400_M_32_K_256_full2bit.log
```

关键结果：

| ef | Recall@10 | latency |
|---:|---:|---:|
| 100 | 0.6740 | 1151.79 us |
| 300 | 0.6753 | 2044.78 us |
| 460 | 0.6751 | 2701.17 us |

结论：

```text
完整 2-bit Recall@10 大约卡在 0.675。
ef 从 100 增到 460 基本不涨 Recall。
```

这说明 full2bit 的瓶颈不是 ef 太小，而是 2-bit 量化距离排序上限太低。

## 4. Primary1Bit 单独导航

日志：

```text
logs/sift10m/test_ef_400_M_32_K_256_primary1bit.log
```

关键结果：

| ef | Recall@10 | latency |
|---:|---:|---:|
| 100 | 0.2770 | 4308.78 us |
| 300 | 0.2716 | 8660.74 us |
| 460 | 0.2708 | 11550.40 us |

结论：

```text
primary 1-bit 单独导航不可行。
Recall 很低，而且 ef 增大后不变好，速度还更慢。
```

原因：

```text
primary 1-bit 距离太粗，局部排序严重失真。
搜索会被错误距离带偏。
```

所以不能把 primary 1-bit 当成独立导航距离。

## 5. Progressive 1+1-bit

日志：

```text
logs/sift10m/test_ef_400_M_32_K_256_progressive1p1bit.log
```

思路：

```text
先算 primary 1-bit。
只有不确定时再读 secondary 1-bit。
目标是接近 full2bit Recall，但减少 secondary 读取。
```

关键结果：

| ef | Recall@10 | latency |
|---:|---:|---:|
| 100 | 0.6744 | 2358.57 us |
| 300 | 0.6753 | 4265.46 us |
| 460 | 0.6751 | 5914.35 us |

和 full2bit 比：

```text
Recall 基本一样。
速度明显更慢。
```

原因：

```text
progressive 为了保证 Recall，最后几乎还是要 refine 大部分节点。
等于先算一遍 primary 1-bit，再对大量节点重算 full2bit。
重复计算太多。
```

所以当前 progressive 不可行。

## 6. Primary Oracle / Gap Threshold 诊断

日志：

```text
logs/sift10m/test/full2bit_primary_oracle.log
```

这个实验不跑正式搜索，而是看 primary 1-bit 和 full2bit 在关键比较上是否一致。

关键结果：

```text
candidate_queue_adjacent_order flip_rate ≈ 0.498
new_neighbor_vs_full_topk_worst flip_rate ≈ 0.270
primary_boundary_stop_proxy flip_rate ≈ 0.450
```

直白解释：

```text
primary 1-bit 做排序时，差不多一半相邻顺序会和 full2bit 不一致。
新候选是否应该进 topK 的判断，也有约 27% 会翻转。
停止条件附近也很危险。
```

这说明：

```text
primary 1-bit 没有稳定的安全区。
如果少 refine，会漏掉大量关键翻转。
如果多 refine，又退化成 full2bit，还多算一遍 primary。
```

所以基于 primary gap 的 progressive threshold 不值得作为主线。

## 7. Full2Bit LUT 加速

日志：

```text
logs/sift10m/test_ef_400_M_32_K_256_full2bit_lut.log
```

关键结果：

| ef | Recall@10 | latency |
|---:|---:|---:|
| 100 | 0.6740 | 1305.30 us |
| 300 | 0.6753 | 2192.33 us |
| 460 | 0.6751 | 2870.79 us |

对比普通 full2bit：

```text
Recall 一样。
速度没有更快，反而略慢。
```

结论：

```text
LUT 版本目前不可取。
full2bit 的主要问题也不是这个 kernel 一点点查表开销，而是 Recall 上限太低。
```

## 8. Full2Bit Upper Primary4 尝试

日志：

```text
logs/sift10m/test_ef_400_M_32_K_256_full2bit_upper_primary4.log
```

思路：

```text
尝试在 upper layer 用更强的 primary4 路由，改善 full2bit 导航。
```

结果：

```text
效果不好，没有解决 full2bit Recall 低的问题。
```

原因：

```text
前面的 navigation oracle 已经说明，full2bit 候选覆盖其实不差。
主要错在最终 full2bit 排序。
所以只改 upper layer 路由，不会解决核心问题。
```

## 9. Blockwise Full2Bit Early Termination

日志：

```text
logs/sift10m/test_ef_400_M_32_K_256_full2bit_blockwise64.log
logs/sift10m/test_ef_400_M_32_K_256_full2bit_blockwise128.log
logs/sift10m/test_ef_400_M_32_K_256_full2bit_blockwise256.log
```

思路：

```text
完整 full2bit 距离按 block 累加。
如果当前候选已经不可能进入 top_candidates，就提前停止。
```

block128 结果：

| ef | Recall@10 | latency | early termination rate | avg processed dims |
|---:|---:|---:|---:|---:|
| 100 | 0.6740 | 2458.49 us | 0.916486 | 128 |
| 300 | 0.6753 | 5034.57 us | 0.927396 | 128 |
| 460 | 0.6751 | 7005.94 us | 0.930075 | 128 |

结论：

```text
Recall 和 full2bit 一致，但速度更慢。
```

为什么不可行：

```text
虽然日志显示 early termination rate 很高，
但 avg processed dims 基本还是 128。

也就是说，大多数候选都已经把完整 128 维算完了，
只是最后被标成 early terminated。
实际没有省距离计算，反而多了 block 判断和分支开销。
```

block64 / block128 / block256 都没有明显收益：

| block | ef=460 Recall@10 | ef=460 latency | avg processed dims |
|---:|---:|---:|---:|
| 64 | 0.6751 | 7035.35 us | 127.87 |
| 128 | 0.6751 | 7005.94 us | 128 |
| 256 | 0.6751 | 7336.58 us | 128 |

所以 blockwise full2bit 暂时不作为主路径。

## 每个日志文件代表什么

```text
logs/sift10m/test/full2bit_distance_validation.log
```

验证 full2bit 距离函数是否正确。重点看 `u=2p+r` 是否错、reference 和 optimized 排序是否一致。

```text
logs/sift10m/test/full2bit_navigation_oracle.log
```

判断 Recall 低是导航错还是最终排序错。结果说明主要是最终 full2bit 排序错。

```text
logs/sift10m/test/full2bit_primary_oracle.log
```

判断 primary 1-bit 是否能安全过滤。结果说明 primary 和 full2bit 的关键决策翻转很多，不能安全少读 secondary。

```text
logs/sift10m/test/full2bit_search_profile.log
```

搜索计时诊断。看 distance 次数、payload 读取字节、访问节点数、heap 开销等。

```text
logs/sift10m/test_ef_400_M_32_K_256_full2bit.log
```

干净 full2bit baseline。也就是完整 2-bit 距离从头到尾导航和返回。

```text
logs/sift10m/test_ef_400_M_32_K_256_primary1bit.log
```

只用 primary 1-bit 搜索。结果很差，用来证明 primary 1-bit 不能独立导航。

```text
logs/sift10m/test_ef_400_M_32_K_256_progressive1p1bit.log
```

progressive 1+1-bit 搜索。结果 Recall 接近 full2bit，但速度更慢，说明 refine 太多。

```text
logs/sift10m/test_ef_400_M_32_K_256_full2bit_lut.log
```

full2bit LUT 加速尝试。Recall 不变，速度没提升。

```text
logs/sift10m/test_ef_400_M_32_K_256_full2bit_upper_primary4.log
```

upper layer 使用 primary4 的尝试。效果不好，因为核心问题不是候选覆盖，而是最终排序。

```text
logs/sift10m/test_ef_400_M_32_K_256_full2bit_blockwise64.log
logs/sift10m/test_ef_400_M_32_K_256_full2bit_blockwise128.log
logs/sift10m/test_ef_400_M_32_K_256_full2bit_blockwise256.log
```

blockwise full2bit early termination。结果 Recall 一样，但速度更慢，因为基本没有真正提前停止距离计算。

## 最终判断

今天 `1+1-bit` 方向的结论是：

```text
full2bit 距离实现正确。
full2bit 导航能找到好候选。
full2bit Recall 低主要来自 2-bit 最终排序误差。
primary1bit 太弱，不能单独导航。
progressive 会大量 refinement，没有加速价值。
blockwise full2bit 没真正省维度计算，也没有加速价值。
```

所以当前不要把这些方法放进主路径。

当前最应该继续保留和优化的是：

```text
K256 + residual4 + block16 + mse_opt_fp16
```

`1+1-bit/full2bit` 可以作为论文或实验对照 baseline，但不适合作为当前主方法。

