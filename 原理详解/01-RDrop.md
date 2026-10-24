# R-Drop：用两个 dropout 视角约束训练一致性

> 阅读顺序：[R-Drop](./01-RDrop.md) → [SCL](./02-SCL.md) → [Unsloth](./03-Unsloth.md) → [监督记忆队列](./04-MoCo监督记忆队列.md) → [小 batch 复盘](./05-小batch与SCL问题复盘.md)

## 1. 一句话结论

本项目的 R-Drop（Regularized Dropout）不是单纯多算一次损失，而是让同一输入经过两次随机 dropout 前向得到的两个预测分布彼此一致，并在两个视角的平均交叉熵之外加入双向对称 KL（Kullback-Leibler divergence）；当前 DeBERTa-v2 实现使用 `ALPHA_RDROP=0.6` 和 `reduction="batchmean"`。

## 2. 问题背景与目标

训练阶段启用了 dropout，推理阶段则关闭 dropout。即使输入完全相同，训练时的前向结果也会因随机失活而波动；普通交叉熵只要求两个随机子网络各自拟合标签，没有显式要求它们给出接近的预测。

对 IMDb 二分类而言，目标损失仍是正确分类，但仅靠一层 dropout 或一次交叉熵不能直接约束模型内部两个随机路径的一致性。R-Drop 的目标是增加一个只依赖同一输入两个预测分布的正则项，使分类器对训练期随机性更稳定。

它解决的是“训练与推理随机性不一致”问题，不等同于数据增强，也不负责把不同类别在表示空间中拉开。后者由 [SCL](./02-SCL.md) 负责。

## 3. 方法原理

设同一批输入为 `x`，标签为 `y`。对同一模型分别执行两次训练态前向：

$$
z_1=f(x;\epsilon_1),\qquad z_2=f(x;\epsilon_2),
$$

其中 `ε_1,ε_2` 是两次独立采样的 dropout 掩码。令

$$
p_1=\mathrm{softmax}(z_1),\qquad p_2=\mathrm{softmax}(z_2).
$$

两次交叉熵的均值为

$$
\mathcal L_{\mathrm{CE}}
=\frac{1}{2}\left[\mathrm{CE}(z_1,y)+\mathrm{CE}(z_2,y)\right].
$$

双向对称 KL 为

$$
\mathcal L_{\mathrm{KL}}
=\frac{1}{2}\left[
D_{\mathrm{KL}}(p_1\Vert p_2)
+D_{\mathrm{KL}}(p_2\Vert p_1)
\right].
$$

最终损失为

$$
\mathcal L_{\mathrm{RDrop}}
=\mathcal L_{\mathrm{CE}}+\alpha_{\mathrm{RDrop}}\mathcal L_{\mathrm{KL}}.
$$

双向形式比单向方向更对称：它同时惩罚“第一个视角偏离第二个视角”和反向偏离，不预设哪一个前向是真值。

### `batchmean` 的作用

PyTorch 的 `kl_div(..., reduction="batchmean")` 对类别维求和，再除以 batch 维大小。项目在 DeBERTa 实现中显式采用这一归一化，使 KL 项保持“每样本平均”的量级，和交叉熵处于相近尺度。

若使用 `reduction="sum"`，KL 会把整个 batch 的分数加总。batch 变大时，即使后向 KL 权重仍是 0.6，实际总损失中 KL 的占比也会随之放大；这不是无法运行，而是损失尺度随 batch 改变，实验之间难以比较。项目源码中的注释以一个 batch 内有 16 个样本的情形说明这一风险。

`batchmean` 是 KL 项内部的归一化。Hugging Face `Trainer` 还会按 `gradient_accumulation_steps` 处理反向传播的损失尺度，因此复现时不应再手工除以梯度累积次数。

## 4. 本项目中的实现

当前主线实现位于 [DeBERTa-v2 的 R-Drop 对照脚本](../imdb_deberta_v2_unsloth_rdrop.py) 和 [R-Drop + SCL 脚本](../imdb_deberta_v2_unsloth_rdrop_scl.py)，组合实验位于 [R-Drop + SCL + 监督记忆队列脚本](../imdb_sentiment_analusis_torch_/imdb_deberta_v2_unsloth_rdrop_scl_queue.py)。

核心路径如下：

```python
pooled1, logits1 = self._encode(input_ids, attention_mask)
if labels is not None and (USE_RDROP or USE_SCL):
    pooled2, logits2 = self._encode(input_ids, attention_mask)

ce1 = F.cross_entropy(logits1.float(), labels)
ce2 = F.cross_entropy(logits2.float(), labels)
kl_part = symmetric_kl(logits1, logits2)
total = 0.5 * (ce1 + ce2) + ALPHA_RDROP * kl_part
```

`symmetric_kl` 先把 logits 转成 `float32` 的 log-softmax，再调用两次带 `log_target=True` 的 `batchmean` KL：

```python
la = F.log_softmax(a.float(), dim=-1)
lb = F.log_softmax(b.float(), dim=-1)
return 0.5 * (
    F.kl_div(la, lb, reduction="batchmean", log_target=True)
    + F.kl_div(lb, la, reduction="batchmean", log_target=True)
)
```

两个视角来自同一个 `DebertaForIMDB.forward` 内连续两次 `_encode`。每次调用都会重新经过 Transformer 内部 dropout 和池化后的 `self.dropout`，所以输入张量相同，随机掩码不同。评估时不传 `labels`，代码只做一次前向。

需要特别区分历史代码 [imdb_bert_rdrop.py](../imdb_bert_rdrop.py)：它同样跑了两次 BERT 前向，但损失写成 `loss + ce_loss + kl_loss`，KL 使用 `reduction="sum"`，也没有 `ALPHA_RDROP`。它不是当前 R-Drop 对照线的等价实现，不能把它的损失数值与 `0.6 * batchmean-KL` 版本直接比较。

## 5. 关键参数与训练流程

| 项目 | 当前 R-Drop 对照 | R-Drop + SCL 对照 | 队列组合 |
|---|---:|---:|---:|
| `ALPHA_RDROP` | 0.6 | 0.6 | 0.6 |
| KL 归一化 | `batchmean` | `batchmean` | `batchmean` |
| `BATCH_SIZE` | 2 | 2 | 2 |
| `GRAD_ACCUM` | 4 | 4 | 4 |
| `SEED` | 3407 | 3407 | 3407 |
| LoRA | `r=16`, `alpha=32` | `r=16`, `alpha=32` | `r=16`, `alpha=32` |
| 学习率 | LoRA `2e-4`，分类头 `1e-3` | 同左 | 同左 |

参数来源分别为 [R-Drop 对照线](../imdb_deberta_v2_unsloth_rdrop.py)、[R-Drop + SCL](../imdb_deberta_v2_unsloth_rdrop_scl.py) 和 [队列组合线](../imdb_sentiment_analusis_torch_/imdb_deberta_v2_unsloth_rdrop_scl_queue.py)。所有脚本都把分类头 logits 转成 `float32` 后计算交叉熵。

训练流程是：

1. 同一 micro-batch 前向两次；
2. 分别计算交叉熵；
3. 对两个 softmax 分布计算双向 KL；
4. 按上式组合 R-Drop 损失；
5. 若启用 SCL，再附加 `ALPHA_SCL * SCL`；
6. 反向传播和更新 LoRA、分类头参数。

## 6. 实验现象、失败过程与改进

源码注释把错误的 KL 归一化列为排查项：若日志中的 `kl` 达到几十或几百，首先应检查是否误用了 `reduction="sum"`。旧 [BERT R-Drop 脚本](../imdb_bert_rdrop.py) 确实使用 `sum`，这为“KL 项过大”的风险提供了直接的源码证据。

当前主线会打印首个训练步的 `ce`、`kl`、`scl` 和 `total`；训练记录还包含 `grad_norm`。可检查的是这些量是否为有限值、KL 是否与 CE 处于合理尺度，以及总损失是否在多个日志点之间波动而非持续发散。现有材料没有保存可直接归因于 R-Drop 的最终准确率差异，因此不能声称 R-Drop 单独带来了某个分数。

改进包括：

- 将 KL 统一到 `float32` 并采用 `batchmean`；
- 用两次预测的平均 CE，而不是旧实现中两个 CE 的直接相加；
- 用显式权重 `ALPHA_RDROP=0.6` 控制一致性项；
- 把 R-Drop-only 作为独立对照脚本，便于和 SCL、队列组合线区分。

## 7. 效果、局限性及适用条件

R-Drop 的直接目标是改善两个 dropout 路径的预测一致性，适合训练中使用 dropout 的分类模型。它的效果取决于两个视角是否真的不同：若在 `eval()` 模式下关闭 dropout，KL 通常接近零；若视差过大，KL 可能主导总损失。

主要风险有：

- KL 项过大或 KL 权重过高，模型可能优先追求两视图一致而忽视标签监督；
- 两次视角差异过大时，梯度噪声和不稳定性增加；
- `sum` 与 `batchmean` 混用会改变不同 batch 大小下的有效权重；
- 两次前向增加计算和激活显存，对 1.5B 模型尤其明显；
- 日志中的 `kl` 很小不自动证明泛化更好，仍需同一数据、种子和更新预算的对照实验。

现有材料没有记录 R-Drop-only、CE-only、R-Drop + SCL 在相同训练预算下的可比准确率。已记录的队列实验低于普通 LoRA，但该实验受 Kaggle GPU 会话时限影响，被迫使用 `MAX_LEN=384`、小 batch 和单 epoch，并增加了两次前向开销；因此不能把分数下降单独归因于 R-Drop。

## 8. 复现与检查清单

- [ ] 确认使用 [当前对称 KL 实现](../imdb_deberta_v2_unsloth_rdrop_scl.py)，而不是旧 `sum` 版本。
- [ ] 确认 `ALPHA_RDROP=0.6`，除非消融设计明确改变这一项。
- [ ] 确认训练态前向两次，评估态只前向一次。
- [ ] 检查首步 `ce`、`kl`、`total` 都是有限值。
- [ ] 对照不同配置时保持数据、seed、batch、梯度累积和更新次数一致。
- [ ] 如需同时启用 SCL，按 [SCL 文档](./02-SCL.md) 确认 CE、R-Drop 和 SCL 是三个独立分量。
- [ ] 不从未记录的日志推断准确率或显存收益。
