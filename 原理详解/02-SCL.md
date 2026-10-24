# SCL：用标签组织同句双视角与同类样本

> 阅读顺序：[R-Drop](./01-RDrop.md) → [SCL](./02-SCL.md) → [Unsloth](./03-Unsloth.md) → [监督记忆队列](./04-MoCo监督记忆队列.md) → [小 batch 复盘](./05-小batch与SCL问题复盘.md)

## 1. 一句话结论

本项目的 SCL（Supervised Contrastive Learning，监督对比学习）把同一句子的两个 dropout 视角以及同标签样本作为正样本、不同标签样本作为负样本，在温度 `0.07` 的学习表示上做归一化对比；它是分类 CE 之外的独立表示约束，不是交叉熵项内部的一部分。

## 2. 问题背景与目标

自监督对比学习（例如 SimCLR）不知道类别标签：同一样本的两个增强视角互为正样本，其他样本为负样本。这样做可以让表示具有判别性，但会把同一类别的不同句子默认当作负样本。

监督对比学习引入标签。IMDb 是二分类任务，同类不同句子应靠近、异类应远离。项目希望 CE 负责标签决策，SCL 直接约束 pooled 句向量，使分类头之外的特征空间也获得类别结构。

项目还同时使用 [R-Drop](./01-RDrop.md)。两者共享两个 dropout 视角，但约束对象不同：R-Drop 约束两个 logits 概率分布的一致性，SCL 约束 pooled 特征之间的相对距离。

## 3. 方法原理

对样本 `i`，记归一化特征为 `z_i`，温度（temperature）为 `τ`。与 `i` 标签相同的其他视角组成正样本集 `P(i)`，其余候选组成负样本集。监督对比损失可写为

$$
\mathcal L_i
=-\frac{1}{|\mathcal P(i)|}
\sum_{p\in\mathcal P(i)}
\log
\frac{\exp(z_i^\top z_p/\tau)}
{\sum_{a\in\mathcal A(i)}\exp(z_i^\top z_a/\tau)},
$$

其中 `A(i)` 是 anchor `i` 的对比候选集合，不含自身。整个 batch 的损失是有效 anchor 的平均。

- anchor：当前被优化的参考特征；
- positive：与 anchor 标签相同、但不是 anchor 自身的特征；
- negative：用于归一化的其他标签特征；
- temperature：控制相似度 logits 的尖锐程度；本项目统一取 0.07。

当前实现的 `base_temperature` 也取 0.07，因此 `temperature / base_temperature = 1`。若两者不同，代码会额外乘上这个比例。

### 两个 dropout 视角为什么可以构成正样本

同一句子只改变随机 dropout 掩码，语义标签不变。两个视角在目标空间中应接近，因此可以作为“同标签、同语义、不同随机子网络”的正样本。

对 batch 内 `B` 个句子，项目构造

$$
Z=[Z^{(1)};Z^{(2)}]\in\mathbb R^{2B\times H}.
$$

每个 anchor 与另一个视角自身配对；若 batch 中还有同标签句子，其两个视角也会成为正样本。全部 `2B` 个特征都参与候选归一化。

### 与自监督对比学习的区别

| 维度 | SimCLR / 自监督对比学习 | 本项目的 SCL |
|---|---|---|
| 正样本 | 同一原样本的不同增强 | 同一句子的另一 dropout 视角；同标签其他样本 |
| 负样本 | 其他样本或视角 | 不同标签候选 |
| 标签 | 不使用 | 必须使用 |
| 目标 | 实例不变性 | 同类聚合、异类分离，并保留实例双视角约束 |

这里的“增强”只有随机 dropout，不是裁剪、回译或句子改写等专门的数据增强。

## 4. 本项目中的实现

当前 DeBERTa-v2 SCL 在 [R-Drop + SCL 脚本](../imdb_deberta_v2_unsloth_rdrop_scl.py) 中实现，历史通用实现位于 [losses.py](../losses.py)，训练器版本的入口位于 [imdb_bert_scl_trainer.py](../imdb_bert_scl_trainer.py)。

关键前向流程如下：

```python
pooled1, logits1 = self._encode(input_ids, attention_mask)
if labels is not None and (USE_RDROP or USE_SCL):
    pooled2, logits2 = self._encode(input_ids, attention_mask)

ce1 = F.cross_entropy(logits1.float(), labels)
total = 0.5 * (ce1 + F.cross_entropy(logits2.float(), labels))
total = total + ALPHA_RDROP * symmetric_kl(logits1, logits2)

if USE_SCL:
    feats = torch.stack([pooled1, pooled2], dim=1)  # [B, 2, H]
    scl_part = SupConLoss(TEMPERATURE)(feats, labels)
    total = total + ALPHA_SCL * scl_part
```

由此得到完整组合

$$
\mathcal L
=\frac{1}{2}\left(\mathrm{CE}_1+\mathrm{CE}_2\right)
+\alpha_{\mathrm{RDrop}}\mathcal L_{\mathrm{KL}}
+\alpha_{\mathrm{SCL}}\mathcal L_{\mathrm{SCL}}.
$$

交叉熵不是 SCL 公式的组成部分。它属于分类目标，R-Drop 的一致性是另一项，SCL 只是第三项。

### 数值稳定处理

当前 `SupConLoss.forward` 做了四件事：

1. 将特征 reshape 成 `[B, 2, H]` 后转成 `float32`；
2. 沿最后一维做 L2 归一化，使余弦相似度等于点积；
3. 减去每行最大 logit，降低指数溢出风险；
4. 用 `mask.sum(1).clamp(min=1.0)` 防止正样本计数为零时出现 `0/0`。

这些处理降低了半精度点积、softmax 和空正样本导致的数值风险，但不能保证所有训练一定不出现 NaN；学习率、梯度爆炸、输入和数据问题仍需单独检查。

### 普通 batch 内 SCL 的候选结构

设 `B=2`，两个 dropout 视角给每个 anchor 至少提供一个正样本：同一句子的另一视角。另一个句子的两个视角是否成为正样本，取决于两个标签是否相同。

- 两标签不同：每个 anchor 有 1 个正样本，来自自身另一视角；异类句子的两个视角是负样本；
- 两标签相同：每个 anchor 有 3 个正样本，batch 中没有异类句子可供区分；
- 普通单视角、且 batch 内某个标签只出现一次：该 anchor 没有同类别的其他样本，旧损失可能遇到正样本分母为零。

项目当前的双视角实现用 clamp 把无正样本 anchor 的贡献压成零，避免直接除以零；但它只是在数值上避开失败，并没有保证该 anchor 获得有效的类别对比信号。

### 历史 BERT 实现需要单独看待

[imdb_bert_scl_trainer.py](../imdb_bert_scl_trainer.py) 以 `BATCH_SIZE=2` 调用 `losses.SupConLoss()(pooled_output, labels)`，只传 `[B,H]` 的单视角特征，且 `losses.py` 没有当前实现的 `float32`、归一化和分母 clamp。

`SupConLoss` 的接口约定输入形状是 `[B, n_views, ...]`；把 `[B,H]` 直接传入后，代码中的 `features.view(features.shape[0], features.shape[1], -1)` 会把隐藏维解释为视图维、把每个“视图”解释为 1 维数值。因此该历史脚本不能视为当前 `[B,2,H]` SCL 的等价基线，也不能用它的结果评价当前实现。

## 5. 关键参数与训练流程

| 参数 | R-Drop + SCL 脚本 | 队列版 SCL | 历史 BERT 脚本 |
|---|---:|---:|---:|
| `USE_SCL` 默认 | `False` | 始终启用 | `True` |
| `ALPHA_SCL` | 0.2 | 0.05 | 0.2 |
| `TEMPERATURE` | 0.07 | 0.07 | 0.07 |
| `BATCH_SIZE` | 2 | 2 | 2 |
| `GRAD_ACCUM` | 4 | 4 | 未设置 |
| 视角数 | 2 | 2 | 实际传入 1 组 pooled 特征 |
| 对比候选 | 当前 batch | 当前 batch + 历史队列 | 当前 batch |

当前 [R-Drop + SCL 脚本](../imdb_deberta_v2_unsloth_rdrop_scl.py) 的默认值是 `USE_SCL=False`，源码注释要求做 SCL 对照时再设为 `True`；[队列版](../imdb_sentiment_analusis_torch_/imdb_deberta_v2_unsloth_rdrop_scl_queue.py) 则始终计算 SCL。旧的 PEFT 脚本 [imdb_deberta_unsloth_rdrop_scl.py](../imdb_deberta_unsloth_rdrop_scl.py) 默认 `USE_SCL=True`，这是不同运行阶段的配置差异，不能混写成当前默认。

## 6. 实验现象、失败过程与改进

已有运行说明把较小的 `scl` 值列为 batch 过小或类别分布不佳的排查信号，并记录过 `scl=nan` 与“batch 至少 8”的建议。由于当前材料没有保留对应原始 traceback，本文只将其视为项目记录中的失败现象，不把它写成所有实现都会复现的确定结果。

从代码可以确认的因果风险是：单视角 SCL 的正样本依赖 batch 内同标签的其他样本；二分类小 batch 很容易让某类样本缺席。旧 [losses.py](../losses.py) 在 `mask.sum(1)` 为零时会执行 `0/0`，这是可产生 NaN 的直接实现路径。当前双视角实现在每句运行两次前向后，至少保留了“另一视角”这个正样本，并增加了分母 clamp，因此风险结构已经变化。

改进过程分为三步：

1. 由单视角 batch 内对比改为同一句子的两个 dropout 视角；
2. 将特征与相似度计算放到 `float32`，归一化并稳定 softmax；
3. 对 batch 候选仍不足的情况，在 [监督记忆队列](./04-MoCo监督记忆队列.md) 中进一步加入历史特征。

现有材料没有给出训练预算完全一致的普通 batch SCL 与队列 SCL 消融。已记录的队列实验位列全部模型第 2，但准确率低于普通 LoRA。Kaggle GPU 会话时限迫使它缩短到 384 序列、小 batch 和单 epoch；队列扩大了候选集合，却没有获得与最高分方案相同的更新预算，因此时间限制导致的配置缩水是分数下降的主要原因。

## 7. 效果、局限性及适用条件

SCL 适合有可靠类别标签、且表示需要判别结构的任务。温度越低，模型越关注最相似的候选，容易形成更尖的分布，也更敏感于噪声；温度越高，分布更平滑，信号可能变弱。0.07 是本项目的实验配置，不是通用最优值。

主要局限有：

- 两个 dropout 视角只提供有限的语义增强，正样本多样性低于真正的数据增强；
- 小 batch 中正负样本结构随标签组合剧烈变化；
- batch 全部同类时没有异类负样本，全部异类时每个 anchor 只有自身另一视角作为正样本；
- 当前双视角版本虽已 clamp，但无有效正样本的 anchor 仍会被贡献为零，形成弱监督；
- 对比损失通常增加两次前向、矩阵相似度计算和额外显存，在 1.5B 模型上代价明显；
- SCL 权重 0.2 与队列版 0.05 不可直接互换比较，因为它同时伴随候选集、warmup 和训练阶段的改变。

“小 batch 不适合 SCL”只能限定为当前 1.5B DeBERTa-v2、T4、IMDb 二分类和当前实现下的工程结论，不是跨模型、跨任务的普遍定律。

## 8. 复现与检查清单

- [ ] 确认输入是 `[B,2,H]`，不是把 `[B,H]` 误传给要求多视角的接口。
- [ ] 确认两个视角来自两次训练态前向，且评估时不需要两次前向。
- [ ] 确认特征转 `float32`、沿最后一维归一化、分母有正样本保护。
- [ ] 检查首步 `scl` 是否为有限值，并单独观察 `ce` 是否下降。
- [ ] 明确 SCL、CE、R-Drop 三项各自权重，不把 CE 写进 SCL 公式。
- [ ] 对照实验保持模型、数据、seed、更新预算和 LoRA 配置一致。
- [ ] 普通 batch 实现与 [监督记忆队列](./04-MoCo监督记忆队列.md) 分开报告，不能只改名字后直接比较。
- [ ] 遇到小 batch 异常时，按 [问题复盘](./05-小batch与SCL问题复盘.md) 区分 micro-batch、有效更新样本和对比候选池。
