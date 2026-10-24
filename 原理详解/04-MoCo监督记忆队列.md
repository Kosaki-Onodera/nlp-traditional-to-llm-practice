# MoCo 思想监督记忆队列：在小显存下扩大 SCL 候选集

> 阅读顺序：[R-Drop](./01-RDrop.md) → [SCL](./02-SCL.md) → [Unsloth](./03-Unsloth.md) → [监督记忆队列](./04-MoCo监督记忆队列.md) → [小 batch 复盘](./05-小batch与SCL问题复盘.md)

## 1. 一句话结论

本项目只借鉴 MoCo（Momentum Contrast）用队列扩大对比候选的思想，实现的是“当前 micro-batch + 4096 条历史 pooled 特征”的监督记忆队列；它没有动量 key encoder，没有动量更新队列特征，也不是完整 MoCo。

## 2. 问题背景与目标

普通 batch 内 SCL 的候选集只有当前 micro-batch。对于 1.5B DeBERTa-v2，增加 `per_device_train_batch_size` 会显著提高前向激活和两次视角的显存压力；T4 上无法通过简单增大 batch 获得丰富候选。

梯度累积可以把多个 micro-batch 的更新贡献合并，却不能把不同 micro-batch 的特征放进同一次 SCL 相似度矩阵。实验因此需要一个独立于当前 batch 大小、又不用复制 1.5B 主干的候选池。

记忆队列只保存句向量和标签。它借用 MoCo 的“用历史 key 扩充候选”思想，同时保留 SCL 的监督正负样本定义。

## 3. 方法原理

### MoCo 原始结构

MoCo 的典型组成是：

- 一个在线 query encoder，产生当前 query 特征；
- 一个动量 key encoder，产生 key 特征；
- 一个 FIFO（first in, first out）候选队列；
- key encoder 参数按动量更新：

$$
\theta_k\leftarrow m\theta_k+(1-m)\theta_q.
$$

detached 的 key 特征进入队列，为当前 query 提供远多于 local batch 的历史候选。标准 MoCo 主要是无标签的实例对比，队列中的 key 作为负样本参与 InfoNCE 类损失。

本项目借用了“历史特征队列扩大候选集”，没有实现动量 key encoder。历史特征由当前在线模型在较早时刻产生，因此会直接承受参数变化和表示漂移。

### 本项目的监督队列损失

设当前批次为 `B`，两个 dropout 视角产生 `2B` 个当前特征 `Z_c`，有效历史特征为 `Z_q`。候选集合为

$$
Z=[Z_c;Z_q].
$$

对 anchor `i`：

$$
\mathcal L_i
=-\frac{1}{|\mathcal P(i)|}
\sum_{p\in\mathcal P(i)}
\log
\frac{\exp(z_i^\top z_p/\tau)}
{\sum_{a}\exp(z_i^\top z_a/\tau)}.
$$

其中 0.07 为温度。候选 `p` 属于正样本，当且仅当它与 anchor 标签相同，并且 `p` 不是 anchor 自己在当前特征中的同一项。历史队列中没有原样本、文本或 `input_ids`，只有 pooled 特征和标签。

## 4. 本项目中的实现

完整实现位于 [imdb_deberta_v2_unsloth_rdrop_scl_queue.py](../imdb_sentiment_analusis_torch_/imdb_deberta_v2_unsloth_rdrop_scl_queue.py)，普通 batch 对照位于 [imdb_deberta_v2_unsloth_rdrop_scl.py](../imdb_deberta_v2_unsloth_rdrop_scl.py)。

### 数据结构

模型初始化时注册三个 buffer：

```python
self.register_buffer(
    "queue_feats",
    torch.zeros(QUEUE_SIZE, config.hidden_size, dtype=torch.float32),
    persistent=False,
)
self.register_buffer(
    "queue_labels",
    torch.full((QUEUE_SIZE,), -1, dtype=torch.long),
    persistent=False,
)
self.register_buffer(
    "queue_ptr",
    torch.zeros(1, dtype=torch.long),
    persistent=False,
)
```

对 DeBERTa-v2-xxlarge，`hidden_size=1536`，所以特征 buffer 形状是 `[4096, 1536]`。标签 shape 是 `[4096]`，空位置标为 `-1`。当前实现的 buffer 本体约 `4096 × 1536 × 4 B = 25,165,824 B`，即 24 MiB；这不代表额外复制了主干权重。

### FIFO 更新

`_enqueue` 在 `torch.no_grad()` 下执行，先归一化特征，再按 `queue_ptr` 分段写入。写满后环形覆盖最旧位置，并令指针回绕，形成 FIFO。一个 micro-batch 有 2 个样本时写入 2 条；每批只写第一个 dropout 视角的 pooled 特征。

### 进入损失与入队的顺序

`forward` 中先计算 SCL，再执行：

```python
scl_part = self._queue_scl_loss(pooled1, pooled2, labels)
total = cut_loss + scl_weight * scl_part
self._enqueue(pooled1.detach(), labels)
self._scl_steps += 1
```

顺序非常关键：当前特征先参与损失，再进入历史队列。这样本步特征不会在本步被当作“历史正样本”。当前特征进入损失时仍保留计算图，梯度可以回传到主干；历史队列在入队时已 detach，只作为常量 key，不参与反向传播。

### 自身排除的精确边界

`pos_mask` 先按标签相等建立，再用单位矩阵把 `[当前 2B 个特征, 当前 2B 个特征]` 的对角位置从正样本中排除。同一句子的另一个 dropout 视角仍保留为正样本。

当前代码在 `torch.logsumexp(logits, ...)` 做归一化时仍包含所有 key，因此自身只从“正样本 mask”排除，并未从分母候选中显式删除。这是当前实现与普通 `SupConLoss` 中 `exp_logits * logits_mask` 写法的一个差异，阅读损失数值时必须注意。

### 监督正负样本

- 当前两句的四个视角都在 key 集合中；
- 同标签、非自身的当前或历史特征为正样本；
- 历史队列标签为 `-1` 的空槽不参与；
- 若某 anchor 没有正样本，代码将该 anchor 从均值中排除；
- 历史特征只保存 pooled 表示，不能重新跑 key encoder，也不保存文本和原始样本。

## 5. 关键参数与训练流程

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `QUEUE_SIZE` | 4096 | 最多保留的历史特征条目数 |
| `ALPHA_SCL` | 0.05 | SCL 最终权重上限 |
| `TEMPERATURE` | 0.07 | 对比相似度温度 |
| `SCL_WARMUP_STEPS` | 800 | 按 micro-batch 计数的线性预热 |
| `BATCH_SIZE` | 2 | 每设备 micro-batch |
| `GRAD_ACCUM` | 4 | 每 4 个 micro-batch 更新一次 |
| `ALPHA_RDROP` | 0.6 | R-Drop KL 权重 |
| `MAX_LEN` | 384 | 当前队列正式配置 |
| `N_TRAIN` | 25000 | 正式训练样本数 |

有效 SCL 权重为

$$
\alpha_{\mathrm{SCL}}^{\mathrm{eff}}
=\alpha_{\mathrm{SCL}}
\min\left(1,\frac{t_{\mathrm{micro}}}
{\max(1,800)}\right),
$$

所以首个 micro-batch 的 SCL 权重为 0，第 800 个 micro-batch 起达到 0.05。在 `BATCH_SIZE=2` 时，800 个 micro-batch 约为 200 次优化器更新。队列在每次 micro-batch 加入 2 条特征，若从空队列开始且不中断，约 2048 个 micro-batch、512 次优化器更新后达到 4096 条容量；而 SCL 权重在此之前已经达到 0.05。

训练顺序是：提取两个视角，计算 CE、R-Drop KL 和队列 SCL，按 warmup 加权，组合损失，反向传播，然后才把当前第一视角入队。SCL 与 CE、R-Drop 是并列损失，不是互相替代。

## 6. 实验现象、失败过程与改进

使用队列的直接动机来自 [小 batch 与 SCL 复盘](./05-小batch与SCL问题复盘.md)：`BATCH_SIZE=2` 加 `GRAD_ACCUM=4` 只能提供 8 个样本的有效更新量，但当前 micro-batch 的对比候选仍只有 2 句话的两个视角。队列最多再加入 4096 条历史特征，使 SCL 不必通过提高每步激活显存来扩大候选。

队列不是为了让模型“额外训练更多参数”；它只把表示历史用于损失。当前材料中，R-Drop-only 脚本和队列脚本共享大部分模型与优化配置，但队列版还改变了 `ALPHA_SCL`、加入 warmup，并使用 `MAX_LEN=384`。因此即使最终指标不同，也不能只归因于队列。

工作区可以核实的 [checkpoint-3125](../checkpoint-3125/trainer_state.json) 记录显示 `global_step=3125`，恰好等于 `25000 / (2×4)`，说明保存记录与“25000 条、batch 2、累积 4、一轮”的步数结构一致。日志中 step 50 的 loss 为 0.663551，step 3100 为 0.559350；记录没有 `eval_accuracy`，也没有足够信息证明该 checkpoint 一定来自队列脚本。结合后续实验记录，队列版本准确率低于普通 LoRA。主要原因是 Kaggle GPU 单次会话时限无法同时承担更长序列、更多更新和双视角/队列开销，最终只能使用 384 截断、小 batch 与单 epoch；该结果不能证明队列提升了准确率。

## 7. 效果、局限性及适用条件

### 普通 batch 内 SCL 与监督记忆队列

| 维度 | 普通 batch 内 SCL | SCL + 监督记忆队列 |
|---|---|---|
| 候选集来源 | 当前 micro-batch 的两视角特征 | 当前 micro-batch + 历史 pooled 特征 |
| 最大历史候选 | 0 | 4096 条 |
| 主干复制 | 无 | 无 |
| 额外显存 | 当前两视角相似度矩阵 | 当前两视角 + 历史特征与相似度矩阵 |
| 实现复杂度 | 较低 | 多出 FIFO、标签、预热和续跑状态 |
| 适用条件 | batch 内类别结构足够 | local batch 很小，但能接受历史特征偏差 |

队列的成本不是只有 24 MiB 特征缓存。每次 SCL 仍要计算当前特征与所有有效历史 key 的相似度，队列越大，矩阵计算和反向传播图越重；保留队列 buffer 本身也不等于保留 1.5B key encoder。

主要局限有：

- 历史特征可能陈旧，在线模型更新后语义漂移；
- FIFO 标签分布可能不均衡，最近到达的类别会占更大比例；
- 队列为空时要依靠当前两视角，历史候选不能立即提供充分监督；
- 当前代码没有动量 key encoder，不能把历史 key 当作稳定且缓慢变化的表示；
- 队列设为 `persistent=False`，模型 checkpoint 不会保存队列；
- `_scl_steps` 是普通 Python 属性，也未显式保存；
- 断点续跑后队列与 warmup 计数会重新开始，除非专门补充状态保存；
- 当前自身仅从正样本 mask 排除，分母仍包含自身，与一些标准 SupCon 实现不完全一致。

因此队列适合“候选不足是主要瓶颈，且离线/单次训练可接受”的场景。换模型、任务或更长训练时仍需重新验证历史特征质量与类别分布。

## 8. 复现与检查清单

- [ ] 确认使用的是 [队列脚本](../imdb_sentiment_analusis_torch_/imdb_deberta_v2_unsloth_rdrop_scl_queue.py)，而不是普通 batch 版本。
- [ ] 确认 `QUEUE_SIZE=4096`、`ALPHA_SCL=0.05`、温度 0.07、warmup 800。
- [ ] 确认当前损失先算、当前特征后入队。
- [ ] 确认历史特征 detach 且以 fp32 保存、归一化后进入相似度矩阵。
- [ ] 检查空槽标签为 `-1`，并确认无效历史不参与候选。
- [ ] 记录队列达到容量的 micro-batch 和优化器 step。
- [ ] 若断点续跑，明确队列与 warmup 会重置，必要时自行持久化这两类状态。
- [ ] 对比普通 SCL 与队列 SCL 时，记录所有不同的超参数，不把多变量变化归因于队列单项。
- [ ] 不把本实现称为“完整 MoCo”或“动量队列”。
