# 小 batch 与 SCL 问题复盘：从候选不足到监督记忆队列

> 阅读顺序：[R-Drop](./01-RDrop.md) → [SCL](./02-SCL.md) → [Unsloth](./03-Unsloth.md) → [监督记忆队列](./04-MoCo监督记忆队列.md) → [小 batch 复盘](./05-小batch与SCL问题复盘.md)

## 1. 一句话结论

在本项目的 DeBERTa-v2-xxlarge、单张 T4、IMDb 二分类和当时的 SCL 实现条件下，`batch=2 + grad_accum=4` 的有效更新样本量虽然可视为 8，但一次 SCL 前向的 micro-batch 候选池并没有变成 8；项目因此保留小 batch，引入最多 4096 条历史特征的监督记忆队列来扩大候选集。

## 2. 问题背景与目标

初始目标是使用 1.5B 的 DeBERTa-v2-xxlarge，在 IMDb 二分类上同时加入 R-Drop 和 SCL，并通过 LoRA 在单张 T4 上完成训练。R-Drop 约束同一输入两个 dropout 视角的预测一致性，SCL 约束 pooled 特征的类别结构，二者都要求训练时前向两次。

1.5B 主干的参数和激活值决定了 per-device batch 不能随意增加；两次前向又让显存争用更明显。目标于是变成：在不增加 micro-batch 的前提下，让 SCL 仍然获得足够的同标签和异标签候选。

## 3. 方法原理

### 三个容易混淆的量

对单卡、`BATCH_SIZE=2`、`GRAD_ACCUM=4`：

$$
\text{micro-batch}=2,
\qquad
N_{\mathrm{update}}=2\times4=8.
$$

这里 `N_{update}` 表示一次优化器更新对应的有效样本量。

但 SCL 在每次 `forward` 内构造当前候选集。双视角实现的当前候选数是

$$
2\times B=4.
$$

4 个特征进入一次对比损失后，微批损失才参与梯度累积；不同 micro-batch 的特征不会在 SCL 矩阵中汇合。因此：

- micro-batch 决定单次对比候选；
- 梯度累积影响一次优化器更新覆盖多少 micro-batch；
- effective batch 是更新层面的等效样本数，不是对比损失层面的候选集大小。

### 二分类小 batch 的结构问题

两样本的双视角 SCL 有 4 个当前特征：

- 若两句话标签不同，每个 anchor 只有自身另一 dropout 视角作为正样本，异类两句的另两个视角是负样本；
- 若两句话标签相同，每个 anchor 有另外三个正样本，但没有异类负样本；
- 二分类 batch 即使只有两句话，也可能因随机采样或长度排序而全部同类。

项目数据管线没有按标签强制分层，因此同一配置的 batch 类别结构会变化。旧单视角 SCL 更严重：若两条样本异类，某 anchor 没有同标签的其他样本，正样本数为零。

### 梯度累积为什么不能修补候选

梯度累积把 4 次 micro-batch 损失对应的梯度累加，再执行一次优化器更新。它不是把 8 句话拼成一个 `[8,H]` 特征矩阵。因此：

$$
\text{effective batch}=8
\quad\not\Rightarrow\quad
\text{SCL candidate matrix}=8.
$$

这个区别解释了为什么“等效 batch 8”没有自动消除小 batch 的对比结构问题。

## 4. 本项目中的实现

早期文档对应的实现见 [旧 PEFT/R-Drop/SCL 脚本](../imdb_deberta_unsloth_rdrop_scl.py)；当前双视角批内实现见 [R-Drop + SCL 脚本](../imdb_deberta_v2_unsloth_rdrop_scl.py)；改进后的队列实现见 [监督记忆队列脚本](../imdb_sentiment_analusis_torch_/imdb_deberta_v2_unsloth_rdrop_scl_queue.py)。

旧 [losses.py](../losses.py) 的正样本掩码为标签相等，随后去掉自身。若没有同标签样本，`mask.sum(1)` 可能为零，代码直接执行

```python
mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
```

这是 `0/0` 导致 NaN 的直接代码路径。[旧的 BERT trainer 入口](../imdb_bert_scl_trainer.py) 使用 `per_device_train_batch_size=2`，并且只把单视角 `pooled_output` 传给该损失；它既不提供双视角正样本，也没有当前版本的分母 clamp。

当前 DeBERTa-v2 版本改为两句各跑两次前向：

- 每句的另一视角保证至少一个同标签正样本；
- 将 `[B,H]` 堆叠为 `[B,2,H]`；
- 转 `float32` 并归一化；
- 分母 `clamp(min=1.0)`，避免直接除零；
- 无正样本 anchor 的损失贡献为零，而不是 NaN。

这些修正改善了“零正样本直接报 NaN”的结构，但候选仍然只有 4 个，因此并未消除候选不足。队列版又在当前特征之外加入历史队列，见 [MoCo 监督记忆队列说明](./04-MoCo监督记忆队列.md)。

## 5. 关键参数与训练流程

### 初始配置

可核实的初始记录集中在 [Kaggle 完整运行步骤](../Kaggle完整运行步骤.md) 和 [每一步的代码](../每一步的代码.md)：

| 配置 | 初始 PEFT 路径 | 当前 Unsloth 队列路径 |
|---|---:|---:|
| 模型 | `microsoft/deberta-v2-xxlarge` | 同左 |
| `BATCH_SIZE` | 2 | 2 |
| `GRAD_ACCUM` | 4 | 4 |
| `MAX_LEN` | 512 | 384 |
| `ALPHA_RDROP` | 0.6 | 0.6 |
| `ALPHA_SCL` | 0.2 | 0.05 |
| 温度 | 0.07 | 0.07 |
| 队列 | 无 | 4096 |
| SCL warmup | 无 | 800 micro-batch |
| 训练样本 | 25000 | 25000 |
| 评估样本 | 2000 | 2000 |
| LoRA | `r=16`, `alpha=32` | 同左 |

旧路径锁 Transformers 4.46.3，不依赖 Unsloth 接管 v2；当前路径在 Unsloth 与 Transformers 5.16.1 的补丁验证后运行。这里是实现代际差异，不应把版本变化和记忆队列变化混成一个实验因子。

### 配置记录中的冲突

项目文档存在以下不一致：

1. [每一步的代码](../每一步的代码.md) 的正式示例写 `BATCH_SIZE=16`，但对应旧脚本源码和另一份运行文档都写 `BATCH_SIZE=2`；
2. [旧 PEFT 脚本](../imdb_deberta_unsloth_rdrop_scl.py) 默认 `USE_UNSLOTH=False`，而新 [v2 Unsloth 脚本](../imdb_deberta_v2_unsloth_rdrop_scl.py) 要求 Unsloth 接管；
3. [R-Drop + SCL 脚本](../imdb_deberta_v2_unsloth_rdrop_scl.py) 当前默认 `USE_SCL=False`，文件名包含 SCL 但默认路径实际是 R-Drop-only 的 smoke 配置；
4. [R-Drop-only 脚本](../imdb_deberta_v2_unsloth_rdrop.py) 当前 `BENCH=True`，不是直接开始完整训练；
5. `MAX_LEN` 在不同阶段有 128、384、512 三种值，不能把一份日志的耗时或结果直接套到另一份脚本。

写作和复盘时，以对应执行版本源码为准。当前队列正式配置是源码中的 `BATCH_SIZE=2`、`GRAD_ACCUM=4`、`MAX_LEN=384`，不是文档旧示例的 16 或 512。

## 6. 实验现象、失败过程与改进

### 6.1 初始目标

先在 DeBERTa-v2-xxlarge 上叠加 R-Drop 与 SCL，希望 R-Drop 稳定随机预测、SCL 加强类别特征，同时通过 LoRA 将 1.5B 模型放入单张 T4。

### 6.2 初始配置

正式配置使用 `BATCH_SIZE=2`、`GRAD_ACCUM=4`。旧 PEFT 路径采用 `MAX_LEN=512`，其中 SCL 权重为 0.2、R-Drop 权重为 0.6、温度为 0.07。

### 6.3 发现小 batch 的候选风险

[Kaggle 完整运行步骤](../Kaggle完整运行步骤.md) 把 `scl=nan` 归因于 batch 太小或某个类别缺失，并提出“batch 至少 8”；它还记录了一个推断的 `scl` 常用范围 1 到 4。现有材料没有保留当时原始 NaN traceback 或完整日志，因此这两项是项目文字记录，不能被本文升级成“所有运行都已实测复现”。

源码可以确认旧单视角损失在无正样本时会 `0/0`。此外，旧 BERT 训练器把 `[B,H]` 传给多视角接口，会被 reshape 成 `[B,H,1]`。因此历史脚本的 SCL 还不能视为当前 `[B,2,H]` 实现的严谨对照。

### 6.4 分析：有效样本量不是候选数

梯度累积到 4 次后，优化器一步看到相当于 8 个样本的梯度贡献；但每次 SCL 只处理当前 2 句形成的 4 个双视角特征。把 `2×4` 叫成“SCL batch=8”会把更新层面的等效量误写成对比候选池。

### 6.5 约束：T4 与 1.5B 主干

提高 per-device batch 会增加单次前向的激活和相似度矩阵大小，还要承担两次视角的重复前向。单张 T4 的显存无法承受随意增大 batch。项目也没有选择同时复制 1.5B 模型来构造大 batch。

### 6.6 尝试与取舍：保留小 batch，增加历史特征

项目保留 `BATCH_SIZE=2 + GRAD_ACCUM=4`，改为监督记忆队列。队列最多保存 4096 条 fp32 归一化 pooled 特征与标签，不额外复制主干；损失先读取历史候选，再让当前第一视角入队。

### 6.7 改进后的配置

当前队列正式配置为：

```text
BATCH_SIZE = 2
GRAD_ACCUM = 4
QUEUE_SIZE = 4096
ALPHA_SCL = 0.05
TEMPERATURE = 0.07
SCL_WARMUP_STEPS = 800
ALPHA_RDROP = 0.6
MAX_LEN = 384
N_TRAIN = 25000
N_EVAL = 2000
EPOCHS = 1
SEED = 3407
```

SCL 权重在 800 个 micro-batch 内从 0 线性升到 0.05，约等于 200 次优化器更新。队列每 micro-batch 加 2 条特征，约 2048 个 micro-batch 后可达到 4096 条容量。

## 7. 效果、局限性及适用条件

### 三个量的对照

| 方案 | micro-batch | 更新有效样本量 | SCL 候选来源 | 主要风险 |
|---|---:|---:|---|---|
| 普通 batch 内 SCL | 2 | 2 × 梯度累积 | 当前 batch | 正负样本太少、类别缺失、NaN |
| SCL + 监督记忆队列 | 2 | 2 × 梯度累积 | 当前 batch + 历史队列 | 历史特征陈旧、队列偏置 |

当前双视角普通 SCL 每次最多使用 `2B=4` 个当前特征。队列最多再加入 4096 条有效历史特征，因此候选池可从 4 扩展到约 4100 个 key（具体有效数取决于已填入条目）。但队列更大不自动等于更好：相似度计算更重，旧表示也可能与当前模型不一致。

### 可核实的运行记录

工作区的 [checkpoint-3125 训练状态](../checkpoint-3125/trainer_state.json) 有 3125 个优化器 step，和 25000 条训练数据、batch 2、累积 4、1 轮所对应的步数一致。日志从 step 50 的 loss 0.663551 到 step 3100 的 0.559350，期间有起伏；该文件没有评估准确率，checkpoint 标签也不能证明它来自普通 SCL 还是队列 SCL。后续实验记录确认队列版本准确率低于普通 LoRA，主要原因是 Kaggle GPU 运行时限迫使实验缩短到 384 序列、小 batch 和单 epoch，双视角与队列开销又进一步压缩了有效训练预算。因此：

- 能确认的是步数结构和训练损失记录；
- 可以确认的是队列版本准确率低于普通 LoRA；不能确认的是精确最终数值、峰值显存和完整训练耗时；
- 不能把 Unsloth 线性探针的 0.9067 当作最终 IMDb 微调准确率；
- 不能把先前 LoRA 第一趟的 0.5104 归因于 SCL 小 batch，那是另一个尚未解决的训练链路问题；
- 不能把 `5.x 0.656 / 4.46 0.934` 与当前队列训练准确率混为一谈，它们是探针背景记录。

精确最终准确率、各方案公平消融、队列最终占比、峰值显存和完整训练耗时：现有材料未记录。可以确认的因果结论是 Kaggle GPU 会话时限迫使配置缩水，配置缩水又导致模型没有获得与普通 LoRA 同等的 token 覆盖和更新次数；这在当前额度约束下是不可避免的工程折中。

### 适用边界

“小 batch 不适用于当前 SCL 方案”应限定为：

> 本项目在 1.5B DeBERTa-v2-xxlarge、单张 T4、IMDb 二分类、双 dropout 视角和当时 SCL 实现下，batch 2 的候选结构不足以支撑稳定、信息充分的类别对比。

它不是“所有 SCL 必须 batch 至少 8”的普遍定律。模型、任务、损失实现、视角数、显存、类别平衡和采样策略都可能改变阈值。

队列适用于无法提高 micro-batch、但可以接受历史特征偏差的场景。新的小模型或不同任务仍应重新验证：队列长度是否足够、warmup 是否合理、类别分布是否偏置、历史特征是否过时，以及收益是否来自候选扩充而不是其他同时改变的参数。

## 8. 复现与检查清单

- [ ] 明确区分 micro-batch、有效更新样本量、SCL 候选数。
- [ ] 记录实际 `BATCH_SIZE`、`GRAD_ACCUM`、`MAX_LEN` 和脚本版本。
- [ ] 确认双视角 SCL 的候选张量为 `[B,2,H]`，普通批内候选数为 `2B`。
- [ ] 检查每步标签组合，统计全同类、异类混合和缺少类别的 batch 比例。
- [ ] 检查 `ce`、`kl`、`scl` 是否有限；不要把 `clamp` 当成完整稳定性证明。
- [ ] 使用队列时先确认当前特征参与损失，再入队。
- [ ] 记录队列达到 4096 容量的时间，以及队列标签比例。
- [ ] 断点续跑前确认队列和 SCL 预热状态是否会丢失。
- [ ] 公平消融时只改待验证因子；不要同时更换 max length、SCL 权重、版本和队列。
- [ ] 没有日志或指标时写“现有材料未记录”，不补造准确率。
- [ ] 若要继续追查，先阅读 [SCL 实现](./02-SCL.md) 和 [队列实现](./04-MoCo监督记忆队列.md) 的检查项。
