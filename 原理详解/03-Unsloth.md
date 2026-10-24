# Unsloth：DeBERTa-v2 的加载、补丁与 LoRA 接管

> 阅读顺序：[R-Drop](./01-RDrop.md) → [SCL](./02-SCL.md) → [Unsloth](./03-Unsloth.md) → [监督记忆队列](./04-MoCo监督记忆队列.md) → [小 batch 复盘](./05-小batch与SCL问题复盘.md)

## 1. 一句话结论

在本项目中，Unsloth 不是“一个更快的训练库”这么简单；它同时承担自定义模型加载、DeBERTa-v2 补丁接管、半精度显存优化、Unsloth 版 LoRA 封装和运行前验证，只有 `Fast Deberta_V2 patching`、`UNSLOTH OK`、`LoRA: unsloth 封装`、LoRA 参数与分类头均非空同时成立，才算真正打通。

## 2. 问题背景与目标

实验使用 `microsoft/deberta-v2-xxlarge`。项目记录给出的参数量约为 1.5B，开发环境是 Kaggle 单张 NVIDIA Tesla T4 GPU 级别设备，可用显存远小于模型全量微调所需的参数、梯度、优化器状态和激活值总和。

已有材料给出的量级是：T4 可用显存约 14.56 GB，而 1.5B 参数全量微调按每参数 16 字节估算约需 23.35 GB 固定显存，尚未计算激活值。这个估算说明缩小公式中的 batch 并不能解决全部问题；真正需要改变的是主干参数的训练方式。

本项目的目标因此不是追求“最快”，而是让 1.5B DeBERTa-v2 在单卡、小 batch 条件下可加载、可反向传播、可保存 LoRA，并且能通过探针证明这不是普通 Transformers 的静默回退。

## 3. 方法原理

### 显存措施各自解决什么

LoRA（Low-Rank Adaptation）冻结主干的大部分参数，只训练插入注意力投影层的低秩矩阵。原来的全量梯度与优化器状态不再为 1.5B 参数分配，因此显著降低固定开销。它改变的是可训练参数集合。

梯度检查点（gradient checkpointing）反向传播时不保留全部中间激活，而是在需要时重新计算部分前向。它降低的是激活值显存，代价是额外计算时间。

梯度累积（gradient accumulation）把多个 micro-batch 的梯度汇总后再更新一次。它改善小 batch 下的更新稳定性，但不降低每次前向的峰值显存，也不会让 micro-batch 内原有矩阵的尺寸变大。

混合精度使用 fp16 或 bf16 进行大部分计算，降低权重和激活占用。T4 不支持原生 bf16，项目通过环境能力检查优先选择原生 bf16，否则选择 fp16；分类头仍转为 fp32，避免半精度分类计算不稳定。

Unsloth 的补丁与封装在上述措施之外加入针对模型实现和反向传播的优化。它不是本项目中唯一的显存来源，也不能替代 LoRA、梯度检查点或小 batch。项目设置 `load_in_4bit=False`，也没有采用 QLoRA（Quantized LoRA）路线；不能把本实验描述成四位量化训练。

### Transformers 5.x 的双重风险

旧记录表明，普通 Hugging Face Transformers 5.x 的 DeBERTa-v2 实现会导致明显的质量退化：项目文档记录约 0.656，而 4.46.x 约 0.90 以上。因此普通 HF 5.x 路线在 [imdb_deberta_unsloth_rdrop_scl.py](../imdb_deberta_unsloth_rdrop_scl.py) 中被直接拦截。

后续环境记录显示，Unsloth 2026.10.7 配合 Transformers 5.16.1 时，Fast DeBERTa-v2 补丁和冻结特征线性探针可以运行，记录值为 `0.9067`。因此当前脚本只有在 Unsloth 可导入时才允许 DeBERTa-v2 使用 5.x。这个“允许”是条件性的，不是对普通 HF 5.x 的平反。

## 4. 本项目中的实现

主线入口见 [DeBERTa-v2 R-Drop 脚本](../imdb_deberta_v2_unsloth_rdrop.py)、[SCL 脚本](../imdb_deberta_v2_unsloth_rdrop_scl.py) 和 [队列脚本](../imdb_sentiment_analusis_torch_/imdb_deberta_v2_unsloth_rdrop_scl_queue.py)。

### 导入顺序

脚本先设置 `CUDA_VISIBLE_DEVICES=0`，再导入 PyTorch，并在 Transformers 之前导入 Unsloth：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

import unsloth
from unsloth import FastModel
import transformers
```

提前导入 Unsloth 是为了让它在 Transformers 模型类被使用前安装补丁。`CUDA_VISIBLE_DEVICES` 必须在 PyTorch 初始化 CUDA 前设置，否则 Kaggle 上的 `Trainer` 可能启用 `DataParallel`，把 1.5B 模型同时放到两张 T4 上而更容易 OOM。

### 用 FastModel 加载自定义类

模型仍继承 `DebertaV2PreTrainedModel` 并自定义 `forward`，通过 `auto_model` 交给 Unsloth：

```python
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    load_in_4bit=False,
    max_seq_length=MAX_LEN,
    dtype=DTYPE,
    auto_model=DebertaForIMDB,
    num_labels=2,
    attn_implementation="eager",
)
```

DeBERTa-v2 不支持 SDPA（scaled dot-product attention）实现，因此显式使用 eager attention。成功后脚本打印 `UNSLOTH OK: 自定义类加载成功`。

### 用 Unsloth 挂 LoRA

```python
model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    task_type="SEQ_CLS",
    target_modules=["query_proj", "key_proj", "value_proj"],
    modules_to_save=["classifier"],
)
```

DeBERTa-v2 的投影层叫 `query_proj`、`key_proj`、`value_proj`，不是 Llama 常见的 `q_proj`、`k_proj`、`v_proj`。显式指定目标模块后，Unsloth 才会在正确的层上创建 `lora_` 参数。

分类头是随机初始化的新层，名字中不含 `lora_`。如果只按 LoRA 名称筛选可训练参数，它会被 PEFT 冻结。`modules_to_save=["classifier"]` 让分类头进入可训练与保存范围，否则模型可能一直输出随机分类结果。

### 四层接管检查

脚本在训练前执行硬检查：

1. `FastModel.from_pretrained` 返回成功；
2. 日志出现 `UNSLOTH OK: 自定义类加载成功`；
3. `FastModel.get_peft_model` 成功并打印 `LoRA: unsloth 封装`；
4. `_lora_names` 与 `_head_names` 都非空。

对应代码要求：

```python
_lora_names = [n for n, _ in model.named_parameters() if "lora_" in n]
_head_names = [n for n, p in model.named_parameters()
               if "classifier" in n and p.requires_grad]

if USE_LORA and not _lora_names:
    raise RuntimeError("一个 lora_ 参数都没有：LoRA 没挂上，先别训练。")
if not _head_names:
    raise RuntimeError("classifier 不可训练：分类头被冻住了，先别训练。")
```

当前主线设置 `REQUIRE_UNSLOTH=True`。加载或 `get_peft_model` 失败时直接停止，不静默回退到普通 Transformers，避免把另一条实验路线当成 Unsloth 结果。

## 5. 关键参数与训练流程

运行环境和已验证记录来自 [Kaggle Unsloth v2 运行步骤](../Kaggle_Unsloth_v2_运行步骤.md)：

```text
Python 3.13
torch 2.11.0+cu128
transformers 5.16.1
unsloth 2026.10.7
```

| 参数 | 值 | 作用 |
|---|---:|---|
| `load_in_4bit` | `False` | 不把本路线说成四位量化 |
| LoRA `r` | 16 | 低秩适配秩 |
| `lora_alpha` | 32 | LoRA 缩放参数 |
| `lora_dropout` | 0.0 | LoRA 路径本身不加 dropout |
| `target_modules` | `query_proj, key_proj, value_proj` | DeBERTa-v2 注意力投影 |
| `modules_to_save` | `classifier` | 让新分类头可训练、可保存 |
| `MAX_LEN` | smoke 128；正式主脚本按文件而异 | 控制序列显存 |

正式主线的 `BATCH_SIZE=2`、`GRAD_ACCUM=4`、LoRA 学习率 `2e-4`、分类头学习率 `1e-3`。R-Drop-only 脚本当前以 `BENCH=True` 作为源码默认，SCL 脚本当前以 `SMOKE=True` 为默认；正式复现必须按对应文件切换阶段，不能只看文件名。

训练流程是：

1. 运行独立探针，确认 Unsloth 和环境能正常工作；
2. 加载自定义 `DebertaForIMDB`；
3. 挂 Unsloth LoRA 并检查参数；
4. 分类头转 fp32；
5. smoke 测试损失有限、反向传播正常；
6. 切换到 25000 条正式训练；
7. 保存 LoRA adapter 与指标。

## 6. 实验现象、失败过程与改进

### 两个独立探针

[kaggle_deberta_v2_unsloth_probe.py](../kaggle_deberta_v2_unsloth_probe.py) 回答“补丁有没有恢复 v2 特征质量”。它用 Unsloth 加载 `DebertaV2Model`，提取 300 条训练和 300 条测试样本的非 pad 平均池化特征，再训练逻辑回归。通过条件是准确率不低于 0.85；已有运行记录为 `0.9067`，日志还必须出现 `Fast Deberta_V2 patching`。

[unsloth_custom_probe.py](../unsloth_custom_probe.py) 回答“自定义类、LoRA、分类头和反向传播是否全部连通”。它先用 `bert-base-uncased` 降低调试成本，检查 `auto_model` 加载、`FastModel.get_peft_model`、`modules_to_save` 与分类头可训练性，再跑 20 个优化步，要求损失有限且末步低于首步。源码要求看到 `PROBE PASSED`。

两个探针覆盖不同风险：线性探针证明模型特征没有被错误实现破坏，自定义探针证明训练封装与梯度链路存在。只看其中一个不足以放行完整训练。

### 曾出现的“能加载”与“真正接管”混淆

`FastModel.from_pretrained` 成功只说明加载入口返回了模型，不能证明 Fast DeBERTa-v2 补丁已经生效；反过来，普通 Transformers 也可能成功加载权重却得到明显退化。项目因此把补丁日志、LoRA 参数和分类头检查都写成显式条件。

[Kaggle Unsloth v2 运行步骤](../Kaggle_Unsloth_v2_运行步骤.md) 规定：没有 `Fast Deberta_V2 patching`、`UNSLOTH OK` 或 `LoRA: unsloth 封装` 就停止，不进入正式训练。

当前代码的自动拦截主要依赖 Unsloth 是否成功导入。它无法仅凭 `HAS_UNSLOTH=True` 证明补丁日志已经出现，因此“运行前人工检查日志”仍是必要步骤，不能被 import 成功替代。

### Transformers 5.16.1 的兼容补丁

队列脚本额外补回 `transformers.logging` 兼容入口，因为 5.16.1 的相关位置与 Unsloth 编译器预期不同。这个 shim 只修复导入兼容，不改变模型数值质量；它不能替代线性探针。

## 7. 效果、局限性及适用条件

Unsloth 路线在本项目中解决的是“如何加载和训练自定义 1.5B DeBERTa-v2”，而不是单独提高分类精度。已有材料能核实的是环境可导入、补丁可启用、线性探针达到 0.9067、LoRA 路径有硬检查。后续队列实验位列全部模型第 2，但准确率低于普通 LoRA；该次运行受 Kaggle GPU 会话时限影响，被迫缩短序列、压缩训练轮次并使用小 batch。配置缩水是该差距的主要原因，Unsloth 版本差异只能视为次要因素。

主要局限有：

- 对特定 Unsloth、Transformers、Python 和模型架构组合有较强版本依赖；
- 普通 HF 5.x 的质量退化风险不能由“Unsloth 已安装”自动消除；
- 补丁日志、LoRA 参数、分类头状态必须分别验证；
- 分类头 fp32 与半精度主干混用需要显式 dtype 对齐；
- `REQUIRE_UNSLOTH=True` 会阻止静默回退，这有利于归因，但会让兼容性失败直接终止实验；
- LoRA 解决参数优化开销，不保证分类效果；表示质量、池化方式、可训练分类头和数据仍会决定结果。

## 8. 复现与检查清单

- [ ] 在导入 PyTorch 初始化 CUDA 前设置单卡可见性。
- [ ] 在 Transformers 之前导入 Unsloth。
- [ ] 先运行 [v2 线性探针](../kaggle_deberta_v2_unsloth_probe.py)，确认 `Fast Deberta_V2 patching` 且准确率不低于 0.85。
- [ ] 再运行 [自定义类/LoRA 探针](../unsloth_custom_probe.py)，确认 20 步内损失下降。
- [ ] 正式脚本必须打印 `UNSLOTH OK` 与 `LoRA: unsloth 封装`。
- [ ] `LoRA tensors > 0` 且 `head tensors > 0`。
- [ ] 确认目标模块是 DeBERTa-v2 的 `query_proj/key_proj/value_proj`。
- [ ] 确认 `modules_to_save=["classifier"]`，并检查分类头参数为 fp32。
- [ ] 查看首步 CE、KL、SCL 是否有限；有限不等于效果一定好，仍需独立评估。
- [ ] 若使用 SCL，继续阅读 [SCL](./02-SCL.md) 与 [监督记忆队列](./04-MoCo监督记忆队列.md)。
