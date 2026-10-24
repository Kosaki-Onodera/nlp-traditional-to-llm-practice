# ==========================================================
# DeBERTa-v2-xxlarge + Unsloth + 自定义类 + LoRA + R-Drop + SCL
# SCL 使用 supervised memory queue（MoCo 思想的小显存版本）：
# 队列只存 pooled 特征和标签，不存原始样本，也不额外复制 1.5B 主干。
#
# Kaggle 用法：
#   Cell 1:  !pip install -q unsloth
#   Cell 2:  Restart & Clear Cell Outputs
#   Cell 3:  把本文件全部内容粘进去
#
# 2026-10-21 实测：Unsloth 2026.10.7 + transformers 5.16.1，
# Fast Deberta_V2 patch 正常，v2 线性探针 0.9067。
# 必须先确认日志里出现 Fast Deberta_V2 patching。
#
# 当前默认：BENCH=False，正式训练（25000 条，完整 1 轮，max_len=384）。
# 如需再测速，把 BENCH 改回 True，只跑 20 步。
# ==========================================================

import os
import time
import math
import json

# 必须在 import torch 之前：Kaggle 给 2 张 T4，不限定的话 Trainer 会用
# DataParallel，把 1.5B 模型在两张卡上各放一份，反而更容易 OOM。
# 不能写成 setdefault：Kaggle 默认已经设了 CUDA_VISIBLE_DEVICES=0,1，
# setdefault 不会覆盖，torch 仍然会看到两张卡并启用 DataParallel。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import unsloth                          # 必须早于 transformers
    from unsloth import FastModel
    HAS_UNSLOTH = True
except Exception as _unsloth_err:
    unsloth = None
    FastModel = None
    HAS_UNSLOTH = False
    print(">>> unsloth 不可用，将使用 PEFT 路径：", repr(_unsloth_err))

import transformers

# transformers 5.16.1 移除了顶层 logging，但 Unsloth 的编译器仍会
# from transformers import logging。这里补回兼容入口。
try:
    import transformers.utils.logging as _tf_logging
    transformers.logging = _tf_logging
    import sys as _sys
    _sys.modules.setdefault("transformers.logging", _tf_logging)
except Exception as _log_err:
    print(">>> transformers.logging shim failed:", repr(_log_err))

from transformers import (
    AutoTokenizer,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
try:
    from transformers import SequenceClassifierOutput
except ImportError:
    try:
        from transformers.modeling_outputs import SequenceClassifierOutput
    except ImportError:
        # transformers 5.x 的兜底：只用到 loss / logits 两个字段。
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class SequenceClassifierOutput:
            loss: Optional[torch.FloatTensor] = None
            logits: Optional[torch.FloatTensor] = None
            hidden_states: Optional[tuple] = None
            attentions: Optional[tuple] = None

from datasets import load_dataset
from sklearn.metrics import accuracy_score
import pandas as pd

# ---------------- 配置：Unsloth 对照线 ----------------
# BENCH=True：20 步基准；BENCH=False + SMOKE=True：64 条 smoke；
# BENCH=False + SMOKE=False：正式训练。
BENCH = False
SMOKE = False

MODEL_NAME = "microsoft/deberta-v2-xxlarge"
N_TRAIN    = 512 if BENCH else (64 if SMOKE else 25000)
N_EVAL     = 64 if (BENCH or SMOKE) else 2000
MAX_LEN    = 384 if BENCH else (128 if SMOKE else 384)
OUT_DIR    = "./bench_queue384" if BENCH else (
    "./out_v2_queue_smoke" if SMOKE else "./out_v2_queue"
)
RESUME     = False if (BENCH or SMOKE) else True

USE_UNSLOTH = True
REQUIRE_UNSLOTH = True
USE_LORA    = True

ALPHA_RDROP = 0.6
ALPHA_SCL   = 0.05
TEMPERATURE = 0.07
QUEUE_SIZE  = 4096
SCL_WARMUP_STEPS = 800   # 按 micro-batch 计；约 200 个 optimizer steps
FULL_N_TRAIN = 25000

LORA_R     = 16
LORA_ALPHA = 32

BATCH_SIZE = 2
GRAD_ACCUM = 4
LR         = 2e-4
HEAD_LR    = 1e-3
EPOCHS     = 1
# 基准 20 步；smoke 和正式跑都按 epoch 走。
MAX_STEPS  = 20 if BENCH else -1
SEED       = 3407

torch.manual_seed(SEED)

def _bf16_supported():
    if not torch.cuda.is_available():
        return False
    try:
        # T4 没有原生 bf16；新版 torch 的 including_emulation=True 会返回 True，
        # 但那是软件模拟，既慢又没必要。这里只认原生 bf16。
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8


if torch.cuda.is_available():
    DTYPE = torch.bfloat16 if _bf16_supported() else torch.float16
else:
    DTYPE = torch.float32

print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("unsloth     :", getattr(unsloth, "__version__", "?"))
print("cuda        :", torch.cuda.is_available(), "| dtype:", DTYPE)
print("gpu count   :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name    :", torch.cuda.get_device_name(0))

# 普通 transformers 5.x 的 DeBERTa-v2 实现会掉点；
# 但 2026-10-21 实测 Unsloth 2026.10.7 的 Fast Deberta_V2 patch 正常：
# UNSLOTH V2 LINEAR PROBE = 0.9067。
# 因此只有“Unsloth 真正接管”时才允许 transformers 5.x。
_TF_MAJOR = int(transformers.__version__.split(".")[0])
ALLOW_TF5_WITH_UNSLOTH = True
_tf5_blocked = (
    MODEL_NAME.startswith("microsoft/deberta-v2")
    and _TF_MAJOR >= 5
    and not (USE_UNSLOTH and HAS_UNSLOTH and ALLOW_TF5_WITH_UNSLOTH)
)
if _tf5_blocked:
    raise SystemExit(
        "停：普通 transformers %s 的 DeBERTa-v2 实现会掉点（5.0 -> 0.656，"
        "4.46 -> 0.934）。这条线必须由 Unsloth 接管；请先确认 "
        "import unsloth 成功，并且 UNSLOTH V2 PROBE 通过。" % transformers.__version__
    )
if _TF_MAJOR >= 5 and USE_UNSLOTH and HAS_UNSLOTH:
    print(">>> transformers 5.x + Unsloth Deberta_V2 patch: allowed (probe 0.9067)")


# ---------------- R-Drop 损失 ----------------
def symmetric_kl(a, b):
    """R-Drop 的双向 KL：0.5 * (KL(a||b) + KL(b||a))"""
    la = F.log_softmax(a.float(), dim=-1)
    lb = F.log_softmax(b.float(), dim=-1)
    # 必须是 batchmean，不能是 sum：sum 会把整个 batch 的 KL 加在一起，
    # batch=16 时相当于把 ALPHA_RDROP 放大了 16 倍。
    return 0.5 * (
        F.kl_div(la, lb, reduction="batchmean", log_target=True)
        + F.kl_div(lb, la, reduction="batchmean", log_target=True)
    )


# ---------------- 自定义模型 ----------------
class DebertaForIMDB(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
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
        self._scl_steps = 0
        self.post_init()

    def _encode(self, input_ids, attention_mask):
        out = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # 用非 pad 位置的平均池化，绕开 DeBERTa-v2 自带的随机 ContextPooler。
        # 你自己的线性探针：CLS 0.610 / 平均池化 0.676；v3 实测也是平均池化更好。
        h = out.last_hidden_state
        if attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        # 主干可能是 fp16/bf16，分类头固定 fp32；这里显式对齐 dtype。
        return pooled, self.classifier(pooled.float())

    def _queue_scl_loss(self, pooled1, pooled2, labels):
        """当前两个视角 + 历史队列特征的监督对比损失。"""
        device = pooled1.device
        if self.queue_feats.device != device:
            self.queue_feats = self.queue_feats.to(device)
            self.queue_labels = self.queue_labels.to(device)
            self.queue_ptr = self.queue_ptr.to(device)

        feats = torch.cat([pooled1, pooled2], dim=0).float()
        feats = F.normalize(feats, dim=-1)
        labels = labels.to(device)
        cur_labels = torch.cat([labels, labels], dim=0)

        valid = self.queue_labels >= 0
        if valid.any():
            q_feats = self.queue_feats[valid].float()
            q_labels = self.queue_labels[valid]
            keys = torch.cat([feats, q_feats], dim=0)
            key_labels = torch.cat([cur_labels, q_labels], dim=0)
        else:
            keys = feats
            key_labels = cur_labels

        logits = torch.matmul(feats, keys.T) / TEMPERATURE
        pos_mask = key_labels.unsqueeze(0).eq(cur_labels.unsqueeze(1))

        # 去掉 anchor 自己的当前特征；另一个视角仍然保留为正样本。
        n_cur = feats.size(0)
        self_mask = torch.zeros_like(pos_mask)
        self_mask[:, :n_cur] = torch.eye(
            n_cur, dtype=torch.bool, device=feats.device
        )
        pos_mask = pos_mask & ~self_mask

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pos_count = pos_mask.sum(dim=1)
        valid_anchor = pos_count > 0
        if not valid_anchor.any():
            return feats.new_zeros(())

        loss = -(log_prob * pos_mask).sum(dim=1)[valid_anchor]
        loss = loss / pos_count[valid_anchor]
        return loss.mean()

    @torch.no_grad()
    def _enqueue(self, feats, labels):
        """FIFO 入队；只存 detached 的归一化特征和标签。"""
        feats = F.normalize(feats.float(), dim=-1)
        n = feats.size(0)
        if n >= QUEUE_SIZE:
            self.queue_feats.copy_(feats[-QUEUE_SIZE:])
            self.queue_labels.copy_(labels[-QUEUE_SIZE:])
            self.queue_ptr[0] = 0
            return

        ptr = int(self.queue_ptr.item())
        end = ptr + n
        if end <= QUEUE_SIZE:
            self.queue_feats[ptr:end].copy_(feats)
            self.queue_labels[ptr:end].copy_(labels)
        else:
            first = QUEUE_SIZE - ptr
            self.queue_feats[ptr:].copy_(feats[:first])
            self.queue_labels[ptr:].copy_(labels[:first])
            rest = n - first
            self.queue_feats[:rest].copy_(feats[first:])
            self.queue_labels[:rest].copy_(labels[first:])
        self.queue_ptr[0] = (ptr + n) % QUEUE_SIZE
        if end >= QUEUE_SIZE and not getattr(self, "_queue_full_logged", False):
            print(">>> SCL memory queue reached capacity:", QUEUE_SIZE)
            self._queue_full_logged = True

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, labels=None, **kwargs):
        # 训练时同一输入跑两次，靠 dropout 形成两个视角；
        # 评估时没有 labels，只跑一次。
        pooled1, logits1 = self._encode(input_ids, attention_mask)
        if labels is not None:
            pooled2, logits2 = self._encode(input_ids, attention_mask)
        else:
            pooled2, logits2 = None, None

        loss = None
        if labels is not None:
            # fp16 的 logits 直接进 cross_entropy 容易下溢；先转 float32。
            ce1 = F.cross_entropy(logits1.float(), labels)
            ce2 = F.cross_entropy(logits2.float(), labels)
            kl_part = symmetric_kl(logits1, logits2)
            scl_part = self._queue_scl_loss(pooled1, pooled2, labels)
            scl_weight = ALPHA_SCL * min(
                1.0, self._scl_steps / max(1, SCL_WARMUP_STEPS)
            )
            total = (
                0.5 * (ce1 + ce2)
                + ALPHA_RDROP * kl_part
                + scl_weight * scl_part
            )

            # 先算 loss，再入队，避免当前特征被自己当成历史正样本。
            self._enqueue(pooled1.detach(), labels)
            self._scl_steps += 1

            # 只在第一步打印一次，看各项损失的比例是否合理
            if not getattr(self, "_loss_logged", False):
                print(">>> loss parts | ce=%.4f kl=%.4f scl=%.4f scl_w=%.3f | total=%.4f" % (
                    float(ce1.detach()), float(kl_part.detach()),
                    float(scl_part.detach()), float(scl_weight),
                    float(total.detach())))
                self._loss_logged = True

            loss = total

        return SequenceClassifierOutput(loss=loss, logits=logits1)


# ---------------- 加载模型 ----------------
def load_model():
    if USE_UNSLOTH and HAS_UNSLOTH:
        try:
            model, tokenizer = FastModel.from_pretrained(
                model_name=MODEL_NAME,
                load_in_4bit=False,
                max_seq_length=MAX_LEN,
                dtype=DTYPE,
                auto_model=DebertaForIMDB,
                num_labels=2,
                attn_implementation="eager",        # DeBERTa v2 does not support SDPA
                unsloth_force_compile=False,
            )
            print(">>> UNSLOTH OK: 自定义类加载成功")
            print(">>> model class:", type(model).__module__, type(model).__name__)
            return model, tokenizer, True
        except Exception as e:
            print("=" * 70)
            print(">>> UNSLOTH 加载失败：", repr(e))
            if REQUIRE_UNSLOTH:
                raise RuntimeError(
                    "Unsloth 加载 v2 失败；这条线要求必须由 Unsloth 接管，"
                    "因此不回退、不继续加载普通 transformers。"
                ) from e
            print(">>> 回退到标准 transformers（unsloth 优势失效）")
            print("=" * 70)

    model = DebertaForIMDB.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer, False


if REQUIRE_UNSLOTH and not HAS_UNSLOTH:
    raise RuntimeError(
        "这条 Unsloth 对照线要求先安装与 transformers 4.46.3 兼容的 Unsloth。"
        "当前环境没有可用的 Unsloth，已在加载 v2 之前停止，避免浪费 GPU。"
    )

model, tokenizer, USED_UNSLOTH = load_model()

if USE_UNSLOTH and REQUIRE_UNSLOTH and not USED_UNSLOTH:
    raise RuntimeError(
        "unsloth 没有真正接管这个模型。不要继续训练：先看上面的报错，"
        "换 bert-base-uncased / deberta-v3-xsmall 验证链路，"
        "或者把 REQUIRE_UNSLOTH 改成 False 明确走回退路径。"
    )


# ---------------- 挂 LoRA ----------------
if USE_LORA:
    lora_ok = False

    if USED_UNSLOTH and FastModel is not None:
        try:
            model = FastModel.get_peft_model(
                model,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=SEED,
                use_rslora=False,
                loftq_config=None,
                task_type="SEQ_CLS",
                # 不写这一行，unsloth 会按 Llama 的 q_proj/k_proj/... 去找，
                # DeBERTa 上要么报找不到模块，要么一个 LoRA 都没挂上。
                target_modules=["query_proj", "key_proj", "value_proj"],
                # 分类头是随机初始化的，必须让它跟着训练，否则永远是随机输出。
                modules_to_save=["classifier"],
            )
            print(">>> LoRA: unsloth 封装")
            lora_ok = True
        except Exception as e:
            print("=" * 70)
            print(">>> unsloth get_peft_model 失败：", repr(e))
            print(">>> 回退到 PEFT 手写 LoRA")
            print("=" * 70)

    if not lora_ok:
        if REQUIRE_UNSLOTH:
            raise RuntimeError(
                "Unsloth 能加载模型，但 FastModel.get_peft_model 没成功。"
                "这条线要求 LoRA 也必须由 Unsloth 挂载，因此拒绝回退到 PEFT。"
            )
        from peft import LoraConfig, get_peft_model, TaskType
        model = get_peft_model(model, LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=["query_proj", "key_proj", "value_proj"],
            modules_to_save=["classifier"],
        ))
        print(">>> LoRA: PEFT 手写")
        # 回退路径没有 unsloth 的 checkpointing，xxlarge 上必须自己开。
        try:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            print(">>> 回退路径：已开梯度检查点")
        except Exception as e:
            print(">>> 回退路径：梯度检查点没开成：", repr(e))

# 分类头统一 fp32：fp16 主干 + fp32 分类头会在 F.linear 里报 dtype 不一致。
# PEFT 的 modules_to_save 会同时有 original_module 和副本，两个都转。
for _n, _p in model.named_parameters():
    if "classifier" in _n:
        _p.data = _p.data.float()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(">>> trainable params: {:,} / {:,} = {:.2%}".format(
    trainable, total, trainable / total))

# 两道硬检查：LoRA 真的挂上了，分类头真的可训练。
_lora_names = [n for n, _ in model.named_parameters() if "lora_" in n]
_head_names = [n for n, p in model.named_parameters()
               if "classifier" in n and p.requires_grad]
print(">>> LoRA tensors :", len(_lora_names))
print(">>> head tensors :", len(_head_names))
if USE_LORA and not _lora_names:
    raise RuntimeError("一个 lora_ 参数都没有：LoRA 没挂上，先别训练。")
if not _head_names:
    raise RuntimeError("classifier 不可训练：分类头被冻住了，先别训练。")


# ---------------- 数据 ----------------
raw = load_dataset("stanfordnlp/imdb")
train_raw = raw["train"].shuffle(seed=SEED)
eval_raw  = raw["test"].shuffle(seed=SEED)

train_raw = train_raw.select(range(min(N_TRAIN, len(train_raw))))
eval_raw  = eval_raw.select(range(min(N_EVAL, len(eval_raw))))

train_raw = train_raw.rename_column("label", "labels")
eval_raw  = eval_raw.rename_column("label", "labels")


def tokenize_fn(examples):
    out = tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)
    # 给 group_by_length 用：把长度接近的样本排到同一批，减少 pad。
    out["length"] = [len(ids) for ids in out["input_ids"]]
    return out


train_ds = train_raw.map(tokenize_fn, batched=True, remove_columns=["text"])
eval_ds  = eval_raw.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


class LengthGroupedSampler(torch.utils.data.Sampler):
    """把长度接近的样本排进同一个 micro-batch，减少 padding。

    transformers 5.16 已删除 TrainingArguments(group_by_length=...)，
    所以这里直接给 Trainer 一个自定义 sampler。
    """

    def __init__(self, lengths, batch_size, seed=3407):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        # 加一点随机扰动，避免完全按长度排序；同类长度仍会排在一起。
        noisy = self.lengths + rng.uniform(0.0, 8.0, size=len(self.lengths))
        order = np.argsort(noisy, kind="stable")
        batches = [
            order[i:i + self.batch_size]
            for i in range(0, len(order), self.batch_size)
        ]
        rng.shuffle(batches)
        self.epoch += 1
        for batch in batches:
            yield from batch.tolist()

    def __len__(self):
        return len(self.lengths)


# ---------------- 训练 ----------------
class SafeTrainer(Trainer):
    """绕过 unsloth 对 prediction_step 的 monkey-patch"""

    def _get_train_sampler(self, *args, **kwargs):
        return LengthGroupedSampler(
            train_ds["length"],
            batch_size=self.args.train_batch_size,
            seed=SEED,
        )

    def create_optimizer(self):
        """LoRA 用 LR，随机初始化的 classifier 用更大的 HEAD_LR。"""
        if self.optimizer is not None:
            return self.optimizer

        lora_params, head_params, other_params = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                head_params.append(param)
            elif "lora_" in name:
                lora_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if lora_params:
            groups.append({"params": lora_params, "lr": self.args.learning_rate})
        if head_params:
            groups.append({"params": head_params, "lr": HEAD_LR})
        if other_params:
            groups.append({"params": other_params, "lr": self.args.learning_rate})
        if not groups:
            return super().create_optimizer()

        print(">>> optimizer groups | lora=%d head=%d other=%d | lr=%.1e/%.1e" % (
            len(lora_params), len(head_params), len(other_params),
            self.args.learning_rate, HEAD_LR))
        try:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args, self.model
            )
        except AttributeError:
            from transformers.optimization import AdamW
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "lr": self.args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            }
        # 每个 group 自带的 lr 会覆盖 optimizer_kwargs 里的默认 lr。
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None, **kwargs):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        with torch.no_grad():
            outputs = model(**inputs)
        loss = getattr(outputs, "loss", None)
        logits = getattr(outputs, "logits", None)
        # unsloth 的梯度检查点会在内部重新打开梯度，
        # 所以即使外面套了 no_grad，返回的张量仍可能挂着计算图。
        # 不 detach，Trainer 收集结果转 numpy 时会报错。
        if loss is not None:
            loss = loss.detach()
        if logits is not None:
            logits = logits.detach()
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    per_device_eval_batch_size=32,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    bf16=(DTYPE == torch.bfloat16),
    fp16=(DTYPE == torch.float16),
    report_to="none",
    label_names=["labels"],
    dataloader_drop_last=True,      # 每个 optimizer batch 大小一致
    seed=SEED,
)

trainer = SafeTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

# ---------------- 评估（不走 Trainer.predict） ----------------
def manual_eval(model, dataset, batch_size=32):
    """
    自己分批算 logits，原因有两层：
      1) dataloader_drop_last=True 会作用到评估集上。评估集比 batch 小的话，
         唯一的一批会被整个丢掉，predictions 直接变成 None。
      2) 评估时不传 labels，模型只跑一次前向，比训练时快一倍。
    """
    model.eval()
    all_preds = []
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            chunk = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            batch = data_collator(chunk)
            dev = next(model.parameters()).device
            batch = {k: v.to(dev) for k, v in batch.items() if k != "labels"}
            out = model(**batch)
            logits = getattr(out, "logits", None)
            if logits is None:
                logits = out[0]
            all_preds.append(logits.argmax(-1).detach().cpu())
            all_logits.append(logits.float().detach().cpu())
    model.train()
    return torch.cat(all_preds).numpy(), torch.cat(all_logits).numpy()


# ---------------- 断点续跑 ----------------
import glob
_ckpts = sorted(glob.glob(os.path.join(OUT_DIR, "checkpoint-*")),
                key=lambda p: int(p.rsplit("-", 1)[-1]))
last_ckpt = _ckpts[-1] if _ckpts else None
if last_ckpt:
    print(">>> 发现断点，从", last_ckpt, "继续训练")


# ---------------- 训练 ----------------
t0 = time.time()
trainer.train(resume_from_checkpoint=last_ckpt if RESUME else None)
dt = time.time() - t0

n_steps = max(trainer.state.global_step, 1)
sec_per_step = dt / n_steps
peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
total_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0.0)

print("=" * 70)
print("训练完成")
print("  steps     :", n_steps)
print("  训练耗时  : %.1f 秒 (%.2f 小时)" % (dt, dt / 3600))
print("  每步      : %.2f 秒" % sec_per_step)
print("  峰值显存  : %.2f GB / %.2f GB" % (peak_gb, total_gb))
full_steps = math.ceil(
    FULL_N_TRAIN / max(1, BATCH_SIZE * GRAD_ACCUM) * EPOCHS
)
print("  完整步数  : ~%d" % full_steps)
print("  完整周期  : ~%.1f 小时（按当前每步耗时估算）" % (
    sec_per_step * full_steps / 3600))
print("  说明       : 前几步含编译/缓存开销，估算略偏保守")
print("=" * 70)

print(">>> 开始评估 %d 条 ..." % len(eval_ds))
t1 = time.time()
preds, logits = manual_eval(model, eval_ds)
eval_time = time.time() - t1
acc = accuracy_score(eval_ds["labels"], preds)

print("=" * 70)
print("RESULT")
print("  unsloth   :", USED_UNSLOTH)
print("  LoRA      :", USE_LORA)
print("  R-Drop    :", True)
print("  SCL       : True")
print("  SCL queue :", QUEUE_SIZE)
print("  SCL alpha :", ALPHA_SCL)
print("  train     :", len(train_ds), "条 | epochs:", EPOCHS,
      "| max_len:", MAX_LEN, "| batch:", BATCH_SIZE)
print("  eval      :", len(eval_ds), "条 | 耗时 %.1f 秒" % eval_time)
print("  accuracy  :", round(float(acc), 4))
print("=" * 70)


# ---------------- 保存结果文件 ----------------
RESULT_DIR = os.path.join(OUT_DIR, "result")
ADAPTER_DIR = os.path.join(RESULT_DIR, "lora_adapter")
os.makedirs(RESULT_DIR, exist_ok=True)

try:
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(">>> LoRA adapter saved:", ADAPTER_DIR)
except Exception as e:
    print(">>> adapter save failed:", repr(e))
    try:
        model.base_model.save_pretrained(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)
        print(">>> LoRA adapter saved via base_model:", ADAPTER_DIR)
    except Exception as e2:
        print(">>> adapter fallback save failed:", repr(e2))

probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
pred_df = pd.DataFrame({
    "text": eval_raw["text"],
    "label": eval_ds["labels"],
    "prediction": preds,
    "prob_negative": probs[:, 0],
    "prob_positive": probs[:, 1],
})
PRED_CSV = os.path.join(RESULT_DIR, "predictions.csv")
pred_df.to_csv(PRED_CSV, index=False)
np.save(os.path.join(RESULT_DIR, "logits.npy"), logits)

metrics = {
    "model_name": MODEL_NAME,
    "unsloth": bool(USED_UNSLOTH),
    "lora": bool(USE_LORA),
    "r_drop": True,
    "scl": True,
    "scl_queue_size": QUEUE_SIZE,
    "scl_alpha": ALPHA_SCL,
    "scl_warmup_steps": SCL_WARMUP_STEPS,
    "alpha_rdrop": ALPHA_RDROP,
    "train_size": len(train_ds),
    "eval_size": len(eval_ds),
    "max_len": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "steps": n_steps,
    "accuracy": float(acc),
    "train_seconds": dt,
    "eval_seconds": eval_time,
    "peak_gpu_gb": float(peak_gb),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "unsloth_version": getattr(unsloth, "__version__", None),
}
METRICS_JSON = os.path.join(RESULT_DIR, "metrics.json")
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(">>> predictions saved:", PRED_CSV)
print(">>> metrics saved    :", METRICS_JSON)
print(">>> result directory :", RESULT_DIR)
