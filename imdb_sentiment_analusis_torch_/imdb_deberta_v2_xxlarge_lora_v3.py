# -*- coding: utf-8 -*-
"""
IMDB 情感分析 · DeBERTa-v2-xxlarge + LoRA —— 「一趟出结果」版

⚠️⚠️ 运行前提：**必须用 transformers 4.x**（脚本会自己检查，是 5.x 就当场停）。
    2026-09-30 量出来的铁证 —— 同一套量法、同一批样本、同一份权重，只换版本：
        transformers 5.0.0    CLS 0.605 / 平均池化 0.656
        transformers 4.46.3   CLS 0.870 / 平均池化 0.934   ← 比 distilbert（0.868）还高
    5.0 的 deberta-v2 实现是坏的（连 LM 头的键名映射都错：checkpoint 里的
    lm_predictions.* 被当 UNEXPECTED 丢掉、cls.predictions.* 全 MISSING）。
    这一星期所有 0.51 / 0.60 / 「训不动 / loss 贴着 0.71」都是它造成的，
    跟模型、超参、池化、batch、步数**都没有关系**。
    跑法（顺序不能反）：
        第 1 格： !pip install -q "transformers==4.46.3"
        Restart Session（必须，否则内存里还是 5.x）
        第 2 格： 整份粘进本文件跑

和 imdb_deberta_v2_xxlarge_lora_trainer.py 的区别只有下面这几处，
每一处都对应**真机上量到的数字**，不是猜的：

  A. 池化换成「非 pad 位置的平均池化」，不再用模型自带那个 ContextPooler。
     那是个 1536×1536 的随机矩阵（预训练权重里根本没有它），真机实测就是
     2026-09-25「跑满两轮 val acc = 0.5104」的直接原因。
     真机线性探针： [CLS] 0.610 / 平均池化 0.676 —— 所以用平均池化。
  B. 分类头单独一组学习率（1e-3），LoRA 一组（2e-4）。
     Adam 每步不管梯度多小都要走 lr 那么远，而分类头初始尺度只有 0.02，
     2e-4 下走几百步只挪了自身尺度的零头 —— 这是「loss 贴着 0.71 不动」的一半原因。
  C. 精度不写死：fp16 / fp32 各跑 10 次优化器更新计时，谁快用谁。
     2026-09-29 真机实测：fp32 26.7 秒/次、fp16 10.2 秒/次 —— fp16 确实快 2.6 倍。
     也就是说更早那次「两个一样快」，确实说明混合精度没生效，这一条现在不用再猜了。
  D. 开跑前两道闸门（只打印、不拦路，照样往下跑、照样出 csv）：
     闸门 1  模型和分词器是不是一套的：慢版 vs 快版分词 id 是否一致、特殊符号 id，
             再用**冻住特征的线性探针**量一遍这套特征里有多少情感（800 条，约 30 秒）。
     闸门 2  能不能过拟合 128 条：同一条训练路径跑 150 次更新，训练准确率必须 > 0.9。
             过不去就说明「不是超参的事」，再调 lr / 步数也没用 —— 那组数会一起打出来。
  E. **按长度排队组批**（2026-09-29 实测之后加的）。随机组批时每批 4 条要补到该批最长
     （常常 470~512 token），而真实平均只有 264 —— 等于一半算力在算 pad。按长度分组之后
     一轮从 3.5 小时降到约 1.6 小时；等效 batch 同时从 16 降到 8，更新次数翻倍。
  F. 「掩码填词」那个闸门删了。transformers 5.0 里 DebertaV2ForMaskedLM 要的是
     cls.predictions.*，而这份 checkpoint 存的是 lm_predictions.*，两边对不上 ——
     MLM 头是随机初始化的，填出来全是噪声（实测填出：urse / locator / niz）。这个测试
     在 5.0 上本来就是废的。⚠️ 别把它误读成「权重坏了」：加载报告里编码器一个键都没缺，
     只有 classifier / pooler 是 MISSING，而那两个本来就该是随机初始化的。

产物（只要模型建起来了就会写）：
    ./result/microsoft_deberta-v2-xxlarge.csv      ← 交作业用
    ./result/deberta-v2-xxlarge-lora-adapter/      ← 想复现不用重训

跑法：Kaggle 新会话 → 第一个碰 GPU 的格子 → 整份粘进去（或 !python 这个文件）。
      时间：闸门约 5 分钟 + 训练 TIME_BUDGET_HOURS + 预测 0.5~1.5 小时。

默认 TIME_BUDGET_HOURS = 4.5（约 1.8 轮）。中途在 **5% / 25% / 50% / 75% / 100%** 各量一次
验证准确率 —— 5% 那一次（约十几分钟）是用来早看出「到底有没有在学」的。真要中断也请在
看完 25% 那一次之后再说，别只看 5% 那一个数就下结论（那时还没收敛）。
"""

import os

# 必须写在 import torch 之前。Kaggle 给 2 张 T4，不限定的话 DataParallel 会把
# 6.3G 的 fp32 权重每张卡各存一份，显存反而更紧。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# T4 一共 15G，这套模型光 fp32 权重就 6.3G，常年余量不到 1G，碎片一多就 OOM。
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import glob
import subprocess
import sys
import time
import types
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Kaggle 自带的 torchao 是 0.10.0，peft 0.19 要求 >= 0.16；版本不够不是「跳过不用」，
# 而是直接抛 ImportError（get_peft_model 挑后端时问到它就炸）。必须在 import peft 之前卸。
try:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
except Exception as _e:                                       # noqa: BLE001
    print("（卸 torchao 没成功，不致命：%s）" % str(_e)[:100])

import transformers                                           # noqa: E402
try:
    from peft import LoraConfig, TaskType, get_peft_model     # noqa: E402
except Exception as _e:                                       # noqa: BLE001
    print("peft 导入失败（%s）：%s" % (type(_e).__name__, str(_e)[:200]))
    print('多半是 peft 和 transformers 4.46 的搭配问题，试：')
    print('    !pip install -q "transformers==4.46.3" "peft==0.14.0"')
    raise
from sklearn.linear_model import LogisticRegression           # noqa: E402
from sklearn.model_selection import train_test_split          # noqa: E402
from sklearn.preprocessing import StandardScaler              # noqa: E402
from transformers import (AutoConfig, AutoModelForMaskedLM,    # noqa: E402
                          AutoModelForSequenceClassification, AutoTokenizer)

try:
    _TF_MAJOR = int(transformers.__version__.split(".")[0])
except Exception:                                             # noqa: BLE001
    _TF_MAJOR = 4

# ⚠️⚠️ 这一条是 2026-09-30 拿真金白银量出来的：deberta-v2 在 transformers 5.0 上的
# 实现是坏的 —— 同一套量法、同一批样本、同一份权重，只换版本：
#       transformers 5.0.0    CLS 0.605 / 平均池化 0.656
#       transformers 4.46.3   CLS 0.870 / 平均池化 0.934   ← 比 distilbert 还高
# 所以整条腿**必须**跑在 4.x 上。这一星期里所有的 0.51 / 0.60 / 「训不动」都是它造成的。
if _TF_MAJOR >= 5:
    sys.exit("停：transformers %s 的 deberta-v2 实现是坏的（实测 5.0 → 0.656、4.46 → 0.934）。"
             "先 Restart Session，然后在新会话的第一格跑：\n"
             '    !pip install -q "transformers==4.46.3"\n'
             "再 Restart Session，最后整份重跑本格。"
             % transformers.__version__)

# transformers 5.0 把 SequenceClassifierOutput 从顶层移走了（modeling_outputs 也重构过），
# 所以按三级兜底找：顶层 → modeling_outputs → 自己造一个长得一样的壳。
# 反正整份脚本只用到它的 .logits 一个字段。
_SCO_FROM = "transformers 顶层"
try:
    from transformers import SequenceClassifierOutput as _SCO           # noqa: E402
except Exception:                                             # noqa: BLE001
    try:
        from transformers.modeling_outputs import SequenceClassifierOutput as _SCO  # noqa: E402
        _SCO_FROM = "transformers.modeling_outputs"
    except Exception:                                         # noqa: BLE001
        _SCO = None
        _SCO_FROM = "自己造的壳（够用，只读 .logits）"


def make_output(logits):
    """包一层带 .logits 的返回值。v5 找不到那个 dataclass 就用 SimpleNamespace 顶替。"""
    if _SCO is None:
        return types.SimpleNamespace(logits=logits)
    return _SCO(logits=logits)


def get_logits(out):
    """不管模型吐回来的是 dataclass 还是裸张量，都取出 logits。"""
    return out.logits if hasattr(out, "logits") else out

# ===========================================================================
# ★ 配置区 ★
# ===========================================================================
MODEL_NAME = "microsoft/deberta-v2-xxlarge"

MAX_LENGTH = 512          # 和另外两条腿（bert / roberta）一样，结果才能横向比
MICRO_BATCH = 4           # 单次前向的样本数（真机实测 batch 4 装得下；batch 8 会 OOM）
ACCUM = 2                 # 梯度累积 → 等效 batch 8（比 16 多一倍更新次数，见文件头的 ⑳ 那段）
LR_LORA = 2e-4            # LoRA 的标准值（老师那份用的是默认 5e-5）
LR_HEAD = 1e-3            # 分类头单独放大：它初始尺度只有 0.02，lr 太小根本挪不动
WARMUP = 100

# 训练阶段最多跑多久。**4.5 小时不是拍出来的**：按「长度分组 + 等效 batch 8 + fp16」
# 估，一次更新约 3~4 秒 → 4.5 小时约 4500 次更新、看过 3.6 万条（约 1.8 轮）。
# 老师那份 demo 是 3 轮，这里取「时间装得下多少跑多少」，不为了省时间砍轮数。
# 整趟时间 ≈ 加载 10 分钟 + 闸门 5 分钟 + 训练 4.5 小时 + 预测 0.5~1 小时 ≈ 6 小时。
TIME_BUDGET_HOURS = 4.5

PRED_BATCH = 24           # 预测时按长度排队，批量放大只是省时间
VAL_BATCH = 8             # 训练中途量验证集时用的小批 —— 那一刻显存还被训练那边占着，
                          # 不能按预测的 24 来（2026-09-29 就是在这一步 OOM 的）
VAL_N = 1000              # 体检固定看验证集前 1000 条（每次同一批，能直接比）
PROBE_N = 800             # 闸门 1b 的线性探针用多少条训练 / 多少条验证
OVERFIT_N = 128           # 闸门 2 的样本数
OVERFIT_STEPS = 150       # 闸门 2 的优化器更新次数（真机实测 fp32 约 4~5 秒/次 → 约 11 分钟）
PROBE_UPDATES = 10        # 计时那段各跑多少次优化器更新

RESULT_CSV = "./result/microsoft_deberta-v2-xxlarge.csv"
ADAPTER_DIR = "./result/deberta-v2-xxlarge-lora-adapter"
LN2 = 0.6931471805599453  # 二分类纯瞎猜的 loss，一切跟它比

T0 = time.time()
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def elapsed():
    return (time.time() - T0) / 60.0


print("=" * 74)
print("DeBERTa-v2-xxlarge + LoRA（v3）")
print("transformers %s / torch %s / 设备 %s"
      % (transformers.__version__, torch.__version__, DEV))
print("SequenceClassifierOutput 来自：%s" % _SCO_FROM)
if torch.cuda.is_available():
    _free, _total = torch.cuda.mem_get_info(0)
    print("显卡：%s，可见 %d 张，当前空闲 %.2f / %.2f GB"
          % (torch.cuda.get_device_name(0), torch.cuda.device_count(),
             _free / 2**30, _total / 2**30))
    # ⚠️ 2026-09-29 踩过：同一个内核里把这一格跑第二遍 → 上一次的模型还压在显存里，
    #    这次加载时在 model.to(DEV) 那里直接 OOM（报错里写着 14.45G 已被占用）。
    #    显存是按「进程」算的，格子跑完不会自动释放。与其让人对着 OOM 猜半天，
    #    不如在花时间之前就停下来讲清楚。
    if _free < 12.0 * 2**30:
        sys.exit("停：这张卡上已经占了 %.2f GB —— 上一次跑的东西还在显存里。这份脚本"
                 "一上来就要加载 1.5B 的模型（中间还有一份临时的 MLM），卡上不空一定 OOM。"
                 "先 Restart Session（保留磁盘缓存，别用 Factory Reset），再整份重跑一次。"
                 % ((_total - _free) / 2**30))
else:
    sys.exit("停：没有 CUDA。这个脚本要在 Kaggle 的 GPU notebook 上跑。")
print("=" * 74)


# ---------------------------------------------------------------------------
# 数据目录：本机 ./corpus/imdb/；Kaggle 自动去 /kaggle/input/ 找（支持 .zip）
# 这一段和 imdb_bert_native.py 里的一模一样，没动过
# ---------------------------------------------------------------------------
def find_data_dir():
    local = "./corpus/imdb"
    if os.path.exists(os.path.join(local, "labeledTrainData.tsv")):
        return local
    hits = glob.glob("/kaggle/input/**/labeledTrainData.tsv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    zips = glob.glob("/kaggle/input/**/labeledTrainData.tsv.zip", recursive=True)
    if zips:
        src = os.path.dirname(zips[0])
        out = "./corpus/imdb"
        os.makedirs(out, exist_ok=True)
        for name in ["labeledTrainData.tsv", "testData.tsv"]:
            z = os.path.join(src, name + ".zip")
            if os.path.exists(z):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)
        found = glob.glob(os.path.join(out, "**", "labeledTrainData.tsv"), recursive=True)
        if found:
            return os.path.dirname(found[0])
    raise FileNotFoundError(
        "找不到 labeledTrainData.tsv —— Kaggle 上先把 word2vec-nlp-tutorial Add Data 进来")


DATA_DIR = find_data_dir()
print("数据目录：%s" % os.path.abspath(DATA_DIR))
train_df = pd.read_csv(os.path.join(DATA_DIR, "labeledTrainData.tsv"),
                       header=0, delimiter="\t", quoting=3)
test_df = pd.read_csv(os.path.join(DATA_DIR, "testData.tsv"),
                      header=0, delimiter="\t", quoting=3)

# random_state=0：和 imdb_bert_native.py / imdb_roberta_large.py 是同一份划分
train_texts, val_texts, train_labels, val_labels = train_test_split(
    list(train_df["review"]), list(train_df["sentiment"]), test_size=.2, random_state=0)
test_texts = list(test_df["review"])
print("样本：训练 %d / 验证 %d / 测试 %d"
      % (len(train_texts), len(val_texts), len(test_texts)))


def load_fp32(cls, name, **kw):
    """这个 checkpoint 发布时是 fp16 存的，不转 fp32 会撞
    「Attempting to unscale FP16 gradients」。transformers 5.x 把 torch_dtype
    改名成了 dtype；按版本先挑对的写法，不行再换另一种（两种写法在另一个版本上
    都不会报错、只会被当成 config 上的一个无关属性静默丢掉，所以必须按版本挑）。"""
    first = "dtype" if _TF_MAJOR >= 5 else "torch_dtype"
    second = "torch_dtype" if first == "dtype" else "dtype"
    last = None
    for key in (first, second):
        try:
            return cls.from_pretrained(name, **{key: torch.float32}, **kw)
        except Exception as e:                                # noqa: BLE001
            last = e
    raise last


# ===========================================================================
# 闸门 1a：分词器自检
# ===========================================================================
print("\n" + "=" * 74)
print("[闸门 1a] 分词器自检")
print("=" * 74)
cfg = AutoConfig.from_pretrained(MODEL_NAME)
tk_slow = tk_fast = None
try:
    tk_slow = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
except Exception as e:                                        # noqa: BLE001
    print("    慢版分词器没加载起来：%s" % str(e)[:120])
try:
    tk_fast = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
except Exception as e:                                        # noqa: BLE001
    print("    快版分词器没加载起来：%s" % str(e)[:120])
if tk_slow is None and tk_fast is None:
    raise RuntimeError("两个版本的分词器都加载不起来 —— 把上面两行报错发我。")

_probe = "This movie was GREAT, I loved it."
_ids_slow = None
tokenizer = tk_slow
if tk_slow is not None:
    _ids_slow = tk_slow(_probe)["input_ids"]
    print("    慢版 %s：%s" % (type(tk_slow).__name__, _ids_slow))
if tk_fast is not None:
    _ids_fast = tk_fast(_probe)["input_ids"]
    print("    快版 %s：%s" % (type(tk_fast).__name__, _ids_fast))
    if _ids_slow is None:
        tokenizer = tk_fast
    else:
        print("    → 两个版本 id %s" % ("一致" if _ids_slow == _ids_fast else "**不一致**"))
        if _ids_slow == _ids_fast:
            tokenizer = tk_fast
print("    用 %s；解回原文：%r"
      % (type(tokenizer).__name__, tokenizer.decode(tokenizer(_probe)["input_ids"])[:80]))
print("    vocab：分词器 %s / config %s（差 100 是 DeBERTa-v2 的已知形态，不是错）"
      % (getattr(tokenizer, "vocab_size", "?"), cfg.vocab_size))
_bad = []
for _n in ["pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id", "unk_token_id"]:
    _a, _b = getattr(tokenizer, _n, None), getattr(cfg, _n, None)
    if _b is None:
        # transformers 5.0 的 DebertaV2Config 上这些字段很多已经没有了（会在别处给默认值），
        # 拿不到就不比较 —— 2026-09-29 那次就是这里误报了一整条「分词器和权重不是一套」。
        print("      %-15s 分词器 %-8s config 上没有这个字段（跳过比较）" % (_n, _a))
        continue
    if _a != _b:
        _bad.append(_n)
    print("      %-15s 分词器 %-8s config %s" % (_n, _a, _b))
if _bad:
    print("    ⚠️ 这几个对不上：%s —— 分词器和权重不是一套的，先别往下看别的。"
          % "、".join(_bad))
else:
    print("    ✓ 能比对的都对得上；再加上「慢版/快版 id 一致 + 解码无损」，")
    print("      分词器这一关就算过了。")


# ---------------------------------------------------------------------------
# 关于「掩码填词」那个闸门，为什么删了（2026-09-29 实测）：
#   那一段本来想拿 checkpoint 自带的 MLM 头做「完形填空」，填出人话就说明权重和词表配套。
#   真机跑出来是：transformers 5.0 里 DebertaV2ForMaskedLM 要的是 cls.predictions.*，
#   而这个 checkpoint 里存的是 lm_predictions.lm_head.* —— 两边**对不上**：
#       lm_predictions.*            → UNEXPECTED（被丢掉）
#       cls.predictions.*           → MISSING（随机初始化）
#   于是那两句填词填出来的是纯噪声（urse / locator / niz …），**不能**拿来判断权重好坏
#   （加载报告里编码器本体一个键都没缺，只有 classifier / pooler 是 MISSING，那是应该的）。
#   结论：这个测试在 5.0 上是废的，删掉；换成下面那个「冻住特征的线性探针」，
#   它问的是同一个问题（这套特征里有没有情感），但直接、便宜、不会被头的问题污染。
# ---------------------------------------------------------------------------


# ===========================================================================
# 分词 + 建模型
# ===========================================================================
print("\n" + "=" * 74)
print("[准备] 分词 + 加载模型（1.5B，这一步省不掉）")
print("=" * 74)
train_enc = tokenizer(train_texts, truncation=True, padding=False, max_length=MAX_LENGTH)
val_enc = tokenizer(val_texts, truncation=True, padding=False, max_length=MAX_LENGTH)
test_enc = tokenizer(test_texts, truncation=True, padding=False, max_length=MAX_LENGTH)
_len_tr = np.array([len(x) for x in train_enc["input_ids"]])
print("    训练集 token 长度：中位数 %d，平均 %.0f，最长 %d（512 是 DeBERTa-v2 的硬顶）"
      % (int(np.median(_len_tr)), float(_len_tr.mean()), int(_len_tr.max())))
print("    加载 %s ..." % MODEL_NAME)
model = load_fp32(AutoModelForSequenceClassification, MODEL_NAME, num_labels=2)
model = model.float()
# ⚠️ 这一行不能漏。少了它，权重留在 CPU、输入在 GPU，前向到 embedding 那一步就报
#    「Expected all tensors to be on the same device」（2026-09-29 真机踩过）。
model = model.to(DEV)
print("    加载完（%.1f 分钟），权重落在 %s"
      % (elapsed(), next(model.parameters()).device))


def find_encoder_attr(mdl):
    """找出「编码器」挂在哪个属性名上。

    以前是写死 self.deberta 的，但 transformers 5.0 动过这些内部名字，
    所以这里按类型名去找（DebertaV2Model / DebertaV2Model 的包装），
    不再假设它一定叫 deberta。
    """
    kids = list(mdl.named_children())
    for name, mod in kids:
        if "Model" in type(mod).__name__ and type(mod).__name__.startswith("Deberta"):
            return name
    for name, mod in kids:
        if type(mod).__name__.startswith("Deberta"):
            return name
    for cand in ("deberta", "model", "base_model"):
        if isinstance(getattr(mdl, cand, None), torch.nn.Module):
            return cand
    raise RuntimeError("找不到编码器子模块，模型结构是：%s"
                       % [(n, type(m).__name__) for n, m in kids])


def find_head_attr(mdl):
    """找出「分类头」挂在哪个属性名上（classifier / score）。"""
    for cand in ("classifier", "score"):
        if isinstance(getattr(mdl, cand, None), torch.nn.Module):
            return cand
    for name, mod in mdl.named_children():
        if isinstance(mod, torch.nn.Linear):
            return name
    raise RuntimeError("找不到分类头，模型结构是：%s"
                       % [(n, type(m).__name__) for n, m in mdl.named_children()])


ENC_ATTR = find_encoder_attr(model)
HEAD_ATTR = find_head_attr(model)
_dropout = getattr(model, "dropout", None)
if not isinstance(_dropout, torch.nn.Module):
    _dropout = torch.nn.Identity()
print("    找到：编码器 self.%s（%s）/ 分类头 self.%s（%s）/ dropout %s"
      % (ENC_ATTR, type(getattr(model, ENC_ATTR)).__name__,
         HEAD_ATTR, type(getattr(model, HEAD_ATTR)).__name__, type(_dropout).__name__))


def patch_mean_pooling(mdl, enc_attr, head_attr, dropout):
    """把前向换成「非 pad 位置的平均池化」。

    原路是 [CLS] → ContextPooler(1536×1536 随机矩阵 + gelu) → Linear(1536, 2)。
    那个随机矩阵不在预训练权重里（加载报告写着 pooler.dense.weight | MISSING），
    真机实测它就是 0.51 的根源 —— 一层乱投影夹在编码器和分类头中间，
    分类头只能看见投影之后的东西（实测跨样本差异只剩 1~5%）。
    这里干脆绕开它，直接把整句平均池化的结果喂给分类头。
    """

    def _forward(self, input_ids=None, attention_mask=None, **kw):
        kw.pop("labels", None)
        out = getattr(self, enc_attr)(input_ids=input_ids,
                                      attention_mask=attention_mask, **kw)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return make_output(getattr(self, head_attr)(dropout(pooled)))

    mdl.forward = types.MethodType(_forward, mdl)


patch_mean_pooling(model, ENC_ATTR, HEAD_ATTR, _dropout)

# LoRA：只挂低秩旁路，15.7 亿预训练权重一个都不动。
# target_modules 写全三个投影（老师那份不写，peft 会自动补 query_proj + value_proj）。
# modules_to_save 必须写分类头 —— 它是随机初始化的、名字里又没有 "lora_"，
# 不写就会被 peft 冻住，顶上挂着一个永远不动的随机线性层，跑到天亮也是 0.51。
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_proj", "key_proj", "value_proj"],
    modules_to_save=[HEAD_ATTR],
    task_type=TaskType.SEQ_CLS,
)
try:
    model = get_peft_model(model, lora_config)
except Exception as _e:                                       # noqa: BLE001
    # 万一 5.x 把注意力的投影层改了名，指定 target_modules 会报「找不到模块」。
    # 那就退回老师那份的写法：不写 target_modules，让 peft 按模型类型自己挑。
    print("    指定 target_modules 没成功（%s），改成让 peft 自己挑。" % str(_e)[:110])
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        modules_to_save=[HEAD_ATTR], task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 梯度检查点：不开这个，batch 4 × 512 token 的激活值就有十几个 G，必 OOM。
# 代价是慢 20~30%（反向要重算一遍前向），但这是能在 T4 上跑起来的前提。
# 配套的 enable_input_require_grads() 不能漏：漏了的话每一层 checkpoint 的输入不带梯度，
# 整个编码器一个梯度都拿不到，等于只有分类头在训（不报错，只是永远 50%）。
try:
    model.gradient_checkpointing_enable()
except Exception as _e:                                       # noqa: BLE001
    print("    gradient_checkpointing_enable 出错：%s" % str(_e)[:120])


def _cp_count(mdl):
    return sum(1 for m in mdl.modules() if getattr(m, "gradient_checkpointing", False))


_n_cp = _cp_count(model)
if _n_cp == 0:
    # 没生效就手动逐层打开。2026-09-29 那次就是这一步没起作用：训练时每层的中间结果
    # 全留着（实测 14.28 G），连「量验证集」那点显存都挤不出来 —— 报错看着像显存不够，
    # 其实是梯度检查点根本没开。
    print("    ⚠️ 梯度检查点没打开 —— 手动逐层打开（不打开的话激活值要 14G，必 OOM）")
    try:
        import functools
        import torch.utils.checkpoint as _ckpt
        _cpf = functools.partial(_ckpt.checkpoint, use_reentrant=False)
        for _m in model.modules():
            if hasattr(_m, "gradient_checkpointing"):
                try:
                    _m.gradient_checkpointing = True
                except Exception:                             # noqa: BLE001
                    pass
            if hasattr(_m, "_gradient_checkpointing_func"):
                _m._gradient_checkpointing_func = _cpf
    except Exception as _e2:                                  # noqa: BLE001
        print("    手动打开也失败了：%s" % str(_e2)[:110])
    _n_cp = _cp_count(model)
print("    梯度检查点：%d 个模块已打开" % _n_cp)
if _n_cp == 0:
    print("    ⚠️ 还是 0 —— micro-batch 4 就是贴着上限在跑了。训练那一段会打一行")
    print("       「这一段峰值显存」，超过 14 GB 就把 MICRO_BATCH 改成 2、ACCUM 改成 4。")
try:
    model.enable_input_require_grads()
except Exception as _e:                                       # noqa: BLE001
    print("    enable_input_require_grads 出错：%s" % str(_e)[:120])
try:
    model.config.use_cache = False
except Exception:                                             # noqa: BLE001
    pass

# 保险：手工再把 modules_to_save 的那份分类头点亮一次。peft 正常会点亮它，
# 但版本之间有过差异（0.7.x 不认 modules_to_save 里的 classifier），
# 只点 wrapper 里那份副本，不去碰 original_module（前向根本不走它）。
for _n, _p in model.named_parameters():
    if "modules_to_save" in _n and HEAD_ATTR in _n:
        _p.requires_grad = True

_head_p = [(n, p) for n, p in model.named_parameters()
           if p.requires_grad and HEAD_ATTR in n]
_lora_n = sum(p.numel() for n, p in model.named_parameters()
              if p.requires_grad and HEAD_ATTR not in n)
print("    可训：分类头 %d 个张量 / %d 个参数；LoRA %d 个参数"
      % (len(_head_p), sum(p.numel() for _, p in _head_p), _lora_n))
for _n, _p in _head_p:
    print("      · %s  shape %s  requires_grad=%s" % (_n, tuple(_p.shape), _p.requires_grad))
if not _head_p or _lora_n == 0:
    sys.exit("停：分类头或 LoRA 被冻住了（可训数 0），训了也是白训。把上面这段发我。")


# ===========================================================================
# 手工批处理（不用 Trainer / DataCollator，少一层会因为版本炸掉的地方）
# ===========================================================================
def make_batch(enc, idx):
    """把 idx 指到的样本手动补到「这一批自己的最长」，返回 (ids, mask)。"""
    idx = [int(j) for j in idx]
    lens = [len(enc["input_ids"][j]) for j in idx]
    L = max(lens)
    ids = np.zeros((len(idx), L), dtype=np.int64)
    mask = np.zeros((len(idx), L), dtype=np.int64)
    for r, j in enumerate(idx):
        l = lens[r]
        ids[r, :l] = enc["input_ids"][j]
        mask[r, :l] = enc["attention_mask"][j]
    return ids, mask


# 干跑一次：确认「平均池化那条路」真的通了、形状对（4 条样本，长短还不一样）。
with torch.no_grad():
    _ids, _mask = make_batch(train_enc, range(4))
    _dry = get_logits(model(input_ids=torch.from_numpy(_ids).to(DEV),
                            attention_mask=torch.from_numpy(_mask).to(DEV)))
print("    干跑：logits %s（应该是 [4, 2]）" % (tuple(_dry.shape),))
if tuple(_dry.shape) != (4, 2):
    sys.exit("停：前向输出形状不对，把上面这段发我。")


# ===========================================================================
# 闸门 1b：冻住特征的线性探针 —— 这套特征里到底有没有情感
# ===========================================================================
def extract_mean_features(enc_mod, enc, n, batch=16, use_fp16=True):
    """按长度排队跑前向，返回每个样本的平均池化特征 [n, hidden]。

    走的就是分类头看到的那条路（非 pad 位置平均池化），只是不接分类头。
    """
    N = min(n, len(enc["input_ids"]))
    lens = [len(x) for x in enc["input_ids"][:N]]
    order = np.argsort(np.array(lens), kind="stable")
    F = np.zeros((N, enc_mod.config.hidden_size), dtype=np.float32)
    enc_mod.eval()
    with torch.no_grad():
        i = 0
        while i < N:
            idx = order[i:i + batch]
            L = max(lens[j] for j in idx)
            ids = np.zeros((len(idx), L), dtype=np.int64)
            mask = np.zeros((len(idx), L), dtype=np.int64)
            for r, j in enumerate(idx):
                ids[r, :lens[j]] = enc["input_ids"][j]
                mask[r, :lens[j]] = enc["attention_mask"][j]
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                h = enc_mod(input_ids=torch.from_numpy(ids).to(DEV),
                            attention_mask=torch.from_numpy(mask).to(DEV)).last_hidden_state
            h = h.float()
            # ⚠️ .to(h.dtype) 只换 dtype、不搬设备 —— 必须显式写 device，
            #    否则报「Expected all tensors to be on the same device」（2026-09-29 踩过）。
            m = torch.from_numpy(mask).unsqueeze(-1).to(device=h.device, dtype=h.dtype)
            F[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).cpu().numpy()
            i += len(idx)
    return F


print("\n" + "=" * 74)
print("[闸门 1b] 冻住特征的线性探针（%d 条训练 / %d 条验证）" % (PROBE_N, PROBE_N))
print("=" * 74)
try:
    _enc_mod = getattr(model.base_model.model, ENC_ATTR)
    _ftr = extract_mean_features(_enc_mod, train_enc, PROBE_N)
    _fva = extract_mean_features(_enc_mod, val_enc, PROBE_N)
    _sc = StandardScaler().fit(_ftr)
    _ytr = np.asarray(train_labels)[:PROBE_N]
    _yva = np.asarray(val_labels)[:PROBE_N]
    print("    特征量级：平均 |元素| %.2f，跨样本标准差占自身量级 %.0f%%"
          % (float(np.abs(_ftr).mean()),
             100 * float(_ftr.std(0).mean()) / (float(np.abs(_ftr).mean()) + 1e-12)))
    _best = (0.0, None)
    for _C in (0.01, 0.1, 1.0, 10.0):
        _clf = LogisticRegression(C=_C, max_iter=3000).fit(_sc.transform(_ftr), _ytr)
        _a = float((_clf.predict(_sc.transform(_fva)) == _yva).mean())
        print("    C=%-5s 验证 %.3f" % (_C, _a))
        if _a > _best[0]:
            _best = (_a, _C)
    print("    → 线性探针：%.3f（C=%s）。样本只有 %d 条，噪声约 ±%.1f%%。"
          % (_best[0], _best[1], PROBE_N, 100 * 0.5 / np.sqrt(PROBE_N)))
    print("      参照：之前用 1200 条量到 0.656，distilbert 同一套量法是 0.868。")
    print("      这个数是「冻住特征的天花板」，编码器在训练中能不能越过它是另一回事。")
    del _ftr, _fva, _sc
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:                                        # noqa: BLE001
    print("    这一段没跑起来（%s: %s）—— 跳过，看闸门 2。"
          % (type(e).__name__, str(e)[:150]))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:                                         # noqa: BLE001
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(mdl, lr_lora, lr_head):
    head, rest = [], []
    for n, p in mdl.named_parameters():
        if not p.requires_grad:
            continue
        (head if HEAD_ATTR in n else rest).append(p)
    return torch.optim.AdamW([
        {"params": rest, "lr": lr_lora, "weight_decay": 0.01},
        {"params": head, "lr": lr_head, "weight_decay": 0.0},
    ])


def predict(mdl, enc, use_fp16, n=None, batch=PRED_BATCH, tag=""):
    """按长度排队跑前向，返回和 enc 顺序对齐的预测。n=None 表示全部。"""
    N = len(enc["input_ids"]) if n is None else min(n, len(enc["input_ids"]))
    lens = [len(x) for x in enc["input_ids"][:N]]
    order = np.argsort(np.array(lens), kind="stable")
    preds = np.zeros(N, dtype=np.int64)
    was_training = mdl.training
    mdl.eval()
    t0 = time.time()
    # 用 no_grad 而不是 inference_mode：模型上挂着 enable_input_require_grads 的钩子，
    # 它每次前向都会给 embedding 输出打 requires_grad，而 inference_mode 造出来的
    # 是「推理张量」，两者会打架。no_grad 一样不建图，慢不了多少。
    with torch.no_grad():
        i = 0
        while i < N:
            idx = order[i:i + batch]
            L = max(lens[j] for j in idx)
            ids = np.zeros((len(idx), L), dtype=np.int64)
            mask = np.zeros((len(idx), L), dtype=np.int64)
            for r, j in enumerate(idx):
                l = lens[j]
                ids[r, :l] = enc["input_ids"][j]
                mask[r, :l] = enc["attention_mask"][j]
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    logits = get_logits(mdl(input_ids=torch.from_numpy(ids).to(DEV),
                                            attention_mask=torch.from_numpy(mask).to(DEV)))
            except torch.cuda.OutOfMemoryError:
                # 显存不够就当场把这个批减半重试，别把整趟跑废掉。
                # 2026-09-29 那次就是量验证集时死的（那一刻显存还被训练那边占着）。
                gc.collect()
                torch.cuda.empty_cache()
                if batch > 1:
                    batch = max(1, batch // 2)
                    print("      [%s] 显存不够，batch 降到 %d 接着跑" % (tag or "预测", batch))
                    continue
                raise
            preds[idx] = logits.float().argmax(-1).cpu().numpy()
            i += len(idx)
            if tag and i % (batch * 40) < batch:
                el = time.time() - t0
                print("      [%s] %d/%d 条，已用 %.1f 分钟，还要 %.1f 分钟"
                      % (tag, i, N, el / 60, el / i * (N - i) / 60))
    if was_training:
        mdl.train()
    return preds


def val_accuracy(mdl, use_fp16, n=VAL_N):
    p = predict(mdl, val_enc, use_fp16, n=n, batch=VAL_BATCH)
    y = np.asarray(val_labels)[:len(p)]
    return float((p == y).mean()), float(p.mean())


def train_loop(mdl, enc, labels, pool_idx, steps, micro, accum, use_fp16,
               lr_lora, lr_head, warmup, tag="训练", log_every=20,
               time_limit=None, val_every=0, val_at=None):
    """按「优化器更新次数」训练。返回 (实际走了多少步, 用了多少秒)。"""
    checks = sorted(set(list(range(val_every, steps + 1, val_every)) if val_every else [])
                    | set(val_at or []))
    opt = build_optimizer(mdl, lr_lora, lr_head)
    scaler = build_scaler(use_fp16)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * max(0.0, 1.0 - s / max(1, steps)))
    params = [p for p in mdl.parameters() if p.requires_grad]
    pool = np.asarray(pool_idx)
    labels_np = np.asarray(labels)
    rng = np.random.default_rng(0)
    # 按长度排队组批：一个 batch 里的样本长度接近，补出来的 pad 就少。
    # 2026-09-29 实测：随机组批时每批 4 条要补到该批最长（常常 470~512），而真实平均
    # 只有 264 token —— 等于**一半算力花在 pad 上**。按长度分组能把一轮从 3.5 小时
    # 压到约 1.6 小时（同样的 wall-clock 能多看一倍样本）。
    _plens = np.array([len(enc["input_ids"][int(j)]) for j in pool])
    _by_len = pool[np.argsort(_plens, kind="stable")]
    chunks = [_by_len[k:k + micro] for k in range(0, len(_by_len) - micro + 1, micro)]
    if not chunks:                                # 池子比 micro 还小
        chunks = [_by_len]
    chunk_order = rng.permutation(len(chunks))
    ci = 0
    mdl.train()
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()      # 用来量这一段的峰值显存
    log = []
    step = 0          # 只数「优化器更新次数」（累积 accum 个 micro-batch 才算一次）
    micro_i = 0
    while step < steps:
        if ci >= len(chunks):
            chunk_order = rng.permutation(len(chunks))
            ci = 0
        sel = chunks[int(chunk_order[ci])]
        ci += 1
        ids, mask = make_batch(enc, sel)
        y = torch.from_numpy(labels_np[sel]).to(DEV)
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                logits = get_logits(mdl(input_ids=torch.from_numpy(ids).to(DEV),
                                        attention_mask=torch.from_numpy(mask).to(DEV)))
                loss = F.cross_entropy(logits, y) / accum
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            print("    [%s] 显存不够：micro-batch %d 装不下。把配置区的 MICRO_BATCH 改成 2、"
                  "ACCUM 改成 4（等效 batch 还是 8），再整趟重跑。" % (tag, micro))
            print("       峰值显存 %.2f GB / 卡上共 %.2f GB。"
                  % (torch.cuda.max_memory_allocated() / 2**30,
                     torch.cuda.get_device_properties(0).total_memory / 2**30))
            raise
        log.append(loss.item() * accum)
        log.append(loss.item() * accum)
        micro_i += 1
        if micro_i % accum:
            continue
        # ---- 到这里才是一次完整的优化器更新 ----
        if use_fp16:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if use_fp16:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if log_every and step % log_every == 0:
            el = time.time() - t0
            _n = min(len(log), log_every * accum)
            print("    [%s] 更新 %5d/%d  loss %.4f（纯瞎猜 %.4f） %.2f 秒/次  已用 %.1f 分钟"
                  % (tag, step, steps, float(np.mean(log[-_n:])), LN2, el / step, el / 60))
        if checks and step >= checks[0]:
            checks.pop(0)
            acc, pos = val_accuracy(mdl, use_fp16)
            print("    [%s] ← 第 %d 步：验证集前 %d 条准确率 %.4f（判成正面的占 %.0f%%）"
                  % (tag, step, VAL_N, acc, pos * 100))
            mdl.train()
        if time_limit is not None and (time.time() - t0) > time_limit:
            print("    [%s] 到时间上限（%.1f 分钟），停在第 %d 步"
                  % (tag, (time.time() - t0) / 60, step))
            break
    if torch.cuda.is_available():
        print("    [%s] 这一段峰值显存 %.2f GB / 卡上共 %.2f GB（留了多少余量一眼可见）"
              % (tag, torch.cuda.max_memory_allocated() / 2**30,
                 torch.cuda.get_device_properties(0).total_memory / 2**30))
    return step, time.time() - t0


def snapshot_trainable(mdl):
    return {n: p.detach().clone() for n, p in mdl.named_parameters() if p.requires_grad}


def restore_trainable(mdl, snap):
    with torch.no_grad():
        for n, p in mdl.named_parameters():
            if n in snap:
                p.copy_(snap[n])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 闸门和计时都要真的更新权重，所以先存一份干净的可训权重，每趟跑完还原
snap = snapshot_trainable(model)


# ===========================================================================
# 闸门 2：能不能过拟合 128 条（同一条训练路径）
# ===========================================================================
print("\n" + "=" * 74)
print("[闸门 2] 过拟合 %d 条 —— 同样的训练路径、%d 次更新" % (OVERFIT_N, OVERFIT_STEPS))
print("=" * 74)
print("    训练准确率必须 > 0.9。过不去就说明「不是超参的事」，改 lr / 加步数都没用。")
GATE_PASSED = None
try:
    train_loop(model, train_enc, train_labels, np.arange(OVERFIT_N), OVERFIT_STEPS,
               micro=MICRO_BATCH, accum=1, use_fp16=True, lr_lora=LR_LORA, lr_head=LR_HEAD,
               warmup=20, tag="过拟合", log_every=50)
    _p = predict(model, train_enc, True, n=OVERFIT_N, batch=16)
    _fit = float((_p == np.asarray(train_labels)[:OVERFIT_N]).mean())
    GATE_PASSED = _fit > 0.9
    print("    → 这 %d 条上的训练准确率：%.4f" % (OVERFIT_N, _fit))
    _va, _ = val_accuracy(model, True)
    print("       同一时刻验证集前 %d 条 %.4f（这条先别当成绩看，才 %d 步）"
          % (VAL_N, _va, OVERFIT_STEPS))
    if GATE_PASSED:
        print("    ✓ 闸门通过：这条训练路径是活的，梯度真的流到了该流的地方。")
    else:
        print("    ✗ **闸门没过**：连 %d 条都记不住，这时候改超参是没用的 ——" % OVERFIT_N)
        print("      要么编码器吐出来的特征在这些样本之间几乎没差别（那要看池化/读哪一层），")
        print("      要么梯度根本没流到编码器（那只训了分类头）。")
        print("      下面会自动降级成「冻住编码器、只训分类头」—— 至少交出去的是这套特征")
        print("      能到的最高水平，不是一个没训动的模型。把上面整段发我。")
except Exception as e:                                        # noqa: BLE001
    print("    闸门 2 自己出错了（%s: %s）—— 跳过去，继续往训练走。"
          % (type(e).__name__, str(e)[:150]))
restore_trainable(model, snap)

if GATE_PASSED is False:
    # 降级：LoRA 冻上，只训分类头。这不是「微调」，是拿全量 2 万条做一个探针 ——
    # 但它是唯一还有意义的做法：编码器既然训不动，就只把顶上那一层用到最好。
    _frozen = 0
    for _n, _p in model.named_parameters():
        if _p.requires_grad and HEAD_ATTR not in _n:
            _p.requires_grad = False
            _frozen += 1
    try:
        model._require_grads_hook.remove()
    except Exception:                                         # noqa: BLE001
        pass
    try:
        model.base_model.model.deberta.gradient_checkpointing = False
    except Exception:                                         # noqa: BLE001
        pass
    print("    [降级] 冻住 %d 个 LoRA 张量；梯度检查点也关掉（前向不建图，反而更快更省）。"
          % _frozen)


# ===========================================================================
# 精度计时：fp16 / fp32 谁快用谁
# ===========================================================================
print("\n" + "=" * 74)
print("[计时] fp32 / fp16 各跑 %d 次优化器更新，看谁快" % PROBE_UPDATES)
print("=" * 74)
_clock_pool = np.arange(max(MICRO_BATCH * ACCUM * 2, 16))
_speed = {}
for _fp16 in (False, True):
    _name = "fp16" if _fp16 else "fp32"
    try:
        _steps, _sec = train_loop(model, train_enc, train_labels, _clock_pool, PROBE_UPDATES,
                                  micro=MICRO_BATCH, accum=ACCUM, use_fp16=_fp16,
                                  lr_lora=LR_LORA, lr_head=LR_HEAD, warmup=5,
                                  tag=_name, log_every=0)
        _per = _sec / max(1, _steps)
        _speed[_name] = _per
        print("    %s：%.2f 秒/次更新（等效 batch %d）→ 跑满一轮（%d 条）约 %.1f 小时"
              % (_name, _per, MICRO_BATCH * ACCUM, len(train_labels),
                 _per * (len(train_labels) / (MICRO_BATCH * ACCUM)) / 3600))
    except Exception as e:                                    # noqa: BLE001
        print("    %s 没跑起来：%s: %s" % (_name, type(e).__name__, str(e)[:120]))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    restore_trainable(model, snap)

if not _speed:
    sys.exit("两种精度都跑不起来 —— 显存真的不够。把上面那段发我。")
USE_FP16 = ("fp16" in _speed) and (_speed.get("fp16", 9e9) < _speed.get("fp32", 9e9))
SEC_PER_STEP = _speed["fp16" if USE_FP16 else "fp32"]
print("    → 用 %s（%.2f 秒/次更新）" % ("fp16" if USE_FP16 else "fp32", SEC_PER_STEP))
if "fp16" in _speed and "fp32" in _speed and _speed["fp16"] > 0.8 * _speed["fp32"]:
    print("    ⚠️ fp16 没有明显更快，这不正常（T4 的 fp16 张量核理论上是 fp32 的 8 倍）。")
    print("       多半是混合精度根本没生效 —— 记下来，别把它当成「模型就这么慢」。")


# ===========================================================================
# 正式训练
# ===========================================================================
print("\n" + "=" * 74)
print("[训练] 等效 batch %d，lr LoRA %.0e / 头 %.0e，最多 %.1f 小时"
      % (MICRO_BATCH * ACCUM, LR_LORA, LR_HEAD, TIME_BUDGET_HOURS))
print("=" * 74)
_epoch_steps = int(np.ceil(len(train_labels) / (MICRO_BATCH * ACCUM)))
_by_time = int(TIME_BUDGET_HOURS * 3600 / max(SEC_PER_STEP, 1e-6))
# 允许跑满 3 轮（老师那份 demo 就是 3 轮）；到底跑多少由时间闸门决定。
MAX_STEPS = max(50, min(3 * _epoch_steps, _by_time))
print("    一轮 = %d 条 ÷ 每次 %d 条 = %d 次更新；时间装得下 %d 次 → 这次训 %d 次（约 %.0f%% 轮）"
      % (len(train_labels), MICRO_BATCH * ACCUM, _epoch_steps, _by_time, MAX_STEPS,
         100.0 * MAX_STEPS / _epoch_steps))
_check_at = sorted({max(50, MAX_STEPS // 20), MAX_STEPS // 4, MAX_STEPS // 2,
                    MAX_STEPS * 3 // 4, MAX_STEPS})
print("    训练期间在第 %s 步各量一次验证集准确率（第一次是 5%% 处，好早点看出有没有在学）。"
      % " / ".join(str(s) for s in _check_at))
_steps_done, _train_sec = train_loop(
    model, train_enc, train_labels, np.arange(len(train_labels)), MAX_STEPS,
    micro=MICRO_BATCH, accum=ACCUM, use_fp16=USE_FP16, lr_lora=LR_LORA, lr_head=LR_HEAD,
    warmup=WARMUP, tag="训练", log_every=25,
    time_limit=TIME_BUDGET_HOURS * 3600, val_at=_check_at)
print("    训练结束：%d 次更新，%.1f 分钟" % (_steps_done, _train_sec / 60))

_va, _vp = val_accuracy(model, USE_FP16, n=len(val_labels))
print("=" * 74)
print("验证准确率（整份 %d 条）：%.4f（%.2f%%）；判成正面的占 %.0f%%"
      % (len(val_labels), _va, _va * 100, _vp * 100))
print("=" * 74)


# ===========================================================================
# 存 adapter + 出 csv（不管准确率多少都出）
# ===========================================================================
try:
    os.makedirs("./result", exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("LoRA adapter 已存 -> %s" % ADAPTER_DIR)
except Exception as e:                                        # noqa: BLE001
    print("存 adapter 出错了（不致命）：%s: %s" % (type(e).__name__, str(e)[:120]))

print("\n开始预测 %d 条测试集（按长度排队）..." % len(test_texts))
_test_pred = predict(model, test_enc, USE_FP16, tag="测试集预测")
os.makedirs("./result", exist_ok=True)
pd.DataFrame(data={"id": test_df["id"], "sentiment": _test_pred}).to_csv(
    RESULT_CSV, index=False, quoting=3)
print("结果已保存 -> %s" % RESULT_CSV)
print("预测成正面/负面的比例：%.1f%% / %.1f%%"
      % (100.0 * _test_pred.mean(), 100.0 * (1 - _test_pred.mean())))
print("\n总耗时：%.2f 小时" % ((time.time() - T0) / 3600))
print("把上面从「[闸门 1a]」到最后这一行的整段打印发我。")
