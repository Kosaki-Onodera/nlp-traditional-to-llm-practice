# -*- coding: utf-8 -*-
"""
IMDB 情感分析 —— RoBERTa-large

模型：roberta-large（355M 参数）

RoBERTa = "Robustly optimized BERT approach"，是 Meta 在 BERT 基础上改进的版本。
与 BERT 的主要区别（跑起来后看程序打印的 config 就能对上）：

  1. 去掉 NSP 任务，只保留 MLM
     → config 里 type_vocab_size 由 2 变成 1，token_type_embeddings 只剩一行
  2. 词表从 30522 扩到 50265（BPE）
     → config 里 vocab_size 变成 50265，pad_token_id 由 0 变成 1
       （BERT 的 0 是 [PAD]，RoBERTa 的 0 是 <s>，1 才是 <pad>）
  3. 掩码方式由静态改成动态
     → BERT 在数据预处理时一次性盖好；RoBERTa 每个 epoch 重新采样要盖的位置
  4. 位置编码仍是绝对位置（没有相对位置），只是 max_position_embeddings 是 514
  5. 训练数据更多、batch 更大、训练更久（架构本身和 BERT 几乎一样）

本文件与 imdb_bert_native.py / imdb_deberta_large.py 除
MODEL_NAME / BATCH_SIZE / LEARNING_RATE 三处配置外，代码完全一致。
"""
import glob
import os
import sys
import logging
import time
import zipfile

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===========================================================================
# ★ 配置区：换模型只改 MODEL_NAME 这一行 ★
#
#   "bert-base-uncased"             原始 BERT-base（110M）
#   "roberta-base"                  RoBERTa-base（125M）
#   "roberta-large"                 RoBERTa-large（355M）  ← 当前
#   "microsoft/deberta-large"       DeBERTa v1-large（约 400M）
#   "microsoft/deberta-v2-xxlarge"  DeBERTa v2-xxlarge（1.5B）
# ===========================================================================
MODEL_NAME = "roberta-large"

# roberta-large 有 355M 参数，显存比 bert-base 紧张，batch 要调小；
# 模型越大学习率要越小，1e-5 是 roberta-large 的常用值（bert-base 用 5e-5）。
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
# ===========================================================================

# ---------------------------------------------------------------------------
# 数据目录：本机用 ./corpus/imdb/；Kaggle 上自动找 /kaggle/input/（支持 .zip）
# ---------------------------------------------------------------------------
def find_data_dir():
    # ① 本机：数据直接放在 ./corpus/imdb/ 下
    local = "./corpus/imdb"
    if os.path.exists(os.path.join(local, "labeledTrainData.tsv")):
        return local

    # ② Kaggle：已经是解压好的 tsv
    hits = glob.glob("/kaggle/input/**/labeledTrainData.tsv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])

    # ③ Kaggle：只有 .tsv.zip。/kaggle/input 是只读的，解压到可写的 ./corpus/imdb/
    zips = glob.glob("/kaggle/input/**/labeledTrainData.tsv.zip", recursive=True)
    if zips:
        src = os.path.dirname(zips[0])
        out = "./corpus/imdb"
        os.makedirs(out, exist_ok=True)
        for name in ["labeledTrainData.tsv", "testData.tsv"]:
            z = os.path.join(src, name + ".zip")
            if os.path.exists(z):
                print("解压：", z)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)
        # zip 里可能还套了一层目录，再找一次
        found = glob.glob(os.path.join(out, "**", "labeledTrainData.tsv"), recursive=True)
        if found:
            return os.path.dirname(found[0])

    raise FileNotFoundError(
        "找不到 labeledTrainData.tsv —— 本机请放到 ./corpus/imdb/ 下；"
        "Kaggle 请先把 word2vec-nlp-tutorial 数据集 Add Data 到 notebook 里"
    )


DATA_DIR = find_data_dir()
print("数据目录：", os.path.abspath(DATA_DIR))

train = pd.read_csv(os.path.join(DATA_DIR, "labeledTrainData.tsv"), header=0, delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(DATA_DIR, "testData.tsv"), header=0, delimiter="\t", quoting=3)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, num_samples=0):
        self.encodings = encodings
        self.num_samples = num_samples

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train_texts, train_labels, test_texts = [], [], []
    for i, review in enumerate(train["review"]):
        train_texts.append(review)
        train_labels.append(train['sentiment'][i])

    for review in test['review']:
        test_texts.append(review)

    # 固定 random_state，保证三个模型用的是同一份验证集划分，结果才可横向比较
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=.2, random_state=0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    train_dataset = TrainDataset(train_encodings, train_labels)
    val_dataset = TrainDataset(val_encodings, val_labels)
    test_dataset = TestDataset(test_encodings, num_samples=len(test_texts))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    # ---- 打印模型配置：几个模型之间的区别就写在这里 ----------------------
    print("=" * 66)
    print("模型：      %s" % MODEL_NAME)
    print("参数量：    %.1f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
    for key in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                "intermediate_size", "max_position_embeddings", "type_vocab_size",
                "pad_token_id", "hidden_dropout_prob", "layer_norm_eps",
                "relative_attention", "position_biased_input"]:
        print("%-28s %s" % (key + ":", getattr(model.config, key, "—")))
    print("=" * 66)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ---- fp16 混合精度（T4 实测 1.394 → 0.512 s/step，快 2.7 倍）----------
    # autocast 管前向：矩阵乘法走 fp16 张量核心（T4 上 fp16 峰值是 fp32 的 8 倍），
    # 而 softmax / LayerNorm / loss 这些对精度敏感的算子仍留在 fp32；
    # GradScaler 管反向：把 loss 放大再缩回，防止 fp16 梯度下溢成 0。
    # 本机没有 CUDA 时 use_amp=False，两个开关自动变成空操作，代码不用动。
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # ----------------------------------------------------------------------

    for epoch in range(3):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_correct, val_correct = 0, 0
        train_total, val_total = 0, 0
        n, m = 0, 0

        model.train()
        with tqdm(total=len(train_loader), desc="Epoch %d" % epoch) as pbar:
            for batch in train_loader:
                n += 1
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_correct += (torch.argmax(outputs.logits, dim=1).cpu() == labels.cpu()).sum().item()
                train_total += labels.size(0)
                train_loss += loss.cpu()

                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_correct / train_total)
                                  })
                pbar.update(1)

            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    m += 1
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss = outputs.loss
                    val_correct += (torch.argmax(outputs.logits, dim=1).cpu() == labels.cpu()).sum().item()
                    val_total += labels.size(0)
                    val_losses += val_loss
            end = time.time()
            runtime = end - start
            pbar.set_postfix({'epoch': '%d' % (epoch),
                              'train loss': '%.4f' % (train_loss.data / n),
                              'train acc': '%.2f' % (train_correct / train_total),
                              'val loss': '%.4f' % (val_losses.data / m),
                              'val acc': '%.2f' % (val_correct / val_total),
                              'time': '%.2f' % (runtime)})

    model.eval()
    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Prediction') as pbar:
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    outputs = model(input_ids, attention_mask=attention_mask)
                test_pred.extend(torch.argmax(outputs.logits.cpu().data, dim=1).numpy().tolist())

                pbar.update(1)

    os.makedirs("./result", exist_ok=True)
    result_name = MODEL_NAME.replace("/", "_")
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/%s.csv" % result_name, index=False, quoting=3)
    logging.info('result saved! -> ./result/%s.csv' % result_name)
