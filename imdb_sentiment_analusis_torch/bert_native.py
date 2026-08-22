import os
import sys
import logging
import time
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

# ===================== 全局随机种子，保证可复现 =====================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===================== 路径配置 =====================
TRAIN_ZIP_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
TEST_ZIP_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip"
SAMPLE_SUB_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/sampleSubmission.csv"
OUTPUT_CSV = "/kaggle/working/bert_submission.csv"
BEST_MODEL_PATH = "/kaggle/working/best_bert.pt"
os.makedirs("/kaggle/working", exist_ok=True)

# ==========日志修复 ==========
root_logger = logging.getLogger()
if root_logger.handlers:
    root_logger.handlers.clear()
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logger.setLevel(logging.INFO)
logger.info("=== Logger initialized successfully ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {device}")

# ===================== Dataset：接收已经encode好的数据 =====================
class ImdbDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ===================== 加载数据 =====================
logger.info("Loading training data ...")
train_df = pd.read_csv(TRAIN_ZIP_PATH, header=0, delimiter="\t", quoting=3)
logger.info(f"train shape = {train_df.shape}")

logger.info("Loading test data ...")
test_df = pd.read_csv(TEST_ZIP_PATH, header=0, delimiter="\t", quoting=3)
logger.info(f"test shape = {test_df.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    train_df["review"].tolist(),
    train_df["sentiment"].tolist(),
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["sentiment"]
)

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
max_seq_len = 256       # 提升序列长度，换取分数
batch_size = 8          # 256序列T4下调batch

logger.info("Pre‑tokenizing train/val/test texts ...")
train_enc = tokenizer(X_train, max_length=max_seq_len, truncation=True, padding="max_length")
val_enc = tokenizer(X_val, max_length=max_seq_len, truncation=True, padding="max_length")
test_enc = tokenizer(test_df["review"].tolist(), max_length=max_seq_len, truncation=True, padding="max_length")

train_dataset = ImdbDataset(train_enc, y_train)
val_dataset = ImdbDataset(val_enc, y_val)
test_dataset = ImdbDataset(test_enc, labels=None)

num_workers = 0
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

epochs = 3
lr = 1.5e-5
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)

total_steps = len(train_loader) * epochs
warmup_steps = int(total_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

scaler = GradScaler()
grad_clip_norm = 1.0

# ===================== 训练验证函数：关闭tqdm逐batch输出刷屏 =====================
def train_one_epoch(model, loader, opt, sch, scaler, dev):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    for batch in tqdm(loader, desc="Train", disable=False):
        input_ids = batch["input_ids"].to(dev)
        attn_mask = batch["attention_mask"].to(dev)
        labels = batch["labels"].to(dev)
        opt.zero_grad()

        with autocast('cuda'):
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss
            logits = out.logits

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(opt)
        scaler.update()
        sch.step()

        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu().numpy(), pred.cpu().numpy())
        total_acc += acc
        steps += 1
    return total_loss / steps, total_acc / steps

def val_one_epoch(model, loader, dev):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", disable=False):
            input_ids = batch["input_ids"].to(dev)
            attn_mask = batch["attention_mask"].to(dev)
            labels = batch["labels"].to(dev)
            with autocast('cuda'):
                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss
                logits = out.logits
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            acc = accuracy_score(labels.cpu().numpy(), pred.cpu().numpy())
            total_acc += acc
            steps += 1
    return total_loss / steps, total_acc / steps

# ===================== 主训练循环，只打印Epoch汇总，不再刷屏逐batch日志 =====================
logger.info("==== Start training ====")
best_val_acc = 0.0

for ep in range(1, epochs+1):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device)
    va_loss, va_acc = val_one_epoch(model, val_loader, device)
    cost = time.time() - t0
    msg = f"Epoch {ep:2d} | Train Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} | Val Loss:{va_loss:.4f} Acc:{va_acc:.4f} | Time:{cost:.1f}s"
    logger.info(msg)

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        logger.info(f"Save best model, val_acc={best_val_acc:.4f}")

logger.info(f"Load best checkpoint val_acc={best_val_acc:.4f}")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

# ===================== 预测 =====================
logger.info("Start predicting test set ...")
model.eval()
preds_all = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predict"):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        with autocast('cuda'):
            out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits
        pred = torch.argmax(logits, dim=1).cpu()
        preds_all.extend(pred.numpy().tolist())

sample = pd.read_csv(SAMPLE_SUB_PATH)
sample["sentiment"] = preds_all
sample.to_csv(OUTPUT_CSV, index=False, quoting=3)
logger.info(f"Prediction finished, saved to {OUTPUT_CSV}, shape {sample.shape}")
