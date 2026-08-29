import os
import sys
import logging
import time
import random
import gc
import warnings

# 全局过滤警告，禁止刷屏
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_scheduler
)
from tqdm.auto import tqdm

# ===================== 全局配置 =====================
SEED = 42
MAX_SEQ_LEN = 256
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL_TEST = 32
LR = 3e-5
EPOCHS = 4
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 2
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# 固定随机种子
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===================== Kaggle路径 =====================
PATH_LABELED_TRAIN = "/kaggle/input/competitions/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
PATH_TEST = "/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip"
OUTPUT_CSV = "/kaggle/working/distilbert_native.csv"
BEST_MODEL_PATH = "/kaggle/working/best_distilbert.pt"

# ===================== GPU检测函数 =====================
def check_gpu():
    print("="*40)
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    if not has_cuda:
        print("⚠️ WARNING: No GPU detected! Please turn on T4 GPU accelerator!")
        return torch.device("cpu")
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}, Total Mem: {total_mem:.2f} GB")
    print("="*40)
    return torch.device("cuda")

# ===================== Dataset =====================
class TrainDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        if self.labels is not None:
            return len(self.labels)
        return len(next(iter(self.encodings.values())))


class TestDataset(Dataset):
    def __init__(self, encodings, num_samples=0):
        self.encodings = encodings
        self.num_samples = num_samples

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        return item

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(logging.INFO)

    device = check_gpu()

    # 读取数据
    train = pd.read_csv(PATH_LABELED_TRAIN, header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(PATH_TEST, header=0, delimiter="\t", quoting=3)

    train_texts = train["review"].tolist()
    train_labels = train["sentiment"].tolist()
    test_texts = test["review"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=SEED, shuffle=True
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)
    val_encodings = tokenizer(val_texts, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)
    test_encodings = tokenizer(test_texts, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)

    train_dataset = TrainDataset(train_encodings, train_labels)
    val_dataset = TrainDataset(val_encodings, val_labels)
    test_dataset = TestDataset(test_encodings, num_samples=len(test_texts))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL_TEST, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_VAL_TEST, shuffle=False)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.to(device)

    # BERT分组权重衰减
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LR)

    num_training_steps = EPOCHS * len(train_loader)
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    scaler = GradScaler()
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batch_cnt = 0

        model.train()
        pbar_train = tqdm(total=len(train_loader), desc=f"Epoch {epoch} Train", leave=False)
        for batch in train_loader:
            train_batch_cnt += 1
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 兼容旧版pytorch，去掉device_type
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            preds = torch.argmax(outputs.logits.cpu(), dim=1)
            train_acc_sum += accuracy_score(preds, labels.cpu())
            train_loss_sum += loss.item()

            pbar_train.update(1)
        pbar_train.close()

        # -------- Validation --------
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batch_cnt = 0

        with torch.no_grad():
            pbar_val = tqdm(total=len(val_loader), desc=f"Epoch {epoch} Val", leave=False)
            for batch in val_loader:
                val_batch_cnt += 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss = outputs.loss

                preds = torch.argmax(outputs.logits.cpu(), dim=1)
                val_acc_sum += accuracy_score(preds, labels.cpu())
                val_loss_sum += val_loss.item()

                pbar_val.update(1)
            pbar_val.close()

        epoch_cost = time.time() - start_time
        avg_train_loss = train_loss_sum / train_batch_cnt
        avg_train_acc = train_acc_sum / train_batch_cnt
        avg_val_loss = val_loss_sum / val_batch_cnt
        avg_val_acc = val_acc_sum / val_batch_cnt

        print(f">>> Epoch {epoch} finished | time: {epoch_cost:.2f}s ")
        print(f"    train_loss={avg_train_loss:.4f} train_acc={avg_train_acc:.4f}")
        print(f"    val_loss={avg_val_loss:.4f}   val_acc={avg_val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, BEST_MODEL_PATH)
            print(f"💾 Save best model, val_loss={best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"⏸ Early stop counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print("🛑 Trigger early stopping!")
                break
        print("-"*70)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 预测
    print("\n🔄 Load best checkpoint for test inference ...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_pred = []
    with torch.no_grad():
        pbar_test = tqdm(total=len(test_loader), desc="Prediction", leave=False)
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
            batch_pred = torch.argmax(outputs.logits.cpu(), dim=1).numpy().tolist()
            test_pred.extend(batch_pred)
            pbar_test.update(1)
        pbar_test.close()

    result_df = pd.DataFrame({"id": test["id"], "sentiment": test_pred})
    result_df.to_csv(OUTPUT_CSV, index=False, quoting=3)
    logging.info(f"Result saved to {OUTPUT_CSV}")
    print(f"✅ Output csv: {OUTPUT_CSV}")
    print(f"📊 Prediction distribution: 0:{sum(x==0 for x in test_pred)}, 1:{sum(x==1 for x in test_pred)}")

