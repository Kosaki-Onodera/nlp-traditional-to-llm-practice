import os
import sys
import logging
import zipfile

import pandas as pd
import numpy as np
import datasets

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split

# ========== Kaggle路径常量 ==========
INPUT_DIR = "/kaggle/input/competitions/word2vec-nlp-tutorial"
WORKING_DIR = "/kaggle/working"

TRAIN_ZIP = os.path.join(INPUT_DIR, "labeledTrainData.tsv.zip")
TEST_ZIP = os.path.join(INPUT_DIR, "testData.tsv.zip")

os.makedirs(os.path.join(WORKING_DIR, "checkpoint"), exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, "result"), exist_ok=True)


def read_tsv_from_zip(zip_path, tsv_filename):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(tsv_filename) as f:
            df = pd.read_csv(f, header=0, delimiter="\t", quoting=3)
    return df


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    print("==== 1.加载数据集 ====")
    train = read_tsv_from_zip(TRAIN_ZIP, "labeledTrainData.tsv")
    test = read_tsv_from_zip(TEST_ZIP, "testData.tsv")
    print(f"原始训练集:{len(train)}  测试集:{len(test)}")

    train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
    print(f"划分训练集:{len(train_df)} 验证集:{len(val_df)}")

    train_dataset = datasets.Dataset.from_dict({"label": train_df["sentiment"], "text": train_df["review"]})
    val_dataset = datasets.Dataset.from_dict({"label": val_df["sentiment"], "text": val_df["review"]})
    test_dataset = datasets.Dataset.from_dict({"text": test["review"]})

    print("==== 2.文本token化 ====")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    def preprocess_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_train = train_dataset.map(preprocess_fn, batched=True)
    tokenized_val = val_dataset.map(preprocess_fn, batched=True)
    tokenized_test = test_dataset.map(preprocess_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = np.sum(preds == labels) / len(labels)
        return {"accuracy": float(acc)}

    print("==== 3.开始训练 ====")
    training_args = TrainingArguments(
        output_dir=os.path.join(WORKING_DIR, "checkpoint"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(WORKING_DIR, "logs"),
        logging_steps=10000,
        save_strategy="no",
        eval_strategy="epoch",
        report_to="none",
        fp16=True,
        disable_tqdm=False,
    )

    # 修复：删除 tokenizer=tokenizer 参数
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print("==== 4.训练完成，开始预测测试集 ====")

    pred_out = trainer.predict(tokenized_test)
    test_pred = np.argmax(pred_out.predictions, axis=-1).flatten()

    result_df = pd.DataFrame({"id": test["id"], "sentiment": test_pred})
    out_file = os.path.join(WORKING_DIR, "result/bert_trainer.csv")
    result_df.to_csv(out_file, index=False, quoting=3)
    print(f"==== 全部完成，结果保存至: {out_file} ====")
