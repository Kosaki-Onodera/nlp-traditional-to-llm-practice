import os
import sys
import logging
import warnings

import pandas as pd
import numpy as np

import datasets
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# -------------------------- Kaggle 路径配置 --------------------------
PATH_LABELED_TRAIN = "/kaggle/input/competitions/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
PATH_TEST = "/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip"

# 创建输出文件夹
os.makedirs("./results", exist_ok=True)
os.makedirs("./result", exist_ok=True)

warnings.filterwarnings("ignore")

# 日志配置
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info(f"running {''.join(sys.argv)}")


train = pd.read_csv(PATH_LABELED_TRAIN, header=0, delimiter="\t", quoting=3)
test = pd.read_csv(PATH_TEST, header=0, delimiter="\t", quoting=3)

logger.info(f"train shape: {train.shape}, test shape: {test.shape}")

train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

train_dict = {'label': train_df["sentiment"], 'text': train_df['review']}
val_dict = {'label': val_df["sentiment"], 'text': val_df['review']}
test_dict = {"text": test['review']}

train_dataset = datasets.Dataset.from_dict(train_dict)
val_dataset = datasets.Dataset.from_dict(val_dict)
test_dataset = datasets.Dataset.from_dict(test_dict)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# 手动计算accuracy，不依赖evaluate库
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    correct = np.sum(predictions == labels)
    total = len(labels)
    acc = correct / total
    return {"accuracy": acc}


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=24,
    warmup_steps=500,
    weight_decay=0.01,
    # 删掉已废弃 logging_dir
    logging_steps=100,
    save_strategy="no",
    eval_strategy="epoch",
    fp16=True,
    report_to="none"
)

# ✅ 删除 Trainer 的 tokenizer=tokenizer 参数，新版本不支持
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

prediction_outputs = trainer.predict(tokenized_test)
test_pred = np.argmax(prediction_outputs.predictions, axis=-1).flatten()

result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
result_output.to_csv("./result/distilbert_trainer.csv", index=False, quoting=3)
logger.info('result saved!')

trainer.save_model("./results/distilbert-imdb-final")
tokenizer.save_pretrained("./results/distilbert-imdb-final")

print("✅完成！提交文件路径： ./result/distilbert_trainer.csv")
print(result_output.head())
