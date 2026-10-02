import os
import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import datasets

from transformers import BertTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class BertScratch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_labels=2):
        config = BertConfig.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
        model = cls(config)
        model.bert = BertModel.from_pretrained(pretrained_model_name_or_path, config=config)
        return model


if __name__ == '__main__':
    DATA_ROOT = "/kaggle/input/competitions/word2vec-nlp-tutorial/"
    TRAIN_PATH = os.path.join(DATA_ROOT, "labeledTrainData.tsv.zip")
    TEST_PATH = os.path.join(DATA_ROOT, "testData.tsv.zip")

    OUTPUT_DIR = "/kaggle/working/checkpoint"
    RESULT_DIR = "/kaggle/working"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH, header=0, delimiter="\t", quoting=3, compression="zip")
    test_df = pd.read_csv(TEST_PATH, header=0, delimiter="\t", quoting=3, compression="zip")

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_dict = {'label': train_df["sentiment"], 'text': train_df['review']}
    val_dict = {'label': val_df["sentiment"], 'text': val_df['review']}
    test_dict = {"text": test_df['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def preprocess_function(examples):
        # 🔴提速：max_length从512降低到256
        return tokenizer(examples['text'], truncation=True, max_length=256)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    # 🔴提速：设置torch格式，减少CPU‑GPU拷贝
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = BertScratch.from_pretrained('bert-base-uncased', num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=200,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=500,      # 🔴不要每epoch完整验证，改为每500步评估一次
        fp16=True,
        report_to="none"
    )

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

    result_output = pd.DataFrame(data={"id": test_df["id"], "sentiment": test_pred})
    result_output.to_csv(os.path.join(RESULT_DIR, "bert_scratch.csv"), index=False, quoting=3)
    print("Result saved to /kaggle/working/bert_scratch.csv")
