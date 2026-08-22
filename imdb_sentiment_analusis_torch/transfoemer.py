import logging
import os
import sys
import time
import math
import re

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ===================== Kaggle路径配置 =====================
TRAIN_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
TEST_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip"
OUTPUT_DIR = "/kaggle/working/result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 超参 =====================
num_epochs = 10
MAX_SEQ_LEN = 512    # 句子最大截断长度，和位置编码max_len对齐
embed_size = 128
num_hiddens = 128
num_layers = 2
num_head = 4
dim_feedforward = 512
batch_size = 32
labels = 2
lr = 1e-4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    return words  # 修复：返回单词list，不再返回字符串！！！


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, train_sents, test_sents, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in train_sents:
            for token in sentence:
                token_freqs[token] += 1
        for sentence in test_sents:
            for token in sentence:
                token_freqs[token] += 1

        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items()
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]


def length_to_mask(lengths, device):
    max_length = torch.max(lengths)
    mask = torch.arange(max_length, device=device).expand(lengths.shape[0], max_length) < lengths.unsqueeze(1)
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=4, num_layers=2, dropout=0.1, max_len=512, activation: str = "relu"):
        super(Transformer, self).__init__()
        assert embedding_dim == hidden_dim, "embedding_dim必须等于hidden_dim(d_model)"
        assert hidden_dim % num_head == 0, "hidden_dim必须可以被num_head整除"

        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        # inputs: [batch, seq_len] batch_first=True
        inputs = torch.transpose(inputs, 0, 1)  # [seq_len, batch]
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)

        attention_mask = length_to_mask(lengths, inputs.device) == False
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# 训练/验证集collate_fn
def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


# 测试集collate_fn
def collate_fn_test(examples):
    lengths = torch.tensor([len(ex) for ex in examples])
    inputs = [torch.tensor(ex) for ex in examples]
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    train = pd.read_csv(TRAIN_PATH, header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(TEST_PATH, header=0, delimiter="\t", quoting=3)

    clean_train_tokens, train_labels = [], []
    for i, review in enumerate(train["review"]):
        tokens = review_to_wordlist(review)
        tokens = tokens[:MAX_SEQ_LEN]  # 长句子截断！
        clean_train_tokens.append(tokens)
        train_labels.append(train["sentiment"][i])

    clean_test_tokens = []
    for review in test["review"]:
        tokens = review_to_wordlist(review)
        tokens = tokens[:MAX_SEQ_LEN]
        clean_test_tokens.append(tokens)

    vocab = Vocab.build(clean_train_tokens, clean_test_tokens)
    logger.info(f"vocab size: {len(vocab)}")

    train_reviews = [(vocab.convert_tokens_to_ids(sentence), train_labels[i])
                     for i, sentence in enumerate(clean_train_tokens)]
    test_reviews = [vocab.convert_tokens_to_ids(sentence)
                    for sentence in clean_test_tokens]

    train_reviews, val_reviews, train_labels, val_labels = train_test_split(
        train_reviews, train_labels, test_size=0.2, random_state=0)

    net = Transformer(
        vocab_size=len(vocab),
        embedding_dim=embed_size,
        hidden_dim=num_hiddens,
        num_class=labels,
        dim_feedforward=dim_feedforward,
        num_head=num_head,
        num_layers=num_layers,
        dropout=0.1,
        max_len=512,
        activation="relu"
    )
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = TransformerDataset(train_reviews)
    val_set = TransformerDataset(val_reviews)
    test_set = TransformerDataset(test_reviews)

    train_iter = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn_test, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0.0, 0.0
        train_acc, val_acc = 0.0, 0.0
        n, m = 0, 0

        net.train()
        with tqdm(total=len(train_iter), desc=f'Epoch {epoch}') as pbar:
            for feature, lengths, label in train_iter:
                n += 1
                net.zero_grad()
                feature = feature.to(device)
                lengths = lengths.to(device)
                label = label.to(device)
                score = net(feature, lengths)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()

                train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
                train_loss += loss.item()

                pbar.set_postfix({
                    'train_loss': f'{train_loss / n:.4f}',
                    'train_acc': f'{train_acc / n:.2f}'
                })
                pbar.update(1)

        net.eval()
        with torch.no_grad():
            for val_feature, val_length, val_label in val_iter:
                m += 1
                val_feature = val_feature.to(device)
                val_length = val_length.to(device)
                val_label = val_label.to(device)
                val_score = net(val_feature, val_length)
                val_loss = loss_function(val_score, val_label)
                val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                val_losses += val_loss.item()

        end = time.time()
        runtime = end - start
        print(f"Epoch {epoch} | train_loss:{train_loss / n:.4f} train_acc:{train_acc / n:.4f} | val_loss:{val_losses / m:.4f} val_acc:{val_acc / m:.4f} | time:{runtime:.2f}s")

    # 预测测试集
    net.eval()
    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, test_len in test_iter:
                test_feature = test_feature.to(device)
                test_len = test_len.to(device)
                test_score = net(test_feature, test_len)
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    save_path = os.path.join(OUTPUT_DIR, "transformer.csv")
    result_output.to_csv(save_path, index=False, quoting=3)
    logger.info(f'result saved to {save_path}')
