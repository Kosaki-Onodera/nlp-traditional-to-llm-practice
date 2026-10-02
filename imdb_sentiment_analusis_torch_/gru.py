import logging
import os
import sys
import pickle
import time
import random
import numpy as np

import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ====================== 超参数 ======================
num_epochs = 10
embed_size = 300
num_hiddens = 120
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 1e-3                  # 更换Adam，降低初始学习率
max_grad_norm = 5.0        # 梯度裁剪，防止梯度爆炸
dropout_rate = 0.2         # GRU层间dropout
use_emb_finetune = False   # 是否微调预训练GloVe
patience = 3               # 早停耐心值

use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu')

# 固定随机种子，保证实验可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu, dropout, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = use_emb_finetune

        self.encoder = nn.GRU(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=dropout if num_layers > 1 else 0, batch_first=False)

        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 2, labels)
        else:
            self.decoder = nn.Linear(num_hiddens, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        if self.bidirectional:
            encoding = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            encoding = hidden[-1]
        outputs = self.decoder(encoding)
        return outputs


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = '/kaggle/working/pickle/imdb_glove.pickle3'
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
     vocab] = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    net = SentimentNet(embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers,
                       bidirectional=bidirectional, weight=weight,
                       labels=labels, use_gpu=use_gpu, dropout=dropout_rate)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    # 替换SGD为Adam，收敛更快更稳定
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 学习率衰减：每轮衰减0.95
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    early_stop_count = 0

    for epoch in range(num_epochs):
        start = time.time()
        train_loss_total = 0.0
        val_loss_total = 0.0
        train_acc_total = 0.0
        val_acc_total = 0.0
        train_batch_num = 0
        val_batch_num = 0

        net.train()
        with tqdm(total=len(train_iter), desc=f'Epoch {epoch} Train') as pbar:
            for feature, label in train_iter:
                train_batch_num += 1
                net.zero_grad()
                feature = feature.to(device)
                label = label.to(device)
                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()

                # 梯度裁剪，解决循环网络梯度爆炸
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

                pred = torch.argmax(score, dim=1)
                train_acc_total += accuracy_score(pred.cpu(), label.cpu())
                train_loss_total += loss.item()

                pbar.set_postfix({
                    'train_loss': f'{train_loss_total / train_batch_num:.4f}',
                    'train_acc': f'{train_acc_total / train_batch_num:.4f}'
                })
                pbar.update(1)

        net.eval()
        with torch.no_grad():
            for val_feature, val_label in val_iter:
                val_batch_num += 1
                val_feature = val_feature.to(device)
                val_label = val_label.to(device)
                val_score = net(val_feature)
                val_loss = loss_function(val_score, val_label)
                val_pred = torch.argmax(val_score, dim=1)
                val_acc_total += accuracy_score(val_pred.cpu(), val_label.cpu())
                val_loss_total += val_loss.item()

        # 计算本轮指标
        avg_train_loss = train_loss_total / train_batch_num
        avg_train_acc = train_acc_total / train_batch_num
        avg_val_loss = val_loss_total / val_batch_num
        avg_val_acc = val_acc_total / val_batch_num
        runtime = time.time() - start

        logger.info(f"Epoch {epoch}: "
                    f"TrainLoss:{avg_train_loss:.4f} TrainAcc:{avg_train_acc:.4f} | "
                    f"ValLoss:{avg_val_loss:.4f} ValAcc:{avg_val_acc:.4f} Time:{runtime:.2f}s")

        # 保存最优模型 + 早停逻辑
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            early_stop_count = 0
            torch.save(net.state_dict(), "/kaggle/working/best_gru_model.pth")
            logging.info(f"Save best model, best val acc: {best_val_acc:.4f}")
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                logging.info(f"Early stop trigger, patience={patience}")
                break

        scheduler.step()

    # 加载验证集最优权重进行预测（关键！不用最后一轮模型）
    net.load_state_dict(torch.load("/kaggle/working/best_gru_model.pth"))
    net.eval()

    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.to(device)
                test_score = net(test_feature)
                batch_pred = torch.argmax(test_score.cpu(), dim=1).numpy().tolist()
                test_pred.extend(batch_pred)
                pbar.update(1)

    # 读取测试集id生成提交文件
    TEST_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip"
    test_df = pd.read_csv(TEST_PATH, header=0, delimiter="\t", quoting=3)
    result_output = pd.DataFrame(data={"id": test_df["id"], "sentiment": test_pred})

    save_path = "/kaggle/working/gru.csv"
    result_output.to_csv(save_path, index=False, quoting=3)
    logging.info(f'result saved to {save_path}! Best Val Acc = {best_val_acc:.4f}')
