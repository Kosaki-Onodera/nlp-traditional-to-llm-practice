import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

test = pd.read_csv("/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter="\t", quoting=3)

num_epochs = 10
embed_size = 300
num_hiddens = 120
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.001

# 强制GPU模式，没有GPU直接终止程序
if not torch.cuda.is_available():
    raise RuntimeError("当前环境无可用GPU，请切换GPU运行时！")
device = torch.device("cuda")
use_gpu = True
print(f"使用GPU设备: {torch.cuda.get_device_name(device)}")

class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        
        # 双向：首尾(240) + max_pool(240) + mean_pool(240) = 720 = num_hiddens * 6
        if self.bidirectional:
            feature_dim = num_hiddens * 6
        else:
            feature_dim = num_hiddens * 3
        
        # 增加BN+Dropout，弱化局部极端词语干扰
        self.bn = nn.BatchNorm1d(feature_dim)
        self.dropout = nn.Dropout(0.25)
        self.decoder = nn.Linear(feature_dim, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        
        forward_last = states[-1, :, :self.num_hiddens]
        backward_last = states[0, :, self.num_hiddens:]
        last_cat = torch.cat([forward_last, backward_last], dim=-1)
        
        max_pool = torch.max(states, dim=0)[0]
        mean_pool = torch.mean(states, dim=0)
        
        encoding = torch.cat([last_cat, max_pool, mean_pool], dim=-1)
        encoding = self.bn(encoding)
        encoding = self.dropout(encoding)
        outputs = self.decoder(encoding)
        return outputs

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'imdb_glove.pickle3')
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
     vocab] = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    weight = weight.to(device)
    net = SentimentNet(embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers,
                       bidirectional=bidirectional, weight=weight,
                       labels=labels, use_gpu=use_gpu)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        error_cases = []
        net.train()
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, label in train_iter:
                n += 1
                net.zero_grad()
                feature = feature.to(device)
                label = label.to(device)
                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()
                train_acc += accuracy_score(torch.argmax(score.data, dim=1).cpu(), label.cpu())
                train_loss += loss

                pbar.set_postfix({
                    'train loss': '%.4f' % (train_loss.data / n),
                    'train acc': '%.2f' % (train_acc / n)
                })
                pbar.update(1)

            net.eval()
            with torch.no_grad():
                for val_feature, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.to(device)
                    val_label = val_label.to(device)
                    val_score = net(val_feature)
                    val_loss = loss_function(val_score, val_label)
                    pred = torch.argmax(val_score.data, dim=1)
                    val_acc += accuracy_score(pred.cpu(), val_label.cpu())
                    val_losses += val_loss

                    wrong_mask = pred != val_label
                    wrong_indices = torch.where(wrong_mask)[0]
                    for idx in wrong_indices:
                        feat_seq = val_feature[idx].cpu().tolist()
                        words = []
                        for wid in feat_seq:
                            if wid == 0:
                                continue
                            words.append(idx_to_word.get(wid, "<unk>"))
                        raw_sentence = " ".join(words)
                        error_cases.append({
                            "sentence": raw_sentence,
                            "true_label": val_label[idx].item(),
                            "pred_label": pred[idx].item()
                        })

            end = time.time()
            runtime = end - start
            pbar.set_postfix({
                'train loss': '%.4f' % (train_loss.data / n),
                'train acc': '%.2f' % (train_acc / n),
                'val loss': '%.4f' % (val_losses.data / m),
                'val acc': '%.2f' % (val_acc / m),
                'time': '%.2f' % runtime
            })

    print("\n==================== 3条预测错误案例（验证集）====================")
    show_cnt = min(3, len(error_cases))
    for i in range(show_cnt):
        case = error_cases[i]
        print(f"\n【案例{i+1}】")
        print(f"真实标签：{case['true_label']}，预测标签：{case['pred_label']}")
        print(f"文本内容：{case['sentence']}")

    test_pred = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.to(device)
                test_score = net(test_feature)
                test_pred.extend(torch.argmax(test_score.data, dim=1).cpu().numpy().tolist())
                pbar.update(1)

    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/lstm_3.csv", index=False, quoting=3)
    logging.info('result saved!')