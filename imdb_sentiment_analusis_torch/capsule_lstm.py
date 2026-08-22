import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ===================== Kaggle路径 =====================
PICKLE_DIR = "/kaggle/working/pickle"
PICKLE_FILE = os.path.join(PICKLE_DIR, "imdb_glove.pickle3")
TEST_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip"
OUTPUT_DIR = "/kaggle/working/result"
MODEL_DIR = "/kaggle/working/model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

test = pd.read_csv(TEST_PATH, header=0, delimiter="\t", quoting=3)

# ===================== 超参调优 =====================
num_epochs = 15
embed_size = 300
num_hiddens = 128
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.001
weight_decay = 1e-5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 胶囊超参
num_capsule = 8
dim_capsule = 32
routings = 3

# 早停
patience = 3

class Capsule(nn.Module):
    def __init__(self, input_dim, num_capsule, dim_capsule, routings, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.W = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(1, input_dim, num_capsule * dim_capsule)))

    def forward(self, inputs):
        # inputs: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = inputs.shape
        u_hat_vecs = torch.matmul(inputs, self.W)
        u_hat_vecs = u_hat_vecs.view(batch_size, seq_len, self.num_capsule, self.dim_capsule)
        u_hat_vecs = u_hat_vecs.permute(0,2,1,3).contiguous() # B,num_caps,seq,dim

        with torch.no_grad():
            b = torch.zeros_like(u_hat_vecs[:, :, :, 0])

        for i in range(self.routings):
            c = F.softmax(b, dim=1)
            outputs = self.squash(torch.sum(c.unsqueeze(-1)*u_hat_vecs, dim=2))
            if i < self.routings - 1:
                b = b + torch.sum(outputs.unsqueeze(2)*u_hat_vecs, dim=-1)
        return outputs

    @staticmethod
    def squash(x, axis=-1):
        s_squared_norm = (x**2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-7)
        return x / scale


class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers, bidirectional, weight, labels,
                 num_capsule, dim_capsule, routings, dropout=0.3,**kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        lstm_out_dim = num_hiddens * 2 if bidirectional else num_hiddens

        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = True
        self.drop_emb = nn.Dropout(dropout)

        self.encoder = nn.LSTM(input_size=self.embed_size, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=dropout, batch_first=False)

        self.capsule = Capsule(input_dim=lstm_out_dim, num_capsule=num_capsule,
                               dim_capsule=dim_capsule, routings=routings)

        self.drop_fc = nn.Dropout(dropout)
        cap_dim = num_capsule * dim_capsule
        self.decoder = nn.Linear(cap_dim + lstm_out_dim*2, labels)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        emb = self.drop_emb(emb)
        states, _ = self.encoder(emb.permute(1,0,2))
        seq_out = states.permute(1,0,2).contiguous()
        cap_out = self.capsule(seq_out)
        cap_flat = cap_out.flatten(start_dim=1)

        avg_pool = torch.mean(seq_out, dim=1)
        max_pool, _ = torch.max(seq_out, dim=1)
        feat = torch.cat([cap_flat, avg_pool, max_pool], dim=1)
        feat = self.drop_fc(feat)
        logits = self.decoder(feat)
        return logits


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
     vocab] = pickle.load(open(PICKLE_FILE, 'rb'))
    logging.info('data loaded!')

    net = SentimentNet(embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers,
                       bidirectional=bidirectional, weight=weight, labels=labels,
                       num_capsule=num_capsule, dim_capsule=dim_capsule, routings=routings, dropout=0.3)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    # 修复：移除verbose=True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_acc = 0.0
    best_epoch = 0
    early_stop_cnt = 0
    best_model_path = os.path.join(MODEL_DIR, "best_capsule_lstm.pt")

    for epoch in range(num_epochs):
        start = time.time()
        train_loss = 0.0
        train_acc = 0.0
        n = 0
        net.train()
        with tqdm(total=len(train_iter), desc=f"Epoch {epoch}") as pbar:
            for feature, label in train_iter:
                n += 1
                optimizer.zero_grad()
                feature = feature.to(device)
                label = label.to(device)
                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()

                train_acc += accuracy_score(torch.argmax(score.cpu(), dim=1), label.cpu())
                train_loss += loss.item()
                pbar.set_postfix({"train_loss":f"{train_loss/n:.4f}", "train_acc":f"{train_acc/n:.4f}"})
                pbar.update(1)

        net.eval()
        val_loss = 0.0
        val_acc = 0.0
        m = 0
        with torch.no_grad():
            for val_feature, val_label in val_iter:
                m +=1
                val_feature = val_feature.to(device)
                val_label = val_label.to(device)
                val_score = net(val_feature)
                loss_v = loss_function(val_score, val_label)
                val_loss += loss_v.item()
                val_acc += accuracy_score(torch.argmax(val_score.cpu(),dim=1), val_label.cpu())
        val_acc_avg = val_acc/m
        train_acc_avg = train_acc/n
        train_loss_avg = train_loss/n
        val_loss_avg = val_loss/m
        runtime = time.time()-start

        print(f"Epoch:{epoch} | TrainLoss:{train_loss_avg:.4f} TrainAcc:{train_acc_avg:.4f} | ValLoss:{val_loss_avg:.4f} ValAcc:{val_acc_avg:.4f} | time:{runtime:.2f}s | lr:{optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(val_acc_avg)

        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            best_epoch = epoch
            early_stop_cnt = 0
            torch.save(net.state_dict(), best_model_path)
            print(f"★保存最优模型, val_acc={best_val_acc:.4f}")
        else:
            early_stop_cnt +=1
            if early_stop_cnt >= patience:
                print(f"早停触发，epoch={epoch}, best_val_acc={best_val_acc:.4f} @epoch{best_epoch}")
                break

    print(f"加载最优模型做预测，best val acc:{best_val_acc:.4f}")
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    net.eval()
    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc="Predicting") as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.to(device)
                test_score = net(test_feature)
                test_pred.extend(torch.argmax(test_score.cpu(), dim=1).numpy().tolist())
                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    save_path = os.path.join(OUTPUT_DIR, "capsule_lstm(1).csv")
    result_output.to_csv(save_path, index=False, quoting=3)
    logging.info(f'预测结果保存 {save_path}')
