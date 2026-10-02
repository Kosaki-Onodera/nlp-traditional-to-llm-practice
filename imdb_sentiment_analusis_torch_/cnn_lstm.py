import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ===================== 路径配置 =====================
TEST_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip"
RESULT_PATH = "/kaggle/working/result/cnnlstm_final.csv"
BEST_MODEL_PATH = "/kaggle/working/result/best_model.pth"
os.makedirs("/kaggle/working/result", exist_ok=True)

test = pd.read_csv(TEST_PATH, header=0, delimiter="\t", quoting=3)

# 超参微调
num_epochs = 15
max_len = 512
embed_size = 300
num_filter = 128
filter_sizes = [3, 4, 5]
pooling_size = 2
num_hiddens = 128
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 8e-4
dropout_rate = 0.35
grad_clip = 5.0
patience = 4
head_num = 4
warmup_epoch = 2  # 前2轮学习率预热
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        assert self.head_dim * heads == hidden_dim, "hidden_dim必须能被头数整除"

        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, lstm_out):
        seq_len, batch, _ = lstm_out.shape
        Q = self.w_q(lstm_out)
        K = self.w_k(lstm_out)
        V = self.w_v(lstm_out)

        Q = Q.view(seq_len, batch, self.heads, self.head_dim).permute(1, 2, 0, 3)
        K = K.view(seq_len, batch, self.heads, self.head_dim).permute(1, 2, 0, 3)
        V = V.view(seq_len, batch, self.heads, self.head_dim).permute(1, 2, 0, 3)

        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)
        attn_out = torch.matmul(attn_weight, V)

        attn_out = attn_out.permute(2, 0, 1, 3).contiguous()
        attn_out = attn_out.view(seq_len, batch, -1)
        attn_out = self.fc(attn_out)
        return attn_out

# 残差卷积块
class ResConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.ln = nn.LayerNorm(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=kernel//2)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        x = self.conv2(x)
        x = x + residual
        x = self.ln(x.transpose(1,2)).transpose(1,2)
        return F.relu(x)

class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_filter, filter_sizes, num_hiddens, num_layers, bidirectional, weight, heads, labels, dropout=0.3, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)

        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = True
        self.drop_emb = nn.Dropout(dropout * 0.6)

        # 多尺度CNN全局分支
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_size, num_filter, fs, padding=fs // 2)
            for fs in filter_sizes
        ])
        cnn_global_dim = num_filter * len(filter_sizes)

        # LSTM时序分支 + 残差卷积预处理
        self.conv_for_lstm = ResConv1d(embed_size, num_filter, kernel=3)
        lstm_in_dim = num_filter

        self.encoder = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        lstm_out_dim = num_hiddens * 2 if bidirectional else num_hiddens

        self.multi_attn = MultiHeadAttention(lstm_out_dim, heads)
        self.ln_attn = nn.LayerNorm(lstm_out_dim)

        self.dropout = nn.Dropout(dropout)
        fusion_dim = lstm_out_dim * 3 + cnn_global_dim
        # 多层融合头，替代单层Linear
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, labels)
        )

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.drop_emb(embeddings)
        emb_t = embeddings.permute(0, 2, 1)  # [B, E, L]

        # ----------多尺度CNN全局特征----------
        cnn_pool_outs = []
        for conv in self.convs:
            out = F.relu(conv(emb_t))
            pool_out = F.max_pool1d(out, kernel_size=out.size(-1)).squeeze(-1)
            cnn_pool_outs.append(pool_out)
        cnn_global_pool = torch.cat(cnn_pool_outs, dim=1)

        # ----------LSTM时序分支----------
        conv_seq = self.conv_for_lstm(emb_t)
        pooling = F.max_pool1d(conv_seq, kernel_size=pooling_size)
        lstm_in = pooling.permute(2, 0, 1)  # [seq, B, C]

        states, (h_n, _) = self.encoder(lstm_in)
        # 多头注意力 + 残差
        attn_raw = self.multi_attn(states)
        attn_seq = self.ln_attn(states + attn_raw)

        attn_feature = torch.mean(attn_seq, dim=0)
        max_seq_feature = torch.max(attn_seq, dim=0)[0]
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)

        # 融合：注意力均值 + 时序最大值 + LSTM双向末态 + CNN全局特征
        fusion = torch.cat([attn_feature, max_seq_feature, last_hidden, cnn_global_pool], dim=1)
        fusion = self.dropout(fusion)
        out = self.mlp(fusion)
        return out

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(f"running {''.join(sys.argv)}")

    logger.info('loading data...')
    pickle_file = os.path.join('/kaggle/working/pickle', 'imdb_glove.pickle3')
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word, vocab] = pickle.load(open(pickle_file, 'rb'))
    logger.info('data loaded!')

    net = SentimentNet(
        embed_size=embed_size,
        num_filter=num_filter,
        filter_sizes=filter_sizes,
        num_hiddens=num_hiddens,
        num_layers=num_layers,
        bidirectional=bidirectional,
        weight=weight,
        heads=head_num,
        labels=labels,
        dropout=dropout_rate
    )
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    embedding_params = list(map(id, net.embedding.parameters()))
    base_params = filter(lambda p: id(p) not in embedding_params, net.parameters())
    optimizer = optim.AdamW([
        {'params': net.embedding.parameters(), 'lr': lr * 0.3},
        {'params': base_params, 'lr': lr}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=1)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features,)

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    best_acc = 0.0
    stop_cnt = 0

    for epoch in range(num_epochs):
        # 学习率预热
        if epoch < warmup_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.2 + (lr * 0.8) * (epoch / warmup_epoch)

        start = time.time()
        net.train()
        total_loss = 0.0
        tr_preds, tr_labels = [], []

        for feature, label in tqdm(train_iter, desc=f"Epoch {epoch} Train"):
            feature, label = feature.to(device), label.to(device)
            score = net(feature)
            loss = loss_function(score, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            tr_preds.extend(torch.argmax(score.cpu(), dim=1).tolist())
            tr_labels.extend(label.cpu().tolist())

        train_acc = accuracy_score(tr_labels, tr_preds)

        net.eval()
        val_loss = 0.0
        va_preds, va_labels = [], []
        with torch.no_grad():
            for feat, lab in val_iter:
                feat, lab = feat.to(device), lab.to(device)
                s = net(feat)
                val_loss += loss_function(s, lab).item()
                va_preds.extend(torch.argmax(s.cpu(), dim=1).tolist())
                va_labels.extend(lab.cpu().tolist())
        val_acc = accuracy_score(va_labels, va_preds)
        scheduler.step(val_acc)

        logger.info(
            f"Epoch {epoch} | "
            f"Train Loss:{total_loss/len(train_iter):.4f} Acc:{train_acc:.4f} | "
            f"Val Loss:{val_loss/len(val_iter):.4f} Acc:{val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            stop_cnt = 0
            torch.save(net.state_dict(), BEST_MODEL_PATH)
            logger.info(f"Save Best Model, Best Val Acc:{best_acc:.4f}")
        else:
            stop_cnt += 1
            if stop_cnt >= patience:
                logger.info("Early Stop!")
                break

    net.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    net.eval()
    test_pred = []
    with torch.no_grad():
        for test_feature, in tqdm(test_iter, desc="Predicting"):
            test_feature = test_feature.to(device)
            test_score = net(test_feature)
            test_pred.extend(torch.argmax(test_score.cpu(), dim=1).numpy().tolist())

    result_output = pd.DataFrame({"id": test["id"], "sentiment": test_pred})
    result_output.to_csv(RESULT_PATH, index=False, quoting=3)
    logger.info("result saved!")
