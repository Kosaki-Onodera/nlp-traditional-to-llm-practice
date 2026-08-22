import logging
import os
import sys
import pickle
import time
import numpy as np

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

# ===================== Kaggle环境超参配置 =====================
num_epochs = 15
embed_size = 300
num_filter = 128
filter_sizes = [3, 4, 5]
dropout_rate = 0.4
batch_size = 64
lr = 0.001
freeze_epoch = 4       # 前4轮冻结GloVe
weight_decay = 1e-4
patience = 3
# ==============================================================

# 自动规避P100(sm_60)兼容问题
def get_safe_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        if cap[0] == 6:
            logging.warning(f"检测到 {gpu_name}(sm_60)，当前PyTorch不兼容，自动切换至CPU运行")
            return torch.device("cpu")
        else:
            logging.info(f"使用GPU: {gpu_name}")
            return torch.device('cuda:0')
    else:
        return torch.device("cpu")

device = get_safe_device()
use_gpu = device.type == "cuda"

# ===================== TextCNN网络 =====================
class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_filter, filter_sizes, dropout_rate, weight, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.use_gpu = use_gpu
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False  # 初始冻结，训练代码动态切换

        self.dropout = nn.Dropout(dropout_rate)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_size, num_filter, k) for k in filter_sizes
        ])
        self.decoder = nn.Linear(num_filter * len(filter_sizes), 1)  # 输出1维logits

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.dropout(embeddings)          # dropout放在embedding之后
        x = embeddings.permute([0, 2, 1])              # [B, seq_len, dim] -> [B, dim, seq_len]
        conv_out = [F.relu(conv(x)) for conv in self.convs]
        pool_out = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in conv_out]
        concat = torch.cat(pool_out, dim=1)
        outputs = self.decoder(concat)
        return outputs

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(f"running {' '.join(sys.argv)}")

    logging.info('loading data...')
    pickle_file = "/kaggle/working/pickle/imdb_glove.pickle3"
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"请先运行预处理代码生成文件：{pickle_file}")
    with open(pickle_file, 'rb') as f:
        [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word, vocab] = pickle.load(f)
    logging.info('data loaded!')

    net = SentimentNet(embed_size=embed_size,
                       num_filter=num_filter,
                       filter_sizes=filter_sizes,
                       dropout_rate=dropout_rate,
                       weight=weight,
                       use_gpu=use_gpu)
    net.to(device)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # 基于验证AUC衰减学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # ------------------- 早停配置（以AUC为指标） -------------------
    best_val_auc = 0.0
    trigger_times = 0
    ckpt_path = "/kaggle/working/best_cnn.pth"
    error_samples = []

    for epoch in range(num_epochs):
        start = time.time()
        train_loss = 0.0
        n = 0
        net.train()

        # 两阶段控制embedding是否可训练
        if epoch < freeze_epoch:
            for param in net.embedding.parameters():
                param.requires_grad = False
        else:
            for param in net.embedding.parameters():
                param.requires_grad = True

        with tqdm(total=len(train_iter), desc=f'Epoch {epoch} Train') as pbar:
            for feature, label in train_iter:
                n += 1
                optimizer.zero_grad()
                feature = feature.to(device)
                label = label.to(device)
                score = net(feature)
                loss = loss_function(score, label.unsqueeze(1).float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{train_loss / n:.4f}'})
                pbar.update(1)

        # 验证阶段：计算AUC + Acc
        val_losses = 0.0
        val_y_true = []
        val_y_prob = []
        temp_errors = []
        m = 0
        net.eval()
        with torch.no_grad():
            with tqdm(total=len(val_iter), desc=f'Epoch {epoch} Val') as pbar:
                for val_feature, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.to(device)
                    val_label = val_label.to(device)
                    val_score = net(val_feature)
                    val_loss = loss_function(val_score, val_label.unsqueeze(1).float())
                    val_losses += val_loss.item()

                    prob = torch.sigmoid(val_score).squeeze(1).cpu()
                    pred = (prob > 0.5).long()
                    true = val_label.cpu()

                    val_y_true.extend(true.numpy().tolist())
                    val_y_prob.extend(prob.numpy().tolist())

                    # 收集错误样本
                    wrong_mask = (pred != true)
                    wrong_idx = torch.nonzero(wrong_mask).squeeze(1)
                    for idx in wrong_idx:
                        seq_ids = val_feature[idx].cpu().tolist()
                        t = true[idx].item()
                        p = pred[idx].item()
                        temp_errors.append({
                            "seq_ids": seq_ids,
                            "true_label": t,
                            "pred_label": p
                        })
                    pbar.update(1)

        end = time.time()
        runtime = end - start
        current_val_loss = val_losses / m
        current_val_auc = roc_auc_score(val_y_true, val_y_prob)
        current_val_acc = accuracy_score(val_y_true, (np.array(val_y_prob) > 0.5))

        logging.info(
            f"Epoch {epoch} | Train Loss:{train_loss/n:.4f} "
            f"Val Loss:{current_val_loss:.4f} Val Acc:{current_val_acc:.4f} Val AUC:{current_val_auc:.4f} Time:{runtime:.2f}s"
        )

        # 早停 & 保存最优模型（依据AUC）
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            trigger_times = 0
            torch.save(net.state_dict(), ckpt_path)
            error_samples = temp_errors.copy()
            logging.info(f"Save best model, val auc = {best_val_auc:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logging.info(f"Early Stop! Best val AUC:{best_val_auc:.4f}")
                break
        scheduler.step(current_val_auc)

    # ----------------【输出3条错误案例，用于错误分析】----------------
    logging.info("\n========== 验证集错误样例（最多打印3条） ==========")
    show_num = min(3, len(error_samples))
    for i in range(show_num):
        item = error_samples[i]
        ids_list = item["seq_ids"]
        words = [idx_to_word[idx] for idx in ids_list if idx != 0]
        print(f"\n【错误案例{i+1}】")
        print(f"真实标签：{item['true_label']}，预测标签：{item['pred_label']}")
        print(f"文本：{' '.join(words)}")

    # ---------------- 预测阶段：加载最优权重 ----------------
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    test_pred_prob = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.to(device)
                test_score = net(test_feature)
                prob = torch.sigmoid(test_score).squeeze(1).cpu().numpy().tolist()
                test_pred_prob.extend(prob)
                pbar.update(1)

    # 生成Kaggle提交csv（阈值0.5转为0/1）
    TEST_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip"
    test = pd.read_csv(TEST_PATH, header=0, delimiter="\t", quoting=3)
    test["sentiment"] = (np.array(test_pred_prob) > 0.5).astype(int)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test["sentiment"]})
    result_output.to_csv("/kaggle/working/cnn.csv", index=False, quoting=3)
    logging.info('result saved!')
