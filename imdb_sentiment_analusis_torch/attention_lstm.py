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

# =================== 超参 ===================
num_epochs = 12
embed_size = 300
num_hiddens = 256
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 1e-3
weight_decay = 1e-4
grad_clip = 5.0
dropout_rate = 0.4
patience = 3
warmup_epochs = 1

# =================== 路径 ===================
PICKLE_PATH = "/kaggle/working/pickle/imdb_glove.pickle3"
TEST_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip"
RESULT_PATH = "/kaggle/working/result/attention_lstm_best.csv"
BEST_MODEL_PATH = "/kaggle/working/result/best_model.pth"

os.makedirs("/kaggle/working/result", exist_ok=True)

# =================== 设备 ===================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()

print(f"Using device: {device}")

# =================== Attention ===================
class Attention(nn.Module):
    def __init__(self, hidden_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output, mask=None):
        seq_len, batch, _ = lstm_output.shape
        attn_weights = torch.tanh(self.attn(lstm_output))
        attn_scores = self.v(attn_weights).squeeze(-1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.T, -1e4)

        attn_dist = F.softmax(attn_scores, dim=0)
        weighted = lstm_output * attn_dist.unsqueeze(-1)
        output = torch.sum(weighted, dim=0)

        return output, attn_dist

class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, dropout=0.3, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding.from_pretrained(weight, padding_idx=0)
        self.embedding.weight.requires_grad = True

        self.encoder = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.num_hiddens,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )

        lstm_out_dim = num_hiddens * 2 if bidirectional else num_hiddens
        self.attention = Attention(lstm_out_dim)

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(lstm_out_dim),
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, labels)
        )

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.permute(1, 0, 2)

        pad_mask = (inputs == 0)

        lstm_out, _ = self.encoder(embeddings)
        attn_pool, _ = self.attention(lstm_out, pad_mask)

        outputs = self.decoder(attn_pool)

        return outputs

# =================== 学习率调度器 ===================
def get_warmup_scheduler(optimizer, warmup_epoch, total_epoch):
    def lr_lambda(epoch):
        if epoch < warmup_epoch:
            return (epoch + 1) / warmup_epoch
        else:
            progress = (epoch - warmup_epoch) / (total_epoch - warmup_epoch)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =================== 主训练 ===================
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO
    )
    logger.info(r"running %s" % ''.join(sys.argv))

    logger.info('loading pickle data...')
    [
        train_features, train_labels,
        val_features, val_labels,
        test_features, weight,
        word_to_idx, idx_to_word, vocab
    ] = pickle.load(open(PICKLE_PATH, 'rb'))
    logger.info('pickle data loaded!')

    test = pd.read_csv(TEST_PATH, header=0, delimiter="\t", quoting=3)

    net = SentimentNet(
        embed_size=embed_size,
        num_hiddens=num_hiddens,
        num_layers=num_layers,
        bidirectional=bidirectional,
        weight=weight,
        labels=labels,
        dropout=dropout_rate
    )
    net.to(device)

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_warmup_scheduler(optimizer, warmup_epochs, num_epochs)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_acc = 0.0
    trigger_times = 0

    for epoch in range(num_epochs):
        start = time.time()

        train_loss = 0.0
        train_acc = 0.0
        n = 0

        net.train()
        with tqdm(total=len(train_iter), desc=f'Epoch {epoch}') as pbar:
            for feature, label in train_iter:
                n += 1
                feature = feature.to(device)
                label = label.to(device)

                score = net(feature)
                loss = loss_function(score, label)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
                optimizer.step()

                train_acc += accuracy_score(
                    torch.argmax(score.cpu().data, dim=1),
                    label.cpu()
                )
                train_loss += loss.item()

                pbar.set_postfix({
                    'train loss': '%.4f' % (train_loss / n),
                    'train acc': '%.4f' % (train_acc / n)
                })
                pbar.update(1)

        val_losses = 0.0
        val_acc = 0.0
        m = 0

        net.eval()
        with torch.no_grad():
            for val_feature, val_label in val_iter:
                m += 1
                val_feature = val_feature.to(device)
                val_label = val_label.to(device)

                val_score = net(val_feature)
                val_loss = loss_function(val_score, val_label)

                val_acc += accuracy_score(
                    torch.argmax(val_score.cpu().data, dim=1),
                    val_label.cpu()
                )
                val_losses += val_loss.item()

        scheduler.step()

        end = time.time()
        runtime = end - start

        epoch_train_acc = train_acc / n
        epoch_val_acc = val_acc / m

        logger.info(
            "Epoch %d: train_loss %.4f, train_acc %.4f, val_loss %.4f, val_acc %.4f, time %.2f, lr=%.6f"
            % (
                epoch,
                train_loss / n,
                epoch_train_acc,
                val_losses / m,
                epoch_val_acc,
                runtime,
                optimizer.param_groups[0]['lr']
            )
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            trigger_times = 0
            torch.save(net.state_dict(), BEST_MODEL_PATH)
            logger.info(f"Save best model! Best val acc: {best_val_acc:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logger.info(f"Early stop at epoch {epoch}!")
                break

    logger.info('load best model for prediction...')
    net.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    logger.info('start prediction...')
    test_pred = []

    net.eval()
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.to(device)
                test_score = net(test_feature)
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    result_output = pd.DataFrame({
        "id": test["id"],
        "sentiment": test_pred
    })
    result_output.to_csv(RESULT_PATH, index=False, quoting=3)

    logger.info(f'result saved to {RESULT_PATH}')
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
