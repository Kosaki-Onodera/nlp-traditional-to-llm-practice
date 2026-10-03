# -*- coding: utf-8 -*-
"""生成原理详解文档里用到的示意图（PNG）。运行：py -3.14 _make_figs.py"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
os.makedirs(OUT, exist_ok=True)

BLUE   = '#DCE9F7'; BLUE_E   = '#3A6EA5'
ORANGE = '#FBD9A8'; ORANGE_E = '#C97C0A'
PINK   = '#F7D3DC'; PINK_E   = '#C24E6B'
YELLOW = '#FBE9A8'; YELLOW_E = '#B8912A'
GREEN  = '#D8EED4'; GREEN_E  = '#4F8A45'
GREY   = '#EDEDED'; GREY_E   = '#9A9A9A'
PURPLE = '#E3DAF2'; PURPLE_E = '#6B4FA0'


def box(ax, x, y, w, h, text, fc=BLUE, ec=BLUE_E, fs=11, bold=False, lw=1.3, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.01,rounding_size=0.045",
                 fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, zorder=z + 1,
            fontweight='bold' if bold else 'normal')


def arrow(ax, p, q, color='#444444', lw=1.4, style='-|>', rad=0.0, ls='-', z=1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, color=color,
                 lw=lw, mutation_scale=13, zorder=z, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off')
    return fig, ax


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('written', name, os.path.getsize(path), 'bytes')


# ----------------------------------------------------------------------
# 图 1：BERT / GPT / ELMo 三种架构怎么看上下文
# ----------------------------------------------------------------------
def fig_three_arch():
    fig, ax = canvas(13.5, 4.8)
    toks = ['w1', 'w2', 'w3', 'w4', 'w5']
    panels = [
        (0.15, 'ELMo：双向 LSTM', 'elmo'),
        (4.60, 'GPT：单向 Transformer', 'gpt'),
        (9.05, 'BERT：双向 Transformer', 'bert'),
    ]
    PW, tw, th, gap = 4.30, 0.52, 0.44, 0.13
    for px, title, kind in panels:
        ax.add_patch(FancyBboxPatch((px, 0.15), PW, 4.35,
                     boxstyle="round,pad=0.02,rounding_size=0.07",
                     fc='#FCFCFC', ec='#C0C0C0', lw=1.2))
        ax.text(px + PW / 2, 4.22, title, ha='center', va='center',
                fontsize=12.5, fontweight='bold')
        total = 5 * tw + 4 * gap
        x0 = px + (PW - total) / 2
        y0 = 3.20
        for i, t in enumerate(toks):
            x = x0 + i * (tw + gap)
            hot = (i == 2)
            box(ax, x, y0, tw, th, t,
                fc=ORANGE if hot else BLUE, ec=ORANGE_E if hot else BLUE_E, fs=10)
        cx = x0 + 2 * (tw + gap) + tw / 2

        if kind == 'elmo':
            arrow(ax, (cx, y0), (cx, 2.72), color=ORANGE_E)
            box(ax, x0, 2.10, total, 0.56, '前向 LSTM　→→→→→',
                fc=GREEN, ec=GREEN_E, fs=10.5)
            box(ax, x0, 1.42, total, 0.56, '后向 LSTM　←←←←←',
                fc=GREEN, ec=GREEN_E, fs=10.5)
            box(ax, x0, 0.48, total, 0.62,
                '两个方向各算各的，算完再拼接',
                fc='#F5F5F5', ec=GREY_E, fs=10.5)
        elif kind == 'gpt':
            ny, r = 1.85, 0.20
            xs = [x0 + i * (tw + gap) + tw / 2 for i in range(5)]
            arrow(ax, (cx, y0), (cx, ny + r + 0.06), color=ORANGE_E)
            for i in range(4):
                arrow(ax, (xs[i] + r, ny), (xs[i + 1] - r, ny), color=BLUE_E, lw=1.6)
            for i, x in enumerate(xs):
                ax.add_patch(Circle((x, ny), r,
                             fc=ORANGE if i == 2 else BLUE,
                             ec=ORANGE_E if i == 2 else BLUE_E, lw=1.2, zorder=2))
            box(ax, x0, 0.48, total, 0.62,
                'w3 只能看到 w1、w2、w3\n（掩码挡住了它右边的词）',
                fc='#F5F5F5', ec=GREY_E, fs=10.5)
        else:
            ny, r = 1.60, 0.20
            xs = [x0 + i * (tw + gap) + tw / 2 for i in range(5)]
            arrow(ax, (cx, y0), (cx, ny + r + 0.06), color=ORANGE_E)
            for i, x in enumerate(xs):
                if i != 2:
                    arrow(ax, (cx, ny), (x, ny), color=PINK_E, lw=1.4, rad=0.30)
            for i, x in enumerate(xs):
                ax.add_patch(Circle((x, ny), r,
                             fc=ORANGE if i == 2 else PINK,
                             ec=ORANGE_E if i == 2 else PINK_E, lw=1.2, zorder=2))
            box(ax, x0, 0.48, total, 0.62,
                'w3 左右两边的词全都能看到\n（没有任何掩码）',
                fc='#F5F5F5', ec=GREY_E, fs=10.5)
    ax.text(6.75, 0.02, '橙色的 w3 是“当前正在算的那个词”，箭头表示它能参考到哪些位置',
            ha='center', va='bottom', fontsize=10.5, color='#666666')
    save(fig, 'arch-bert-gpt-elmo.png')


# ----------------------------------------------------------------------
# 图 2：BERT 的输入 = 三种 embedding 相加
# ----------------------------------------------------------------------
def fig_bert_input():
    fig, ax = canvas(13.3, 5.0)
    rows = [
        (3.60, 'Token Embedding\n（这是哪个词）',
         ['[CLS]', '我', '爱', '你', '[SEP]'], PINK, PINK_E),
        (2.80, 'Segment Embedding\n（属于第几句）',
         ['A', 'A', 'A', 'B', 'B'], BLUE, BLUE_E),
        (2.00, 'Position Embedding\n（排在第几位）',
         ['0', '1', '2', '3', '4'], GREEN, GREEN_E),
    ]
    lx, w, gap, h = 1.75, 0.85, 0.14, 0.55
    total = 5 * w + 4 * gap
    for y, label, cells, fc, ec in rows:
        ax.text(lx - 0.20, y + h / 2, label, ha='right', va='center',
                fontsize=9.5, color='#333333', linespacing=1.5)
        for i, s in enumerate(cells):
            box(ax, lx + i * (w + gap), y, w, h, s, fc=fc, ec=ec, fs=10.5)

    ax.text(lx + total + 0.55, 3.07, '逐元素\n相加\n=',
            ha='center', va='center', fontsize=12.5,
            fontweight='bold', color=PURPLE_E, linespacing=1.6)

    rx = lx + total + 1.30
    for i, t in enumerate(['[CLS]', '我', '爱', '你', '[SEP]']):
        box(ax, rx + i * (w + gap), 2.00, w, 2.15, t,
            fc=PURPLE, ec=PURPLE_E, fs=11, bold=True)

    cx = rx + total / 2
    ax.text(cx, 1.52, '送进 Encoder 的最终输入向量',
            ha='center', va='center', fontsize=11.5, fontweight='bold', color=PURPLE_E)
    ax.text(cx, 1.10, '每个词 = 词义 ＋ 属于哪句 ＋ 在第几位',
            ha='center', va='center', fontsize=10.5, color='#555555')

    ax.text(0.30, 0.35,
            '[CLS] 放在句首，它的输出专门拿来当整句摘要；[SEP] 用来分隔两个句子。\n'
            'RoBERTa 去掉了“属于第几句”这一整行（改用它自带的 <s> 和 </s>），所以 Segment 那一行全部是同一个值。',
            ha='left', va='center', fontsize=10.5, color='#444444', linespacing=1.7)
    save(fig, 'bert-input-embedding.png')


# ----------------------------------------------------------------------
# 图 3：BERT 的“内容+位置搅在一起” vs DeBERTa 的“解耦”
# ----------------------------------------------------------------------
def fig_deberta():
    fig, ax = canvas(12.9, 5.8)

    # ---------------- 左：BERT / RoBERTa ----------------
    ax.add_patch(FancyBboxPatch((0.15, 0.95), 5.70, 4.50,
                 boxstyle="round,pad=0.02,rounding_size=0.07",
                 fc='#FCFCFC', ec='#C0C0C0', lw=1.2))
    ax.text(3.00, 5.15, 'BERT / RoBERTa：先加成一个向量，再算注意力',
            ha='center', va='center', fontsize=11.5, fontweight='bold')
    box(ax, 0.60, 4.10, 1.65, 0.58, '词义 c', fc=PINK, ec=PINK_E, fs=11.5)
    box(ax, 0.60, 3.18, 1.65, 0.58, '位置 p', fc=GREEN, ec=GREEN_E, fs=11.5)
    box(ax, 2.75, 3.45, 1.55, 1.20, 'c ＋ p', fc=YELLOW, ec=YELLOW_E, fs=14, bold=True)
    arrow(ax, (2.25, 4.39), (2.73, 4.33), color=PINK_E)
    arrow(ax, (2.25, 3.47), (2.73, 3.76), color=GREEN_E)
    box(ax, 4.80, 3.70, 0.95, 1.00, 'Q\n\nK', fc=BLUE, ec=BLUE_E, fs=11.5, bold=True)
    arrow(ax, (4.30, 4.05), (4.78, 4.15), color=BLUE_E)
    box(ax, 2.35, 2.15, 3.40, 0.72, '注意力分数 = Q · K',
        fc=BLUE, ec=BLUE_E, fs=12.5, bold=True)
    arrow(ax, (5.28, 3.70), (4.72, 2.89), color=BLUE_E, rad=-0.1)
    ax.text(3.05, 1.45,
            '内容和位置已经混在同一个向量里，\n事后分不清“这一项里有多少是位置的贡献”',
            ha='center', va='center', fontsize=10.5, color=PINK_E, linespacing=1.6)

    # ---------------- 右：DeBERTa ----------------
    ax.add_patch(FancyBboxPatch((6.35, 0.95), 6.40, 4.50,
                 boxstyle="round,pad=0.02,rounding_size=0.07",
                 fc='#FCFCFC', ec='#C0C0C0', lw=1.2))
    ax.text(9.55, 5.15, 'DeBERTa：拆成两份，各算一份再相加',
            ha='center', va='center', fontsize=11.5, fontweight='bold')
    box(ax, 6.75, 4.10, 1.65, 0.58, '内容 c', fc=PINK, ec=PINK_E, fs=11.5)
    box(ax, 6.75, 3.18, 1.65, 0.58, '相对位置 P', fc=GREEN, ec=GREEN_E, fs=11.5)
    box(ax, 8.75, 4.28, 1.30, 0.52, 'Qc   Kc', fc=PINK, ec=PINK_E, fs=11)
    box(ax, 8.75, 3.36, 1.30, 0.52, 'Qr   Kr', fc=GREEN, ec=GREEN_E, fs=11)
    arrow(ax, (8.40, 4.39), (8.73, 4.55), color=PINK_E)
    arrow(ax, (8.40, 3.47), (8.73, 3.62), color=GREEN_E)

    terms = [('Qc · Kc', '内容对内容', PINK, PINK_E),
             ('Qc · Kr', '内容查位置', YELLOW, YELLOW_E),
             ('Qr · Kc', '位置查内容', YELLOW, YELLOW_E),
             ('Qr · Kr', '位置对位置（省略）', GREY, GREY_E)]
    for i, (t, sub, fc, ec) in enumerate(terms):
        y = 2.70 - i * 0.42
        box(ax, 8.55, y, 2.40, 0.36, t, fc=fc, ec=ec, fs=10.5)
        ax.text(8.45, y + 0.18, sub, ha='right', va='center',
                fontsize=9, color='#777777')
        arrow(ax, (10.95, y + 0.18), (11.08, y + 0.18), color='#AAAAAA', lw=1.1)
    ax.plot([11.10, 11.10], [1.62, 2.88], color='#AAAAAA', lw=1.2, zorder=1)
    box(ax, 11.25, 1.95, 1.45, 0.60, '注意力\n分数',
        fc=BLUE, ec=BLUE_E, fs=11, bold=True)
    arrow(ax, (11.10, 2.25), (11.23, 2.25))

    ax.text(0.30, 0.42,
            'BERT 把“词义”和“位置”加成同一个向量再算注意力，事后再也分不出哪部分来自位置；'
            'DeBERTa 让内容算一份、相对位置算一份，最后相加，四项各自说得清。\n'
            '另外，DeBERTa 的位置编码的是两个词“相隔多远”，不是“排在第几号”，所以换个位置顺序也照样成立。',
            ha='left', va='center', fontsize=10.5, color='#444444', linespacing=1.8)
    save(fig, 'deberta-disentangled-attention.png')


# ----------------------------------------------------------------------
# 图 4：LoRA —— 冻结 W0，只训 B·A
# ----------------------------------------------------------------------
def fig_lora():
    fig, ax = canvas(12.0, 4.9)

    box(ax, 0.35, 2.85, 0.95, 0.62, 'x', fc=GREY, ec=GREY_E, fs=13, bold=True)

    box(ax, 2.05, 2.72, 2.55, 0.88,
        'W0   d×k\n冻结，不更新', fc=BLUE, ec=BLUE_E, fs=12)
    arrow(ax, (1.30, 3.16), (2.03, 3.16))

    box(ax, 2.05, 1.35, 1.28, 0.72, 'A\nr×k', fc=ORANGE, ec=ORANGE_E, fs=12)
    box(ax, 3.55, 1.35, 1.28, 0.72, 'B\nd×r', fc=ORANGE, ec=ORANGE_E, fs=12)
    arrow(ax, (1.30, 2.90), (1.62, 2.90), color='#444444', style='-')
    arrow(ax, (1.62, 2.90), (1.62, 1.71), color='#444444', style='-')
    arrow(ax, (1.62, 1.71), (2.03, 1.71), color=ORANGE_E)
    arrow(ax, (3.33, 1.71), (3.53, 1.71), color=ORANGE_E)

    ax.add_patch(Circle((5.30, 3.16), 0.24, fc='white', ec='#444444', lw=1.4, zorder=2))
    ax.text(5.30, 3.16, '+', ha='center', va='center', fontsize=15,
            fontweight='bold', zorder=3)
    arrow(ax, (4.60, 3.16), (5.05, 3.16))
    arrow(ax, (4.83, 1.71), (5.30, 1.71), color=ORANGE_E)
    arrow(ax, (5.30, 1.95), (5.30, 2.91), color=ORANGE_E)

    box(ax, 6.45, 2.72, 1.55, 0.88, 'h', fc=PURPLE, ec=PURPLE_E, fs=13, bold=True)
    arrow(ax, (5.54, 3.16), (6.43, 3.16))

    ax.text(6.35, 3.90, 'h = W0·x  +  B·A·x', ha='center', va='center',
            fontsize=13.5, fontweight='bold', color='#333333')
    ax.text(8.55, 2.30,
            '只训练 A、B 这两个小矩阵\n'
            '其余 1.5B 的权重全部冻结\n\n'
            'r = 16，d = 1536\n'
            '单个模块参数量：\n'
            '  1536×1536  →  1536×16 + 16×1536\n'
            '  2,359,296  →  49,152（约 2.1%）\n\n'
            'B 初始化为全零 → 训练开始时\n'
            'B·A = 0，模型输出和原模型完全一致，\n'
            '等于从“已经很好的起点”出发',
            ha='left', va='center', fontsize=10.5, color='#333333', linespacing=1.6)

    ax.text(3.45, 0.62,
            '前向：x 同时走两条路，输出相加。反向：梯度只更新 A、B，不碰 W0。',
            ha='center', va='center', fontsize=11, color='#666666')
    save(fig, 'lora-reparametrization.png')


# ----------------------------------------------------------------------
# 图 5：做文本分类时，Transformer 那张图里只用到左半边
# ----------------------------------------------------------------------
def fig_cls_flow():
    fig, ax = canvas(10.5, 8.4)
    x, w = 1.05, 4.30
    cxc = x + w / 2
    box(ax, x, 7.55, w, 0.55, '输入文本　"I love this movie"', fc=GREY, ec=GREY_E, fs=11.5)
    for i, (t, fc, ec) in enumerate([
            ('Tokenizer 分词 → 整数 ID', PINK, PINK_E),
            ('Embedding 词嵌入（查表换向量）', PINK, PINK_E),
            ('＋ 位置编码 Positional Encoding', GREEN, GREEN_E)]):
        y = 6.70 - i * 0.85
        box(ax, x, y, w, 0.55, t, fc=fc, ec=ec, fs=11.5)
        arrow(ax, (cxc, y + 0.85), (cxc, y + 0.57))
    # Encoder 大框
    by, bh = 2.90, 1.85
    ax.add_patch(FancyBboxPatch((x - 0.18, by), w + 0.36, bh,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc='#F2F6FB', ec=BLUE_E, lw=1.6, zorder=1))
    ax.text(x + 0.02, by + bh - 0.13, 'Encoder × N', ha='left', va='center',
            fontsize=11.5, fontweight='bold', color=BLUE_E, zorder=3)
    for i, (t, fc, ec) in enumerate([
            ('Multi-Head Self-Attention', ORANGE, ORANGE_E),
            ('Add & Norm（残差 ＋ 层归一化）', YELLOW, YELLOW_E),
            ('Feed Forward（两层全连接）', BLUE, BLUE_E),
            ('Add & Norm', YELLOW, YELLOW_E)]):
        box(ax, x + 0.15, 4.05 - i * 0.35, w - 0.30, 0.28, t,
            fc=fc, ec=ec, fs=10, z=3)
    arrow(ax, (cxc, 5.00), (cxc, by + bh + 0.02))
    box(ax, x, 2.05, w, 0.55, '取 [CLS] 位置的向量当整句表示',
        fc=PURPLE, ec=PURPLE_E, fs=11.5)
    arrow(ax, (cxc, by), (cxc, 2.62))
    box(ax, x, 1.20, w, 0.55, 'Dropout → Linear（输出 2 维）',
        fc=PURPLE, ec=PURPLE_E, fs=11.5)
    arrow(ax, (cxc, 2.05), (cxc, 1.77))
    box(ax, x, 0.35, w, 0.55, '正面 / 负面', fc=GREEN, ec=GREEN_E, fs=13, bold=True)
    arrow(ax, (cxc, 1.20), (cxc, 0.92))

    # 右侧说明
    px, pw = 6.00, 4.35
    ax.add_patch(FancyBboxPatch((px, 2.05), pw, 6.05,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc='#FFF8F0', ec=ORANGE_E, lw=1.3))
    ax.text(px + pw / 2, 7.78, '原论文那张图里', ha='center', va='center',
            fontsize=12, fontweight='bold', color=ORANGE_E)
    ax.text(px + 0.30, 5.95,
            '左半边（Encoder）\n'
            '   → 本实验完整用到\n\n'
            '右半边（Decoder）\n'
            '   → 完全不用\n'
            '      （那是做翻译、生成用的）\n\n'
            '顶部 Linear ＋ Softmax\n'
            '   → 换成 2 维分类头\n'
            '      （不是词表维度）',
            ha='left', va='center', fontsize=11, color='#333333', linespacing=1.8)
    box(ax, px + 0.30, 2.45, pw - 0.60, 0.95,
        '所以本实验的模型\n＝ 这张图的左半边 ＋ 换掉的分类头',
        fc='#FFFFFF', ec=ORANGE_E, fs=11)
    save(fig, 'transformer-classification-flow.png')


# ----------------------------------------------------------------------
# 图 6：自注意力的矩阵形状怎么变（3 个词 × 4 维的小例子）
# ----------------------------------------------------------------------
def fig_matrix_flow():
    fig, ax = canvas(13.5, 5.3)
    cy = 3.00
    cell = 0.22

    def mat(cx, rows, cols, fc, ec):
        w, h = cols * cell, rows * cell
        x0, y0 = cx - w / 2, cy - h / 2
        for r in range(rows):
            for c in range(cols):
                ax.add_patch(Rectangle((x0 + c * cell, y0 + r * cell), cell, cell,
                                       fc=fc, ec=ec, lw=0.9, zorder=3))
        return cx - w / 2, cx + w / 2

    for cx, rows, cols, fc, ec, name, shape, note in [
        (1.60, 3, 4, PINK, PINK_E, '① 输入 X', '3 个词 × 4 维', '一行是一个词'),
        (4.00, 3, 4, ORANGE, ORANGE_E, '② Q / K / V', '3 × 4，一共三个', '分别乘 Wq / Wk / Wv'),
        (6.90, 3, 3, YELLOW, YELLOW_E, '③ 分数 S = Q·K^T', '3 × 3', '变成「词 × 词」'),
        (9.40, 3, 3, YELLOW, YELLOW_E, '④ 权重 A = softmax', '3 × 3', '每一行加起来 = 1'),
        (12.00, 3, 4, GREEN, GREEN_E, '⑤ 输出 Z = A·V', '3 × 4', '形状又回来了'),
    ]:
        l, r = mat(cx, rows, cols, fc, ec)
        ax.text(cx, cy + 0.62, name, ha='center', va='center', fontsize=10.5,
                fontweight='bold', color='#222222')
        ax.text(cx, cy - 0.55, shape, ha='center', va='center', fontsize=9.5,
                color='#333333')
        ax.text(cx, cy - 0.82, note, ha='center', va='center', fontsize=9,
                color='#666666')

    for x0, x1 in [(2.06, 3.54), (4.46, 6.55), (7.25, 9.05), (9.75, 11.54)]:
        arrow(ax, (x0 + 0.03, cy), (x1 - 0.03, cy), lw=1.6)

    ax.text(6.75, 4.45, '自注意力：矩阵是怎么一层层变的（示例：3 个词，每个 4 维）',
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#222222')

    ax.add_patch(FancyBboxPatch((0.55, 0.42), 12.40, 0.92,
                 boxstyle="round,pad=0.01,rounding_size=0.06",
                 fc='#FFF6E8', ec=ORANGE_E, lw=1.4, zorder=2))
    ax.text(6.75, 0.88,
            '整条链走完，形状从 3 × 4 出发又回到 3 × 4 —— 中间只有 ③ 短暂变成「词 × 词」。',
            ha='center', va='center', fontsize=10.5, color='#222222')
    ax.text(6.75, 0.62,
            '正因如此，编码器里每一层的进和出形状完全一样，才能一层叠一层地堆 N 层。',
            ha='center', va='center', fontsize=10.5, color='#222222')
    save(fig, 'attention-matrix-flow.png')


# ----------------------------------------------------------------------
# 图 7：两种掩码盖在分数矩阵上是什么样
# ----------------------------------------------------------------------
def fig_mask():
    fig, ax = canvas(12.6, 5.7)
    cell = 0.52
    KEEP, MASK = '#CFE6C7', '#DCDCDC'
    KEEP_E, MASK_E = '#4F8A45', '#8C8C8C'

    def grid(cx, cy, tokens, keep, title, cap1, cap2):
        n = len(tokens)
        x0, y0 = cx - n * cell / 2, cy - n * cell / 2
        for i in range(n):
            yc = y0 + (n - 1 - i) * cell
            ax.text(x0 - 0.14, yc + cell / 2, tokens[i], ha='right', va='center',
                    fontsize=11.5, color='#222222')
            for j in range(n):
                xc = x0 + j * cell
                ok = keep[i][j]
                ax.add_patch(Rectangle((xc, yc), cell, cell,
                             fc=KEEP if ok else MASK, ec=KEEP_E if ok else MASK_E,
                             lw=1.2, zorder=3))
                if not ok:
                    ax.text(xc + cell / 2, yc + cell / 2, '×', ha='center', va='center',
                            fontsize=15, color='#8C8C8C', zorder=4)
        for j in range(n):
            ax.text(x0 + j * cell + cell / 2, y0 + n * cell + 0.20, tokens[j],
                    ha='center', va='center', fontsize=11.5, color='#222222')
        ax.text(cx, cy + n * cell / 2 + 0.72, title, ha='center', va='center',
                fontsize=12, fontweight='bold', color='#222222')
        ax.text(cx, cy - n * cell / 2 - 0.42, cap1, ha='center', va='center',
                fontsize=10, color='#333333')
        ax.text(cx, cy - n * cell / 2 - 0.72, cap2, ha='center', va='center',
                fontsize=10, color='#333333')

    toks = ['我', '爱', '电影']
    grid(2.75, 3.05, toks,
         [[j <= i for j in range(3)] for i in range(3)],
         '① 因果掩码：右上角全遮掉',
         '每个位置只留「自己」和「左边」。',
         '所以第 1 行只剩「我」，第 3 行一个都没少。')

    ptoks = ['好', 'PAD', 'PAD']
    grid(9.35, 3.05, ptoks,
         [[ptoks[i] != 'PAD' and ptoks[j] != 'PAD' for j in range(3)] for i in range(3)],
         '② padding 掩码：和补齐位有关的一律遮掉',
         '第 2、3 个位置是补出来的 [PAD]，不是真词。',
         '第 2、3 行整行被遮 —— 它们本来也没有可看的位置。')

    ax.add_patch(Rectangle((4.85, 0.62), 0.28, 0.28, fc=KEEP, ec=KEEP_E, lw=1.2))
    ax.text(5.22, 0.76, '能看', ha='left', va='center', fontsize=10.5, color='#333333')
    ax.add_patch(Rectangle((6.20, 0.62), 0.28, 0.28, fc=MASK, ec=MASK_E, lw=1.2))
    ax.text(6.57, 0.76, '遮掉（分数改成负无穷，softmax 后正好是 0）',
            ha='left', va='center', fontsize=10.5, color='#333333')

    save(fig, 'attention-mask.png')


if __name__ == '__main__':
    fig_three_arch()
    fig_bert_input()
    fig_deberta()
    fig_lora()
    fig_cls_flow()
    fig_matrix_flow()
    fig_mask()
    print('done')
