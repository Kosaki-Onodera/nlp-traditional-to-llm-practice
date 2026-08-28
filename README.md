# nlp-traditional-to-llm-practice
本项目基于IMDB电影评论情感二分类任务，系统梳理NLP表征学习与模型架构的演进脉络，完整演示词袋模型 → Word2Vec → GloVe预训练词向量 + 深度学习网络 → Transformer → BERT预训练微调的技术路线，覆盖从静态词嵌入到上下文动态表征的关键跨越，为理解大语言模型（LLM）的底层机制奠定基础。

## 静态预训练词向量模型
静态预训练词向量的核心作用，就是把文本里的每一个单词，转换成计算机能看懂的一组小数向量。  例如 apple:[0.56,0.48,0.24...]  cat:[0.19,0.42,0.96...]     
相比传统词袋模型只统计单词出现次数、不区分语义的问题，Word2Vec和GloVe可以让语义相似的单词，在向量空间里距离更近，更好地表达词语含义。之所以叫静态，是因为每个单词训练完成后只有唯一固定的向量，不会根据句子的上下文发生变化，无法区分一词多义。  

### Word2Vec  
Word2Vec是轻量级的词向量训练模型，核心训练逻辑依托文本的局部滑动窗口，遵循“上下文相近的单词，语义大概率相似”的核心思想。模型不依赖复杂的全局统计数据，仅通过句子内局部词语的搭配规律学习语义，主要分为两种训练模式，适配不同的训练场景：  
- CBOW：通过周围的上下文单词，预测中间的中心单词。CBOW（连续词袋模型）：核心是“上下文预测中心词”。设定固定大小的文本滑动窗口，选取窗口内中心单词的前后多个上下文单词，将这些上下文单词的向量整合计算，通过模型推理预测出中间的中心单词。该模式训练速度快，擅长学习文本中的高频通用词汇，适合大规模数据集训练。  
- Skip-Gram：通过中间的中心单词，预测周围的上下文单词。Skip-Gram模型：核心是“中心词预测上下文”。以滑动窗口内的单个中心单词作为输入，让模型预测该单词周边出现的所有上下文单词。相比于CBOW，该模式对低频、小众词汇的表征效果更好，能够更好挖掘少量出现的词语语义细节。
 
训练完成后，每个单词就会得到一个专属的稠密向量，能够简单表达词语的语义关系，弥补了词袋模型没有语义信息的缺点，但它只关注局部窗口内的词语关系，没有利用整个数据集的全局统计信息。两种训练方式的最终目的都是通过词语的搭配规律学习语义，训练完成后，模型的权重参数就对应每个单词的专属稠密向量。相较于词袋模型，彻底解决了无语义信息、维度稀疏的问题，但存在明显短板：仅聚焦单个滑动窗口内的局部词语关联，没有整合整个数据集的全局词语共现规律，对整体语料的语义挖掘不够全面。  

### Glove  
GloVe（全局向量模型）是针对Word2Vec短板优化升级的词向量模型，完美融合了全局统计特征和局部上下文特征，弥补了Word2Vec仅关注局部文本的缺陷。它的训练逻辑更加全面：首先遍历全部训练语料，统计所有单词两两之间的共现次数，构建完整的全局词共现矩阵，记录整个数据集的词语搭配规律；再结合Word2Vec的局部上下文窗口训练思想，通过专属损失函数拟合词语共现概率，迭代优化词向量。同时模型会弱化超高频虚词的干扰，让语义学习更聚焦核心词汇。   

| 对比维度 | Word2Vec | GloVe |
| :--- | :--- | :--- |
| 核心设计思想 | 基于局部滑动窗口，通过「上下文-中心词」的预测关系学习词向量 | 融合全局词共现统计与局部上下文，基于共现概率关系学习词向量 |
| 训练方式 | 预测式训练：CBOW 用上下文预测中心词；Skip-Gram 用中心词预测上下文 | 拟合式训练：直接拟合词共现矩阵的对数关系，构造加权损失优化向量 |
| 语义表征效果 | 基础语义关系表现良好，对局部搭配敏感；整体语义一致性弱于 GloVe | 兼顾全局统计规律，词语义关系更稳定，语义相似度、词语类比任务表现更优 |
| 典型范式 | CBOW、Skip-Gram 两种经典训练模式 | 基于共现矩阵的加权最小二乘损失训练 |
| 适用场景 | 小语料快速训练、简单文本任务、轻量语义表征 | 通用语义任务、对向量质量要求高的场景、预训练词向量迁移使用 |  

## 动态词向量模型  
### BERT 预训练模型

静态词向量（如 Word2Vec、GloVe）为每个单词赋予了一个固定的“语义快照”，但自然语言的精妙之处恰在于“一词多义”——“苹果”既可以是水果，也可以是公司。BERT（Bidirectional Encoder Representations from Transformers）的诞生，正是为了打破这种静态的桎梏。它的核心作用，是**根据单词所处的具体上下文，动态生成该词的向量表示**。同一个单词在不同句子里，会拥有截然不同的、随环境而变的“流动向量”。例如：

- 句子1：“我吃了**苹果**。” → `apple` 的向量在此刻偏向食物语义。
- 句子2：“我买了**苹果**手机。” → 同一个 `apple`，其向量在此时会动态调整，偏向科技产品语义。

BERT不再是一个简单的“词-向量”映射表，而是一个深层的双向编码网络。它不再孤立地看单词，而是让句子中的每一个词都能“看见”它左侧和右侧的所有词语，从而实现深度的上下文理解。

---

### BERT 核心架构

BERT基于Transformer的编码器（Encoder）堆叠而成，其核心创新在于**双向注意力机制（Bidirectional Attention）**。与Word2Vec仅通过局部滑动窗口捕捉有限上下文不同，BERT通过全局自注意力（Self-Attention）机制，在模型的每一层都让句子中的每个词直接与所有其他词进行信息交互，从而捕捉整个句子乃至跨句子的复杂依赖关系。

为了在海量无标注文本上进行高效学习，BERT设计了两个创新的预训练任务，取代了传统的“预测中心词”或“共现统计”：

- **掩码语言模型（MLM，Masked Language Model）**：这是BERT实现双向性的关键。在输入文本中，随机将15%的单词替换为特殊的 `[MASK]` 标记，然后要求模型根据该词左右两侧的全部上下文（即双向信息），去预测被遮盖住的是哪个原始单词。这个任务迫使模型不仅要理解左侧词汇，也要理解右侧词汇，从而学到深层次的句法和语义知识。例如，输入“我 `[MASK]` 了一个苹果”，模型需结合“我”和“苹果”来预测“吃”或“买”。

- **下一句预测（NSP，Next Sentence Prediction）**：为了理解句子间的逻辑关系（如因果、顺承、转折），BERT额外引入了二分类任务。模型会接收一对句子（句子A和句子B），并预测句子B是否是句子A的紧接着的下一句（IsNext）还是随机从语料中抽取的无关句子（NotNext）。这一任务让BERT具备了篇章级别的理解能力，为问答系统、文本蕴含等下游任务奠定基础。

---

### 训练产物与本质差异

与Word2Vec训练完成后直接得到“词向量表”不同，**BERT训练完成后，产物是整个神经网络的权重参数**。在实际使用时，我们不再通过查表获取词向量，而是将整个句子输入网络，经过多层双向注意力计算后，从网络的某一层输出层中，提取每个单词对应的动态向量。

这两个预训练任务的目标殊途同归：都是通过海量文本（如维基百科、书籍）中的语言规律，让模型学会词语在真实语境下的用法。预训练结束后，这套拥有数亿参数的权重，便构成一个强大的语义提取引擎。开发者可以将其在下游任务（如情感分类、命名实体识别）上进行微调（Fine-tuning），从而以极小的标注成本获得极佳的效果。

相较于Word2Vec，BERT彻底解决了静态词向量无法处理“一词多义”和缺乏深度上下文理解的痛点，并且在全局语义捕捉、长距离依赖建模方面拥有碾压性优势。但其代价也显而易见：模型体积庞大、推理速度较慢、对算力要求极高，同时在极短文本或特定领域的轻量化部署场景中，其优势可能无法充分施展。

## 深度神经网络特征提取器（网络结构）  
Word2Vec、GloVe 仅能将单个单词转化为固定向量，只能表示孤立单词语义，无法自动整合整句话、整个句子的整体特征。因此需要借助神经网络对词向量序列进行二次特征提取，将多个词向量融合为最终的句子向量，从而完成文本分类任务。本实验使用的 CNN、RNN、LSTM 均为序列特征提取网络，它们需要搭配词向量嵌入层与分类输出层，共同组成完整的文本分类模型。它们本身不产生词向量，核心作用：**在词向量（Word2Vec/GloVe Embedding）的基础上，进一步提取句子的语义特征，完成文本分类任务**.      
### 输入数据的格式  

无论使用 CNN、LSTM 还是 Transformer，所有深度学习模型的输入数据格式都是相同的——**一个形状为 `[Batch_Size, 序列长度, 嵌入维度]` 的三维张量**。

---

### 三个维度的含义

| 维度 | 名称 | 含义 | 常见取值 |
|:---|:---|:---|:---|
| **第1维** | `Batch_Size`（批次大小） | 一次输入多少个句子 | 16、32、64、128 |
| **第2维** | `序列长度`（Sequence Length） | 每个句子固定包含多少个词 | 512、768、1024 |
| **第3维** | `嵌入维度`（Embedding Dimension） | 用多少个数字描述一个词 | 300（Word2Vec/Glove）、768（BERT-base）、1024（BERT-large） |  

形象的理解：
把序列长度和嵌入维度想象成一张excel表格的大小规模，对于一条句子来说，序列长度代表有多少行，即有多少单词，而嵌入维度就代表列数，即有多少属性（偏向男性化的程度，偏向食物的程度，偏向红色的程度...等等）。那么batch就代表有几张这样规模大小一致的excel表格，即一个批次里有多少句子，同一批次里的所有excel表格上下摞起来就是标准的输入格式。  
注意：对于长度短于序列长度的句子，应在末尾继续添加无意义的占位符（通常是 [PAD]，值为0），强行补长到序列长度。对于长度长于序列长度的句子，应强行截断，使其长度等于序列长度。

---

### 为什么必须是三维的？

**1. Batch_Size：为了并行计算**

显卡（GPU）擅长一次性处理大量数据。如果一次只处理1个句子，算力浪费严重；一次处理太多，显存放不下。`Batch_Size` 就是在“效率”和“显存”之间取的平衡值。

**2. 序列长度：为了统一成矩阵**

句子有长有短，但神经网络要求所有输入必须是一个**整齐的矩形**，不能是锯齿状。所以通过截断（长句）和补零（短句），把所有句子强行拉到相同长度。

**3. 嵌入维度：为了用数字表示语义**

计算机不认识文字，只认识数字。每个词用一个固定长度的浮点数数组（向量）来表示其语义，这个数组的长度就是嵌入维度。

---

### 形状演变路线图

从原始文本到最终输入模型，数据的形状变化如下：    
"我 爱 这 部 电 影" ← 原始文本（字符串）   
          ↓ 分词  
["我", "爱", "这", "部", "电", "影"] ← 词列表（长度6）   
          ↓ 编码（词→数字ID）  
[24, 583, 1024, 56, 32, 789] ← 数字ID序列（长度6）  
          ↓ 填充/截断（统一到固定长度，比如512）  
[24, 583, 1024, 56, 32, 789, 0, 0, ...] ← 固定长度512的数字序列  
          ↓ 查表（每个ID替换成300维词向量）   
[[0.1, 0.2, ...], [0.3, 0.4, ...], ...] ← 形状 [512, 300]（单个句子）   
          ↓ 堆叠成Batch（比如32个句子）  
[32, 512, 300]   
          ↓
最终输入模型的形状      

### 一、CNN 文本卷积神经网络
#### 1. 核心功能
TextCNN 是适配文本任务的轻量化卷积网络，由图像CNN优化改造而来，核心作用是高效提取文本局部组合特征，聚焦词语搭配、短句语义，不依赖上下文记忆与语序逻辑，适配文本短语级特征挖掘，多用于情感分类任务。
#### 2. 详细工作机制
- 输入结构：以 GloVe 预训练词向量构成的二维矩阵为输入，矩阵行数为句子单词数，列数为词向量维度。
- 多尺度卷积核滑动采样：设置 2、3、4 等不同尺寸的卷积核，分别对应二元、三元、多元词语组合，模拟人工提取 n-gram 语法特征，可精准捕捉 “not happy”“very wonderful” 等关键情感短语。
- 卷积特征计算：卷积核在词向量矩阵上逐行滑动，对窗口内局部词向量加权运算，提取相邻词语的组合语义特征。   
🏃 运动方式：一维滑动    
我们用一个具体的例子来演示。假设：   

句子长度（单词数）= 5，即矩阵有5行。   

词向量维度 = 4，即矩阵有4列。   

卷积核大小（kernel_size）= 3，即每次看3个连续单词。   

我们的二维矩阵长这样（每一行是一个词的向量）：   

text   
行1 (词1): [a1, a2, a3, a4]    
行2 (词2): [b1, b2, b3, b4]   
行3 (词3): [c1, c2, c3, c4]   
行4 (词4): [d1, d2, d3, d4]    
行5 (词5): [e1, e2, e3, e4]   
现在，一个大小为3的卷积核（其实是一个 (3, 4) 的二维权重矩阵，卷积核的宽度固定等于词向量维度大小）开始运动了：   

第一步：卷积核覆盖 行1、行2、行3。也就是这三个词的所有向量数值（共 3×4=12 个数）会与卷积核的12个权重进行逐元素相乘并求和（加上偏置），得到一个标量（一个数字），这就是第一个特征值。

第二步（滑动）：卷积核向下移动一行，现在覆盖 行2、行3、行4。同样，对这三行的12个数值进行计算，得到第二个特征值。

第三步（滑动）：卷积核再次向下移动一行，覆盖 行3、行4、行5。计算得到第三个特征值。

运动结束。 因为卷积核滑到了底部，无法再往下了。

$$
c_i=\text{ReLU}(W\cdot E_{i:i+k-1}+b)
$$

$W$ 代表卷积核权重，$E_{i:i+k-1}$ 为窗口内词向量，经过ReLU激活得到一组局部特征。
- 最大池化降维筛选：对卷积生成的特征图进行全局最大池化，保留每组特征里数值最大的有效信息，过滤冗余特征，同时完成维度压缩。

$$
\hat{c}=\max\{c\}
$$

- 特征融合分类：将2、3、4尺度卷积核池化后的特征拼接融合，输入全连接层，完成情感正负分类。
示例：

 1. 原始文本输入
**输入**：一段自然语言文本，例如一个电影评论。
`"I really love this movie, it's fantastic!"`

 2. 文本预处理
将原始文本转换为模型可处理的数字序列。

1.  **分词**：将句子拆分成独立的单词（Token）。
    `["I", "really", "love", "this", "movie", "it's", "fantastic"]`
2.  **转小写与清洗**：统一为小写，去除标点符号（可选）。
    `["i", "really", "love", "this", "movie", "it's", "fantastic"]`
3.  **序列化**：将每个单词映射为一个唯一的整数ID（根据预建的词典）。
    `[45, 832, 124, 91, 567, 33, 2098]`
4.  **填充与截断**：将所有句子统一为固定长度（例如 `max_len=10`）。长度不足的用`0`填充，超过的截断。
    `[45, 832, 124, 91, 567, 33, 2098, 0, 0, 0]`

 3. 词嵌入（Word Embedding）
将整数ID序列转换为稠密的向量矩阵，为模型提供语义信息。

-   **方式**：使用预训练的词向量（如GloVe）进行映射，或在模型训练中学习。
-   **输入**：整数ID序列 `[45, 832, 124, 91, 567, 33, 2098, 0, 0, 0]`。
-   **操作**：系统根据每个ID去“词向量表”中查找对应的向量。
    -   ID `45` -> 查询得到向量 `[0.12, -0.34, ...]`
    -   ID `832` -> 查询得到向量 `[0.56, 0.78, ...]`
    -   ...
-   **输出**：一个二维矩阵，形状为 **(`max_len`, `embedding_dim`)**。
    -   行数 = 序列长度（10），代表每个词的位置。
    -   列数 = 词向量维度（例如300）。
    -   这个矩阵就是CNN的输入特征图。

 4. 卷积层（Convolutional Layer）
提取文本中的局部特征（即连续的N-Gram模式）。

-   **输入**：二维矩阵 `(10, 300)`。
-   **操作**：多个一维卷积核（`Conv1D`）在矩阵上从上到下滑动。
    -   假设有 `filters=128` 个卷积核。
    -   每个卷积核的大小 `kernel_size=3`，意味着每次覆盖连续的3行（3个词）。
    -   每个卷积核滑动 `10 - 3 + 1 = 8` 次，产生8个数值。
-   **输出**：一个新的二维矩阵，形状为 **(`filters`, `滑动次数`)**，即 `(128, 8)`。
    -   每一行代表一个卷积核提取出的特征序列。
    -   每一列对应一个特定的3词片段（N-Gram）的特征值。

 5. 池化层（Pooling Layer）
压缩特征，保留最显著的信息，并将数据转换为固定长度。

-   **输入**：卷积层输出的矩阵 `(128, 8)`。
-   **操作**：使用**全局最大池化（Global Max Pooling）**。
    -   在矩阵的每一行（每个卷积核的特征序列）中，选出最大的那个数值。
-   **输出**：一个固定长度的一维向量，长度等于卷积核的数量，即 **(`filters`, )** 或 `(128, )`。
    -   这个向量浓缩了整个句子中，所有卷积核捕捉到的最强特征。

 6. 全连接层（Fully Connected Layer）
将提取到的高级特征映射到样本标记空间，进行最终分类。

-   **输入**：池化层输出的固定长度向量 `(128, )`。
-   **操作**：进行线性变换和非线性激活。
    1.  **线性变换**：向量与权重矩阵相乘并加上偏置。
        -   假设任务有3个类别（积极、消极、中性），则全连接层有3个神经元。
        -   计算：`输出向量 (1, 3) = 输入向量 (1, 128) · 权重矩阵 (128, 3) + 偏置 (1, 3)`。
    2.  **非线性激活**：通常使用ReLU或Tanh函数，增加模型的表达能力。
-   **输出**：一个长度为类别数（3）的向量，即原始得分（logits），例如 `[2.1, 0.5, -1.2]`。

 7. 输出层与分类判断
将全连接层的原始得分转换为概率，并输出最终类别。

-   **输入**：全连接层的原始得分 `[2.1, 0.5, -1.2]`。
-   **操作**：应用 **Softmax** 函数，将得分归一化为概率分布。
    -   概率计算：`P(类别) = exp(得分) / sum(exp(所有得分))`。
    -   结果是 `[0.75, 0.20, 0.05]`，三个概率值加起来等于1。
-   **最终判断**：取概率最大的类别作为预测结果。
    -   类别0（积极）的概率最高（0.75），因此模型判断该评论为**积极**。

#### 3. 优缺点分析
- 优点：网络结构简单，全程并行计算，训练速度快、模型收敛稳定；对短句、固定情感短语的识别精度高，泛化能力优秀。
- 缺点：卷积核感受野固定，仅能捕捉局部相邻词语关联，无法建模远距离语义依赖；对语序不敏感，长文本建模效果较差。

| 项目 | 内容 |
| ---- | ---- |
| 核心功能 | 面向文本任务的轻量化卷积网络，专注抽取局部词语组合特征，捕获短语层面语义信息。 |
| 输入形式 | GloVe词向量拼接形成二维矩阵 $E\in R^{T\times d}$，$T$为序列长度，$d$为词向量维度。 |
| 特征提取流程 | 通过尺寸2、3、4卷积核滑动提取多元词组特征；卷积得到特征图后利用全局最大池化筛选有效特征、压缩维度；融合多尺度特征送入全连接层完成分类。 |
| 计算特性 | 支持并行运算，训练效率高。 |
| 优势 | 模型结构简洁，收敛稳定；擅长识别固定语义短语，短句分类效果较好。 |
| 局限性 | 感受野有限，难以捕捉远距离词语关联；对语序不敏感，长文本语义建模能力不足。 |
### 二、LSTM 长短期记忆网络
#### 1. 核心功能
LSTM 是针对时序数据优化的循环神经网络，核心作用是解决传统RNN梯度消失问题，精准捕捉文本语序信息与长距离上下文依赖，挖掘全文时序语义逻辑，适用于长文本情感建模任务。
#### 2. 详细工作机制
- 输入结构：以 GloVe 预训练词向量时序序列为输入，词语按照文本先后顺序逐词送入网络。
- 遗忘门：接收当前词向量与上一时刻隐状态，输出0~1之间数值，控制丢弃上一时刻细胞状态中的无用历史信息。

$$
f_t = \sigma(W_f[h_{t-1}, e_t] + b_f)
$$

$f_t$越接近0代表丢弃记忆，越接近1代表保留原有记忆。
- 输入门：分为两部分，一部分判定需要新增多少信息，另一部分生成候选记忆内容。

$$
i_t = \sigma(W_i[h_{t-1}, e_t] + b_i)
$$

$$
\tilde{c}_t = \tanh(W_c[h_{t-1}, e_t] + b_c)
$$

$i_t$控制新增信息保留比例，$\tilde{c}_t$为待写入细胞状态的候选信息。
- 细胞状态更新：利用遗忘门清理旧记忆，输入门叠加新候选信息，更新长期记忆细胞状态。

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

- 输出门：根据当前细胞状态筛选有效信息，计算当前时刻向外输出的隐状态。

$$
o_t = \sigma(W_o[h_{t-1}, e_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

$h_t$即为当前单词融合上下文后的时序特征。
- 全局时序特征输出：遍历完整条文本词序列后，取最后时刻隐藏状态作为整句文本全局时序特征。
- 特征分类预测：将全局时序特征输入全连接层，完成情感正负分类。
#### 3. 优缺点分析
- 优点：严格遵循文本语序计算，语义逻辑性强；具备长距离依赖捕捉能力，长文本建模效果优于CNN；有效解决传统RNN梯度消失问题。
- 缺点：采用串行逐词计算方式，训练速度慢、耗时久；仅依赖最后时刻状态输出特征，容易丢失文本前部关键语义信息，特征利用率较低。

| 项目 | 内容 |
| ---- | ---- |
| 核心功能 | 时序循环网络，依靠门控机制筛选语义信息，捕捉文本语序逻辑与长距离上下文依赖。 |
| 输入形式 | 按文本语序排列的GloVe词向量时序序列 $e_1,e_2,\dots,e_T$。 |
| 特征提取流程 | 词向量逐词输入网络→遗忘门、输入门、输出门协同调控记忆→迭代更新细胞状态与隐藏状态→取末端隐状态 $h_T$ 作为全局特征→全连接层分类。 |
| 计算特性 | 串行时序计算，训练速度较慢。 |
| 优势 | 对词语语序敏感，语义逻辑贴合文本规律；擅长长文本建模，解决RNN梯度消失问题。 |
| 局限性 | 训练效率低；单末端状态输出，易丢失前文关键特征，特征提取不充分。 |


### 三、GRU 门控循环单元
#### 1. 核心功能
GRU 是 LSTM 的轻量级变体，同样针对时序数据设计，用于解决传统 RNN 梯度消失问题，捕捉文本语序信息与中长距离上下文依赖。其核心功能与 LSTM 类似，但以更简洁的结构实现相近效果，适用于中等长度文本的情感建模任务。
#### 2. 详细工作机制
- 输入结构：以 GloVe 预训练词向量时序序列为输入，词语按文本先后顺序逐词送入网络。
- 重置门：接收当前词向量与上一时刻隐状态，输出0~1之间数值，控制上一时刻隐状态中被忽略的信息比例，帮助模型决定如何融合新输入与过往记忆。

$$
r_t = \sigma(W_r[h_{t-1}, e_t] + b_r)
$$

$r_t$越接近0代表忽略更多历史信息，越接近1代表保留更多历史信息。
- 更新门：同时承担 LSTM 中遗忘门和输入门的功能，输出0~1之间数值，一方面控制上一时刻隐状态有多少信息被保留，另一方面决定当前候选状态有多少信息被写入。

$$
z_t = \sigma(W_z[h_{t-1}, e_t] + b_z)
$$

$z_t$越接近1代表更多地保留旧状态并减少新信息写入，越接近0代表更多地更新为新状态。
- 候选隐状态：利用重置门筛选上一时刻有用信息，结合当前词向量生成候选记忆内容。

$$
\tilde{h}_t = \tanh(W_h[r_t \odot h_{t-1}, e_t] + b_h)
$$

$r_t \odot h_{t-1}$表示按元素重置历史隐状态，丢弃与当前预测无关的过往信息。
- 隐状态更新：通过更新门在上一时刻隐状态与当前候选隐状态之间进行插值，得到最终输出隐状态。无需像 LSTM 那样维护独立的细胞状态。

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

$h_t$即为当前单词融合上下文后的时序特征，同时也作为下一时刻的历史隐状态传递。
- 全局时序特征输出：遍历完整条文本词序列后，取最后时刻隐藏状态 $h_T$ 作为整句文本全局时序特征。
- 特征分类预测：将全局时序特征输入全连接层，完成情感正负分类。
#### 3. 优缺点分析
- 优点：严格遵循文本语序计算，语义逻辑性强；结构比 LSTM 更简洁（少一个门控单元，无独立细胞状态），参数量更少，训练速度更快；有效缓解 RNN 梯度消失问题，在中长文本上表现良好。
- 缺点：采用串行逐词计算方式，训练速度仍慢于 CNN 等并行模型；仅依赖最后时刻状态输出特征，同样存在丢失文本前部关键语义信息的风险；对于超长文本（如长篇小说）的长距离依赖捕捉能力略逊于 LSTM。

| 项目 | 内容 |
| ---- | ---- |
| 核心功能 | 时序循环网络，依靠重置门与更新门筛选语义信息，捕捉文本语序逻辑与中长距离上下文依赖。 |
| 输入形式 | 按文本语序排列的GloVe词向量时序序列 $e_1,e_2,\dots,e_T$。 |
| 特征提取流程 | 词向量逐词输入网络→重置门与更新门协同调控记忆→通过门控插值更新隐状态→取末端隐状态 $h_T$ 作为全局特征→全连接层分类。 |
| 计算特性 | 串行时序计算，参数量少于LSTM，训练速度优于LSTM。 |
| 优势 | 对词语语序敏感，语义逻辑贴合文本规律；结构轻量，训练效率高于LSTM；有效解决RNN梯度消失问题。 |
| 局限性 | 仍为串行计算，并行性不足；单末端状态输出，易丢失前文关键特征；超长文本建模能力略弱于LSTM。 |

### 四、CNN-LSTM 混合特征提取模型
#### 1. 核心功能
CNN-LSTM 是融合卷积网络与循环网络的混合模型，结合CNN局部短语提取优势与LSTM长时序建模优势，同时挖掘文本局部细粒度语义与全局上下文时序逻辑，实现双层特征互补。
#### 2. 详细工作机制
- 输入结构：以 GloVe 预训练词向量构成的二维矩阵作为模型底层输入。
- CNN局部特征挖掘：使用不同尺寸卷积核滑动提取文本n-gram短语特征，捕获局部词语搭配、情感词组，输出连续特征序列。

$$
c_i=\text{ReLU}(W\cdot E_{i:i+k-1}+b)
$$

原始词向量经过卷积运算，转换为短语级别特征序列。
- LSTM时序深度建模：将CNN输出的短语特征序列代替原始词向量送入LSTM，按时序学习短语之间先后关系与上下文依赖。

$$
f_t = \sigma(W_f[h_{t-1}, c_t] + b_f)
$$

$$
i_t = \sigma(W_i[h_{t-1}, c_t] + b_i)
$$

$$
\tilde{c}_t = \tanh(W_c[h_{t-1}, c_t] + b_c)
$$

$$
c_t' = f_t \odot c_{t-1}' + i_t \odot \tilde{c}_t
$$

$$
o_t = \sigma(W_o[h_{t-1}, c_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(c_t')
$$

LSTM以短语特征为基础，建模短语之间的时序关联。
- 全局特征融合输出：取LSTM最后时刻隐状态，融合局部短语特征与全局时序依赖，作为文本整体表征。
- 分类预测：融合特征输入全连接层，完成情感正负分类。
#### 3. 优缺点分析
- 优点：双模块优势互补，既保留CNN短语特征提取能力，又具备LSTM时序建模能力；特征维度更丰富，长短文本适配性更强，综合表征能力优于单一模型。
- 缺点：模型结构复杂，参数量更大；双层网络叠加进一步降低训练速度；仍采用末端单状态输出，存在一定的信息丢失问题。

| 项目 | 内容 |
| ---- | ---- |
| 核心功能 | 混合双层特征提取模型，融合CNN局部短语特征与LSTM全局时序特征，实现粗细粒度语义结合。 |
| 输入形式 | GloVe预训练词向量二维矩阵 $E\in R^{T\times d}$。 |
| 特征提取流程 | 词向量输入→CNN卷积提取局部短语特征，生成特征序列→特征序列送入LSTM建模长距离时序依赖→取末端隐状态作为全局特征→全连接层分类。 |
| 计算特性 | 先并行卷积、后串行循环计算，结构复杂，训练开销较大。 |
| 优势 | 局部细节与全局语义兼顾，特征表达全面；适配长短各类文本，鲁棒性更强。 |
| 局限性 | 模型参数量大、训练耗时；依旧依赖末端输出，无法完全规避前文信息丢失问题。 |


### 五、Attention-LSTM 注意力增强时序模型
#### 1. 核心功能
Attention-LSTM 在LSTM基础上引入注意力机制，解决普通LSTM仅依靠最后时刻隐状态造成前文信息丢失的缺陷；自适应为每个时序位置分配权重，自动聚焦关键词语义，实现全局特征最优聚合。
#### 2. 详细工作机制
- 输入结构：GloVe词向量序列按照文本顺序依次送入LSTM网络。
- LSTM时序编码：利用LSTM门控机制逐词计算，保留文本全部位置对应的隐状态序列，记录全程时序信息。

$$
f_t = \sigma(W_f[h_{t-1}, e_t] + b_f)
$$

$$
i_t = \sigma(W_i[h_{t-1}, e_t] + b_i)
$$

$$
\tilde{c}_t = \tanh(W_c[h_{t-1}, e_t] + b_c)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

$$
o_t = \sigma(W_o[h_{t-1}, e_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

输出全部时刻隐状态 $h_1,h_2...h_T$，不直接丢弃中间信息。
- 注意力权重计算：对每一个位置的隐状态进行变换打分，经过Softmax归一化，得到各词语对应的重要程度权重。

$$
u_t=\tanh(W_a h_t + b_a)
$$

$$
s_t=v_a^\top u_t
$$

$$
\alpha_t=\frac{\exp(s_t)}{\sum_{k=1}^T\exp(s_k)}
$$

$\alpha_t$数值越大，代表该词语对情感分类结果影响越高。
- 特征加权聚合：利用权重对所有隐状态加权求和，重点突出情感关键词信息，整合得到文本全局特征向量。

$$
r=\sum_{t=1}^T \alpha_t h_t
$$

- 分类预测：加权聚合得到全局表征 $r$ 送入全连接层，完成情感正负分类。
#### 3. 优缺点分析
- 优点：继承LSTM时序建模能力；注意力机制自适应筛选关键信息，缓解长文本前部信息丢失；模型具备可解释性，可可视化词语权重；特征聚合方式更加合理。
- 缺点：LSTM串行计算存在训练速度短板；注意力模块引入额外参数，增大计算开销；底层依旧使用静态GloVe词向量，无法解决一词多义问题。

| 项目 | 内容 |
| ---- | ---- |
| 核心功能 | 注意力增强时序模型，通过LSTM提取完整时序特征，利用注意力自适应分配权重，加权聚合全局语义表征。 |
| 输入形式 | 按语序排列的GloVe词向量时序序列 $e_1,e_2,\dots,e_T$。 |
| 特征提取流程 | GloVe词向量序列输入LSTM→输出全部时序隐状态序列→注意力层计算各位置权重→加权求和得到全局特征向量 $r$→全连接层完成分类。 |
| 计算特性 | LSTM串行运算叠加注意力权重求解，计算开销高于基础LSTM。 |
| 优势 | 完整保留全部时序信息，自动聚焦关键词；克服基础LSTM末端状态信息损失问题，特征表征效果更优。 |
| 局限性 | 训练速度较慢；注意力带来额外参数量；依赖静态词向量，不具备动态语义编码能力。 |      


### 六、Capsule_LSTM 胶囊时序融合网络
#### 1. 核心功能
Capsule_LSTM 是将 LSTM 的时序建模能力与 Capsule Network 的层级特征表征能力相结合的混合模型。LSTM 负责捕捉文本语序逻辑与长距离上下文依赖，Capsule 层进一步将 LSTM 输出的时序特征向量编码为胶囊形态，通过动态路由机制挖掘特征间的部分-整体层级关系，增强模型对文本中关键情感词组和句式结构的识别能力，适用于复杂情感表达与细粒度情感分类任务。
#### 2. 详细工作机制
- 输入结构：以 GloVe 预训练词向量时序序列为输入，词语按文本先后顺序逐词送入 LSTM 子网络。
- LSTM 时序编码：LSTM 按标准流程遍历整条文本，输出每个时间步的隐藏状态 $h_1, h_2, \dots, h_T$，而非仅取末端状态。所有时间步隐状态共同构成时序特征序列，作为胶囊网络的输入源。

$$
h_t = \text{LSTM}(e_t, h_{t-1}, c_{t-1})
$$

- 主胶囊层（Primary Capsule）：将每个时间步的 LSTM 隐状态 $h_t$ 通过仿射变换映射为多个主胶囊向量，每个胶囊捕捉该时间步不同语义子空间的特征信息。

$$
u_i = W_i h_t + b_i
$$

$u_i$ 表示第 $i$ 个主胶囊的初始特征向量，$W_i$ 为可学习的仿射变换矩阵。
- 动态路由与数字胶囊层（Digit Capsule）：主胶囊向量通过迭代动态路由协议，计算耦合系数 $c_{ij}$，将主胶囊输出加权求和后传入高层数字胶囊，形成更抽象的情感语义表征。

$$
s_j = \sum_{i} c_{ij} \cdot \hat{u}_{j|i}
$$

$$
v_j = \frac{\|s_j\|^2}{1 + \|s_j\|^2} \cdot \frac{s_j}{\|s_j\|}
$$

其中 $\hat{u}_{j|i}$ 为预测向量，$c_{ij}$ 由路由迭代更新，$v_j$ 为第 $j$ 个数字胶囊的输出向量，其模长表示对应情感类别（或情感属性）的激活概率。
- 全局特征融合输出：将所有数字胶囊的输出向量拼接为一维全局特征向量，或取模长最大的胶囊向量作为最终表征。

$$
V_{\text{global}} = \text{concat}(v_1, v_2, \dots, v_K)
$$

$K$ 为数字胶囊总数，通常对应预设的情感类别数或属性维度数。
- 特征分类预测：将融合后的全局特征输入全连接层（或直接依据胶囊模长），完成情感正负分类或多类别情感分类。
#### 3. 优缺点分析
- 优点：融合 LSTM 的语序敏感性与胶囊网络的层级表征能力，对文本中关键情感短语、否定转移等结构更敏感；动态路由机制能够自动学习特征间空间关系，提高模型对文本变形的鲁棒性；适用于复杂情感文本（如评价、评论）的细粒度分析。
- 缺点：整体结构较为复杂，参数量远大于单一 LSTM；动态路由迭代计算导致训练和推理速度显著下降；对短文本或简单情感分类任务而言收益有限，性价比不高。

| 项目 | 内容 |
| ---- | ---- |
| 核心功能 | LSTM 提取时序上下文特征，胶囊网络通过动态路由编码层级语义关系，提升复杂情感文本的分类性能。 |
| 输入形式 | 按文本语序排列的GloVe词向量时序序列 $e_1,e_2,\dots,e_T$。 |
| 特征提取流程 | LSTM逐词编码输出全时序隐状态→仿射变换生成主胶囊向量→动态路由迭代生成数字胶囊→拼接或取模长得到全局特征→全连接层分类。 |
| 计算特性 | 串行时序计算叠加胶囊路由迭代，训练速度较慢。 |
| 优势 | 兼具时序语序建模与层级特征表征能力；对情感词组、否定结构敏感；鲁棒性优于纯LSTM。 |
| 局限性 | 结构复杂，参数量大；训练推理耗时显著增加；短文本/简单任务收益不明显。 |

### 七、Transformer 自注意力序列建模网络

#### 1. 核心功能

Transformer 是完全基于自注意力机制的序列建模架构，由 Google 于 2017 年提出，核心作用是并行处理序列中任意位置之间的全局依赖关系，动态生成每个词的上下文相关表征，解决 RNN/LSTM 串行计算效率低、长距离依赖建模困难的问题，是当前所有大语言模型（BERT、GPT、LLaMA 等）的基础架构。

#### 2. 详细工作机制

- 输入结构：以词向量序列与位置编码相加构成的矩阵为输入，矩阵行数为句子单词数，列数为词向量维度，位置编码显式注入语序信息。

$$
X = E + PE
$$

$E \in \mathbb{R}^{T \times d}$ 为词向量矩阵，$PE$ 为正弦/余弦位置编码，$T$ 为序列长度，$d$ 为向量维度。

- 多头自注意力全局交互：将输入线性映射为 Query、Key、Value 三组矩阵，通过点积计算任意两词之间的关联权重，加权聚合全局语义信息。多头机制并行执行多组独立的注意力计算，捕捉语法、指代、语义相似等不同子空间的关系。

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
$$

$d_k$ 为缩放因子，防止点积过大导致 softmax 梯度消失；$h$ 为注意力头数。

- 前馈网络非线性变换：对每个位置的注意力输出独立通过两层全连接网络，增强模型的非线性表达能力。

$$
\text{FFN}(x)=\text{ReLU}(xW_1+b_1)W_2+b_2
$$

- 残差连接与层归一化：每个子层（注意力层、前馈层）输出与输入相加后做层归一化，稳定深层网络训练，缓解梯度消失。

$$
\text{Output}=\text{LayerNorm}(x+\text{Sublayer}(x))
$$

- 多层堆叠深度编码：将上述模块堆叠 $N$ 层（编码器）或配合掩码注意力（解码器），逐层抽象更高级的语义表征。最终输出为每个词的上下文相关向量，可直接用于分类或序列生成。

#### 3. 优缺点分析

- 优点：任意两位置直接连接，路径长度为 $O(1)$，长距离依赖建模能力强；完全并行计算，训练效率远超 RNN；架构高度可扩展，从百万参数到万亿参数均可稳定训练。
- 缺点：自注意力的计算复杂度为 $O(T^2)$，长序列处理内存开销大；位置编码对语序的表征能力弱于显式循环结构；缺乏内置的归纳偏置，小数据集上容易过拟合。

| 项目 | 内容 |
| ---- | ---- |
| 核心功能 | 完全基于自注意力机制的序列建模架构，并行捕捉全局上下文依赖，动态生成上下文相关表征。 |
| 输入形式 | 词向量与位置编码相加形成输入矩阵 $X \in \mathbb{R}^{T \times d}$，$T$ 为序列长度，$d$ 为向量维度。 |
| 特征提取流程 | 多头自注意力计算任意词对关联权重并加权聚合全局信息；前馈网络对每个位置独立做非线性变换；残差连接与层归一化稳定训练；堆叠 $N$ 层实现深层语义抽象。 |
| 计算特性 | 支持全序列并行运算，注意力复杂度 $O(T^2)$。 |
| 优势 | 长距离依赖建模能力强；训练效率高；架构可扩展性极强，是当前 LLM 的基础架构。 |
| 局限性 | 长序列计算开销大；位置信息依赖显式编码；缺乏归纳偏置，小数据场景易过拟合。 |    

在深度学习里，并行主要指数据维度上的同时计算。
### 1. RNN/LSTM 为什么不能并行？（串行计算）

RNN的设计缺陷在于，它有一个“隐藏状态（Hidden State）”像接力棒一样传递。

- 计算第3个词的时候，**必须**等第2个词的结果算完；
- 计算第2个词的时候，**必须**等第1个词的结果算完。

这种时间上的依赖关系，导致GPU里虽然有成千上万个计算核心，但在处理RNN时，大部分时间都在“干等”，无法充分利用算力。

### 2. Transformer 为什么能并行？（并行计算）

Transformer抛弃了循环结构，它计算的核心是**自注意力（Self-Attention）**。

- 在计算句子中某个词（比如第3个词）的特征时，它直接把句子中**所有词（第1、2、3、4、5个词）**的向量一次性同时读进来，通过数学矩阵乘法算出权重。
- **关键点**：计算第3个词，**不需要**等第2个词的计算结果。所有词的计算是**同时、独立**进行的。

体现在代码层面，就是一次输入一个矩阵（Batch × Sequence Length），GPU把这个矩阵扔进计算核心，**一眨眼的功夫**，所有词的特征就都算出来了。  

### 🍳 一个厨房的比喻

假设你要做一顿大餐，需要完成三个步骤：**洗菜 → 切菜 → 炒菜**。

- **RNN/LSTM（串行）**：就像一个**单线程的厨师**。他必须先把菜**全部洗完**，然后才能开始切；切完所有菜，才能开火炒菜。**每一步都必须等上一步彻底完成**。如果洗菜要10分钟，切菜要10分钟，炒菜要10分钟，总共就是**30分钟**。

- **Transformer（并行）**：就像一个**拥有多个帮手的厨房**。洗菜工、切菜工、炒菜师傅**同时开工**。洗菜工洗好一个西红柿，立刻传给切菜工，切菜工切好立刻传给炒菜师傅。虽然每道工序耗时不变，但在第1分钟时，三个岗位就**同时**在运转了。处理一整批菜的总时间，几乎就等同于**最慢的那道工序的时间**（10分钟），而不是三者相加。
 

##   模型  
Word2Vec + RF 和 词袋模型 + RF作为基准模型，依靠这两个模型的分数评测其他模型的性能  
### Word2Vec+RF
Word2Vec + RF 是**静态词向量与传统机器学习结合**的经典文本分类方案，属于「词级语义表征 → 句子级向量聚合 → 传统分类器预测」的技术路线。相比词袋模型，它引入了词语语义信息，能显著提升分类效果。

#### 1. Word2Vec：单词语义向量化

Word2Vec 基于大规模语料训练，将每个单词映射为一个低维稠密的实数向量，让语义相近的单词在向量空间中距离更近。

- 训练逻辑：通过滑动窗口内的上下文与中心词的预测关系（CBOW/Skip-Gram）学习词语的语义关联，捕获词语间的相似度、类比关系。
- 在组合中的作用：为文本中的每个单词生成带有语义信息的向量表示，替代词袋模型的稀疏词频向量，解决「无语义、维度爆炸」的问题。

#### 2. 句子向量生成：均值池化聚合

Word2Vec 输出的是**单词级向量**，而随机森林需要固定维度的句子级向量作为输入，因此需要对句子内所有词向量进行聚合，最常用的方式是**均值池化（Average Pooling）**：

- 将句子中所有单词的词向量按维度逐位取平均值，得到一个和词向量维度相同的向量，作为整条句子的语义表征。
- 优势：计算简单，能在一定程度上保留句子的整体语义，输出维度固定，适配传统机器学习模型的输入要求。

#### 3. 随机森林（RF）：分类决策

随机森林是基于决策树的集成学习分类器，通过多棵决策树的投票结果输出最终分类。

- 工作机制：随机抽取样本与特征构建多棵互不依赖的决策树，每棵树独立输出分类结果，最终以少数服从多数的投票方式决定文本的情感类别。
- 在组合中的优势：对中高维向量泛化能力强，不易过拟合，训练速度快，能自动评估特征重要性。

#### 4. 完整工作流程

1. 文本预处理：对影评文本进行清洗、分词，去除停用词与无意义符号。
2. 词向量映射：将分词后的每个单词通过 Word2Vec 模型映射为对应的稠密词向量。
3. 句子向量聚合：对单条文本内所有词向量做均值池化，生成固定维度的句子向量。
4. 模型训练：将训练集的句子向量与情感标签输入随机森林，训练分类模型。
5. 预测输出：输入测试集句子向量，通过随机森林投票得到最终的情感分类结果。

|项目|内容|
| ---- | ---- |
|模型架构|Word2Vec词向量 + 均值池化聚合 + 随机森林(RF)，静态词向量搭配传统集成学习的文本分类方案|
|Word2Vec作用|在语料上学习得到低维稠密词向量，依靠CBOW/Skip-Gram机制挖掘词语语义相似度，替代稀疏词袋特征，引入语义信息|
|句子表征方式|均值池化，对文本内全部词向量逐维度求平均，生成固定维度句子向量，适配传统机器学习输入规范|
|分类模块原理|随机森林通过随机采样样本与特征训练多棵独立决策树，依靠投票机制输出分类结果，泛化性较好，不易发生过拟合|
|整体流程|文本预处理→词语映射为Word2Vec向量→均值池化生成句子向量→送入随机森林完成训练与情感预测|
|特点|相比词袋模型具备基础语义能力；实现简单；但均值池化丢失语序信息，Word2Vec为静态向量，无法区分一词多义|  

#### 结果  
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| word2vec.csv | 0.84364 |


### 词袋模型+随机森林（RF）原理
词袋模型结合随机森林是传统机器学习文本分类常用基线模型，整体分为特征构建与分类预测两个阶段。

#### 1. 词袋模型（BoW）
词袋模型忽略句子语序、语法关系，仅统计单词出现频次。
1. 根据全部语料构建统一词汇表；
2. 针对每条文本，统计词汇表内各个单词的出现次数；
3. 生成高维稀疏词频向量作为文本特征。

该方法实现简单，但不包含任何语义信息，近义词被当作独立特征，同时丢失语序信息，词汇量较大时容易出现维度灾难。

#### 2. 随机森林（RF）
随机森林是集成学习模型，由多棵独立决策树组成：
- 训练时随机采样样本与特征，分别训练单棵决策树；
- 预测时所有决策树独立输出结果，通过投票确定最终类别。
随机森林稳定性较好，不易过拟合，适合处理词袋产生的高维特征。

#### 3. 整体流程
1. 文本预处理：分词、去停用词；
2. 基于词袋模型生成词频稀疏特征；
3. 将特征与标签输入随机森林完成训练；
4. 测试文本转为词袋特征，通过随机森林得到分类结果。

#### 4. 优缺点
**优点**
- 结构简单，无需预训练词向量，易于搭建，适合作为实验基线；
- 随机森林对高维稀疏特征适应性较好。

**缺点**
- 只统计词频，无法学习词语语义；
- 完全丢失语序、短语组合信息；
- 特征向量稀疏、维度高，数据量大时计算开销上升。

| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | 词袋模型(BoW) + 随机森林(RF)，传统机器学习基线文本分类方案 |
| 特征构建模块 | 词袋模型舍弃语序与语法，统计单词频次；依托全局词汇表生成高维稀疏词频向量作为文本特征，不包含词语语义信息 |
| 分类模块原理 | 随机森林属于集成学习，通过随机选取样本和特征训练多棵独立决策树，预测阶段采用投票机制输出类别，抗过拟合，适配高维稀疏特征 |
| 整体流程 | 文本预处理（分词、去除停用词）→ 构建词频稀疏特征 → 特征与标签送入随机森林训练 → 利用训练完成模型完成分类预测 |
| 优势 | 搭建便捷，不需要预训练词向量，常用来作为实验基准模型；随机森林能够较好处理稀疏高维特征 |
| 局限性 | 仅依靠词频统计，无法理解词语语义；丢失语序、短语搭配信息；特征维度高且稀疏，大规模数据下计算成本较高 |   

#### 结果  
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| bow.csv | 0.82896 |


###  GloVe + TextCNN 模型原理
GloVe+TextCNN是预训练静态词向量结合卷积网络的经典文本分类模型，分为词向量表征、局部特征提取、分类输出三个阶段。

#### 1. GloVe词向量层
GloVe属于静态预训练词向量，结合全局词语共现统计与局部上下文信息训练得到。
文本经过分词后，每个单词查询预训练GloVe权重，转化为低维稠密词向量；整段文本组合形成词向量矩阵，作为CNN的输入。
特点：单词向量固定，不随上下文变化，无法区分一词多义，但相比词袋模型具备基础语义信息。

#### 2. TextCNN特征提取层
TextCNN利用多种尺寸的卷积核在词向量矩阵上滑动：
- 不同大小卷积核可以捕捉2词、3词、4词等局部短语特征（n-gram特征）；
- 卷积运算提取局部组合语义；
- 通过最大池化筛选出最关键的特征，压缩维度；
- 将多组卷积特征拼接，形成整条文本的综合特征向量。

TextCNN擅长挖掘短语、固定搭配等局部情感特征，采用并行计算，训练速度较快；缺点是难以捕捉远距离词语依赖。

#### 3. 分类输出层
将CNN提取得到的文本特征送入全连接层，经过激活函数与Softmax运算，输出文本所属类别。

#### 4. 完整流程
1. 文本预处理：分词、去除停用词；
2. 词语映射为GloVe预训练词向量，构成输入矩阵；
3. TextCNN执行卷积、池化操作，提取文本局部语义特征；
4. 特征送入全连接层，完成情感分类预测。

#### 5. 优缺点
**优点**
- GloVe提供带有全局统计信息的词向量，语义表征优于Word2Vec；
- CNN高效捕捉短语特征，模型训练速度快，收敛稳定；
- 稠密向量避免词袋模型高维稀疏问题。

**缺点**
- GloVe为静态词向量，不能解决一词多义；
- CNN只关注局部相邻词语，无法建模长距离上下文依赖；
- 不重视语序信息，难以区分语序不同带来的语义变化。

| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | GloVe预训练词向量 + TextCNN卷积网络，静态词向量搭配卷积神经网络的文本分类模型 |
| 词向量表征层 | 使用融合全局词共现统计与局部上下文信息的GloVe静态词向量；分词后的单词映射为稠密向量，拼接形成词向量矩阵作为网络输入，向量固定，无法区分一词多义 |
| 特征提取模块 | TextCNN借助多种尺寸卷积核滑动提取n-gram短语特征；经过卷积运算、最大池化筛选关键特征并降维，拼接多尺度特征得到文本整体语义向量，采用并行运算，训练速度较快 |
| 分类输出模块 | 提取后的特征送入全连接层，结合激活函数与Softmax得到文本分类结果 |
| 整体流程 | 文本预处理→词语转换为GloVe词向量构成输入矩阵→卷积、池化提取局部语义特征→全连接层完成情感分类预测 |
| 优势 | GloVe语义表征质量优于Word2Vec；CNN擅长捕获短语特征，训练收敛稳定；稠密向量规避词袋模型高维稀疏缺陷 |
| 局限性 | 静态词向量无法处理一词多义；仅能捕获局部词语关联，缺少长距离依赖建模能力；对语序变化不敏感 |   

#### 结果  
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| cnn.csv | 0.74548 |
| CNN (1).csv | 0.83604 |
| CNN (2).csv | 0.88816 |  
#### 对于第一版代码的结果分析及修改  
##### 结果分析  
- 学习率设置严重不合理：lr=0.8,前两轮 loss剧烈震荡，参数来回跳跃，难以找到最优区间，模型很难平稳收敛。
- 网络结构过于简陋：只用单一卷积核 size=3
- 缺少正则，容易震荡 / 轻微过拟合
##### 修改方案  
- lr设为很小的数，优化器改用 Adam
- 采用经典多卷积核 `[3,4,5]` TextCNN 结构
- 加入 Dropout 防止过拟合

#### 对于第二版代码的结果分析及修改  
##### 结果分析   
此次，我在代码中加入输出错误案例模块来输出3条错误案例
真实标签：0，预测标签：1  
文本：natural born killers cinema cut r director s cut nc it s an unusual oliver stone picture but when i read he was on drugs during the filming i needed no further explanation natural born killers is a risky mad all out film making that we do not get very often strange psychotic artistic pictures natural born killers is basically the story of how two mass killers were popularised and glorified by the media there is a great scene where an interviewer questions some teenagers about mickey and mallory and the teenager says murder is wrong but if i was a mass murderer i d be mickey and mallory mickey describes this with a situation of frankenstein the monster and dr frankenstein dr frankenstein is the media who has turned them into these monstrous killersmost oliver stone films examine the flaws of the america the country that the director loves and admires i guess natural born killers is about the effect of mass media technology and how obsessive as a nation americans are and most of the world over things such as mass killers and bizarre situations the killers played by woody harrelson mickey and juliette lewis mallory are executed astonishingly by two excellent actors who step into the lives of two interestingly brutal killers mickey and mallory believe that some people are worthy of killing perhaps in the cruel theory of social darwinism survival of the fittest mickey says in his interview in prison that other species commit murder we as humans ravage other species and exploit the environment the script is interesting but it is questionable how much this film amounts to in the sense of making us think about society and human behaviour rather than the intensity of a hour bloodbath that we have seen the last hour of the film takes place in a maximum security prison we see the harsh realities of prison life the attitudes of the warden etc overfilling of prisons maybe stone is questioning the future the path that society is leading to two other interesting characters first a reporter who runs a show about america s maniacs and is obsessed with boosting ratings that he goes to any length to capture the story of mickey and mallory the other is police officer scagnetti an insane perhaps sadistic officer that is in love with mallory he also has some weird obsession with mass killers since his mother was killed during the massacre at waco texas by charles whitman the cinematography is superb different colours shadows styles create a feeling of disorientation the green colour most evident of all is green to resemble the sickness of the killers in the drugstore when they are looking for rattlesnake antidote the camera work is insane shaky buzzy it takes some determination to get use to it and accept it highly unorthodox psychedelic and unusual natural born killers does not glamourise the existence of insane murderers it questions it and how we as the public may fuel this attribute although the above review sound quite positive i did  
文本：长篇讨论《天生杀人狂》电影手法、镜头、叙事、主题，大量客观分析，看上去有很多褒义词（cinematography is superb、excellent actors），**但是结尾转折：although the above review sound quite positive i did**  
###### 出错原因

1. **模型偏向捕捉局部正向词汇**  
文中大量 `superb、excellent、interesting` 等正面词语集中在前大半段；  
2. **无法识别长文本末尾的转折逻辑**  
否定 / 负面观点出现在全文尾部，TextCNN 依靠固定窗口局部特征提取，**缺少长距离语义依赖能力**；卷积窗口只能抓取片段词语，看不到远距离转折关系；  
3. 文本过长，局部正向信号压制了结尾负面结论。      
核心：**长文本 + 文末反转，CNN 缺乏全局上下文感知**

真实标签：1，预测标签：0
文本：there is nothing remotely scary about modern horror which is an insult to the word horror freddie vs jason the scream movies cabin trash and especially stephen king s infantile attempts he s recycled every story from the monkey s paw to whatever often in the same story at horror in both writing and on film except for kubrick s version of the shining which actually was scary unlike king s books which are as frightening as my big toe the left one which still has the nail but the woman in black is that rare modern film that will make the hairs on the back of your neck stand on end this is the way it should be done the director creates tension and the scariest ghost ever actually seen simply by having her suddenly turn up standing still somewhere or other with that incredible look on her face then he brings it all to a ghastly disturbing close he s learned his lessons from the masters who knew how to make horror val lewton original cat people and robert wise a val lewton disciple and director of the haunting and the body snatcher jacques tournier another val lewton disciple who directed a truly horrifying zombie film not the gross rubbish raimi did gross isn t scary folks it s just gross and lewis allen the uninvited and of course jack clayton s turn on henry james the innocents and the way the master of suspense hitchcock can still bring you to the edge of your seat even with a slow building and burning period piece like under capricorn ten stars    

文本：大量痛斥现代恐怖片很差、吐槽 Stephen King、吐槽杰森、惊声尖叫，出现大量负面词汇 `nothing scary、insult、infantile、rubbish`；作者先大范围批评其他影片，**最后才夸赞《黑衣女人》是优秀恐怖片**。  

###### 出错原因

1. **大量负面修饰词作为 “干扰噪声” 占据文本前半部分**；  
2. 模型被密集负面词汇干扰，误判整体情感；  
3. 区分不了：**“批评别的电影” ≠ “批评本片”**。  
TextCNN 只统计词语情感倾向，**无法区分评价对象**，分不清负面词是吐槽竞品还是评价目标影片。  

真实标签：1，预测标签：0
文本：very good except for the ending which was a huge disappointment the script was very good as was the acting the visuals were often very grainy but this in a way added to the film as the snowy features were in good places that helped create a mood towards the film this affect was ruined by the extremely unbelievable ending i was going to give this film an out of ten but the ending knocked it down a point to because it seemed to depart radically from the first minutes of the movie and seemed quite forced at the end to make the film makers look clever this movie though was much better than films with quite a lot larger budgets and seemed to be filmed like a home movie with some extra equipment not much in the way of special effects as these go but for suspense it was very good     

`very good` 开篇肯定，称赞剧本、演技、氛围感；但是重点抱怨：**结局极其离谱、让人失望（huge disappointment、unbelievable ending）**。  

###### 出错原因  

1. 影评典型模式：**大体好评，但重点抨击结局**；   
2. 模型过度放大 “disappointment、ruined” 这类强负向词汇的权重；  
3. 无法分辨：局部针对【结局】的批评 ≠ 整部电影全盘否定。   
人类能区分 “局部不满” 和 “整体态度”，CNN 单纯依靠词语无法精细区分评价范围。

###### 修改方案  
- 当前仅使用最大池化，容易被单个强极性词汇主导预测结果。可以采用**最大池化 + 平均池化拼接**，同时保留显著情感特征与全局文本特征，缓解混合情感文本误判问题；也可引入轻量注意力机制，让模型自动聚焦文本中表达核心观点的语句，降低无关干扰内容的权重。
- 目前采用单一 GloVe 预训练向量微调。可以设置可选模式：区分低频词与高频词，支持冻结 / 解冻 Embedding 层。
- 另外还采用余弦退火学习率调度，EarlyStopping 早停 + 保存最优模型


### GloVe + LSTM 模型原理
GloVe+LSTM是预训练静态词向量结合循环神经网络的文本分类模型，整体分为词向量表征、时序特征提取、分类输出三个阶段。

#### 1. GloVe词向量层
GloVe属于静态预训练词向量，融合全局词语共现统计与局部上下文信息训练得到。文本分词后，每个词语加载预训练GloVe权重转化为低维稠密词向量，按照文本语序依次排列，构成时序向量序列送入LSTM。词向量固定不变，无法区分一词多义，但具备可靠的全局语义表征能力。

#### 2. LSTM特征提取层
LSTM对传统RNN存在的梯度消失问题进行优化，依靠遗忘门、输入门、输出门共同调控细胞状态：
- 按照文本顺序逐时序读取词向量，持续更新隐藏状态；
- 通过三门控机制选择性保留、丢弃历史语义信息，有效捕捉长距离上下文依赖；
- 遍历完整序列后，取最后时刻隐藏状态作为整条文本的时序特征向量。
模型能够感知词语语序差异引发的语义变化，但采用串行计算，训练耗时高于TextCNN。

#### 3. 分类输出层
将LSTM输出的文本特征送入全连接层，配合激活函数与Softmax，输出文本类别预测结果。

#### 4. 完整流程
1. 文本预处理：分词、去除停用词；
2. 词语映射为GloVe预训练词向量，组成时序向量序列；
3. LSTM逐时序处理向量序列，学习上下文时序语义；
4. 时序特征传入全连接层，完成情感分类预测。

#### 5. 优缺点
**优点**
- GloVe词向量融合全局统计信息，语义表达质量较高；
- LSTM具备时序建模能力，重视词语语序，能够捕获长距离上下文依赖；
- 稠密词向量解决词袋模型高维稀疏问题。

**缺点**
- GloVe属于静态词向量，无法处理一词多义现象；
- LSTM串行逐词运算，训练速度较慢；
- 仅依靠最终时刻状态容易丢失文本前部关键信息。

| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | GloVe预训练词向量 + LSTM循环网络，静态词向量结合时序循环网络的文本分类模型 |
| 词向量表征层 | 采用融合全局词共现统计与局部上下文的GloVe静态词向量；分词词语映射为稠密向量，按语序组成时序序列送入网络，向量固定，无法区分一词多义 |
| 特征提取模块 | LSTM依靠遗忘门、输入门、输出门调控细胞状态，按顺序逐词处理向量序列，捕捉长距离上下文依赖；可以识别语序带来的语义差异，串行计算，训练速度较慢 |
| 分类输出模块 | LSTM输出的时序特征送入全连接层，结合激活函数与Softmax实现文本分类预测 |
| 整体流程 | 文本预处理→词语转换为GloVe时序词向量序列→LSTM提取上下文时序特征→全连接层完成情感分类预测 |
| 优势 | GloVe语义表征效果较好；LSTM可建模长距离语义依赖，对语序敏感；稠密向量规避词袋高维稀疏问题 |
| 局限性 | 静态词向量无法解决一词多义；串行运算训练开销大；仅使用末端隐藏状态容易损失前文有效信息 |   
#### 结果   

#### 结果分析   
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| lstm.csv | 0.87652 |
| lstm(1).csv | 0.79920 |
| lstm (2).csv | 0.88032 |  
| lstm (3).csv | 0.89368 |  
| lstm (4).csv | 0.89420 |  

#### 结果分析及修改方案   

##### 第一版代码    
###### 结果分析   
1. 当前模型只拿时序最后时刻拼接输出，**全局语义捕捉不足**；  
2. 无正则，容易过拟合，简单关键词就直接判定极性；   
3. 词向量冻结，无法微调适配 IMDB 数据集语境；  
4. 没有 Dropout，泛化能力差；  
5. 损失函数平等看待两类样本，对负样本识别不足。

###### 修改方案  
1. **向量放开微调**  
原先冻结 GloVe，模型无法学习 IMDB 影评特有表达方式，对讽刺、隐性情感识别差。  
2. **增加最大池化 + 最后时序拼接**  
原来只使用 LSTM 最后一步输出，长文本远端情感信息丢失；池化可以抓取全局关键信息，改善叙事类文本预测。  
3. **类别权重 `class_weights = [1.2, 1.0]`**  
针对性解决：**真实 0 (负样本) 大量被预测为 1**，加大负样本损失惩罚。  
如果依然负样本误判多，可以调高到 `[1.3,1.0]`。  
4. **加入 Dropout、梯度裁剪、学习率衰减**  
抑制过拟合，增强泛化，提升陌生弱情感文本识别能力。

##### 第二版代码   
###### 结果分析  
1. **放开 Embedding 微调 + 无充足正则**：GloVe 预训练权重在小数据集上微调极易破坏通用语义，反而效果不如冻结；  
2. **类别权重设置不当**，负样本惩罚过高，模型矫枉过正；  
3. 新增 Dropout、特征融合、学习率调度叠加在一起，超参不匹配，训练很难收敛；  
4. 原基线本身是**冻结词向量、极简结构**，贸然一次性堆所有改进，风险极高。


###### 修改方案    
退回你最初原版代码，只做最小、精准的单点优化，杜绝一堆修改堆在一起互相冲突
1. 网络结构完全和初始代码一模一样；
2. 词向量保持冻结；
3. **暂时删掉 dropout、删掉梯度裁剪、关闭学习率调度**；
4. 只保留可选类别权重（如果权重带来掉点，直接删掉）；
5. 仅优化打印逻辑（过滤 padding、清空 error_cases）。

##### 第三版代码   
###### 结果分析  
只拼接 `states[0] + states[-1]`，仅依靠序列首尾，长文本中间情感信息丢失   
###### 修改方案  
1. **融合「首尾状态 + 全局最大池化」**，捕捉句子里最强情感关键词；
2. 网络输入维度同步修正；
3. 其余所有配置保持原样：冻结 GloVe、无额外 dropout、无梯度裁剪、损失函数不变；
4. 保留全部业务代码、添加错误样本收集逻辑。

##### 第四版代码
###### 结果分析  
此次，我在代码中加入输出错误案例模块来输出3条错误案例  
【案例 1】
真实标签：0，预测标签：1
文本内容：i vaguely remember ben from my sci fi fandom days of the s i was doing several interviews bios of obscure actors actresses most notably ben actress fay spain and jody fair who played angela in s the young savages ben was one of the people at a low key sci fi con in chicago about when i had a nice chat with him and his career and life all these were published in some now long forgotten fanzine of the day wish i still had copies of those interviews but time marches on and any of those people surely wouldn t remember me at all so many years later ben was a really nice fellow ekeing out a living the cons of those days didn t even pay their guest unless of course they were big name stars and even then the pay was a couple hundred dollars at most good to know ben s still alive kicking how bout a remake of creature but years older ugly then uglier now   

【案例 2】
真实标签：1，预测标签：0
文本内容：very good except for the ending which was a huge disappointment the script was very good as was the acting the visuals were often very grainy but this in a way added to the film as the snowy features were in good places that helped create a mood towards the film this affect was ruined by the extremely unbelievable ending i was going to give this film an out of ten but the ending knocked it down a point to because it seemed to depart radically from the first minutes of the movie and seemed quite forced at the end to make the film makers look clever this movie though was much better than films with quite a lot larger budgets and seemed to be filmed like a home movie with some extra equipment not much in the way of special effects as these go but for suspense it was very good   

【案例 3】
真实标签：0，预测标签：1
文本内容：this hodge podge adapted from a gore vidal novel actually one of the great american writers makes the magic christian and valley of the dolls look like fellini art works raquel welch with an incredible body and she s actually not very tall in a lead role except for kansas city bomber when she was quite good playing rex reed s bad movie reviewer not critic alter ego only to be surrounded by drag queen great chick mae west horny john huston a young and naive farrah fawcett pre lee majors what a shame and other various creep azoids to pretend to spoof way too may things has nothing going for it except inter spliced old films clips i e widmark in kiss of death lena horne just so they can continue to bleed the life out of everyone a out of best performance it s so bad it s worth seeing     



1. **案例 1、3（负样本 0 → 预测 1）**
影评整体偏中性，夹杂少量正面词汇（`nice fellow`、`it's so bad it's worth seeing`），模型被局部正向短语带偏，**最大池容易抓取局部强正向词，忽略全文整体消极基调**。
2. **案例 2（正样本 1 → 预测 0）**
大量负面吐槽（`huge disappointment`、`unbelievable ending`），只是结尾扣分，**全文主体依然好评，但模型被连续负面词汇干扰**。


###### 修改方案   
**首尾状态 + MaxPool + MeanPool**

- MaxPool：抓最强情感词；
- MeanPool：全局平均，约束局部极端词汇，缓解中性文本被个别词语带偏；
同时调整特征融合方式，增加一层 BN+Dropout 轻微正则，降低过拟合；
冻结 GloVe、训练流程全部保留。

   
### GloVe + Attention-LSTM 模型原理
GloVe+Attention-LSTM是预训练静态词向量结合注意力机制长时序网络的文本分类模型，整体分为词向量表征、LSTM时序特征提取、注意力权重优化、分类输出四个阶段。

#### 1. GloVe词向量层
GloVe属于静态预训练词向量，融合全局词语共现统计与局部上下文信息训练得到。文本分词后，每个词语加载预训练GloVe权重转化为低维稠密词向量，按照文本语序依次排列，构成时序向量序列送入LSTM。词向量固定不变，无法区分一词多义，但具备可靠的全局语义表征能力，为后续时序建模提供稳定的基础语义特征。

#### 2. LSTM特征提取层
LSTM对传统RNN存在的梯度消失问题进行优化，依靠遗忘门、输入门、输出门共同调控细胞状态：
- 按照文本顺序逐时序读取GloVe词向量，逐时刻更新隐藏状态；
- 通过三门控机制选择性保留、丢弃历史语义信息，有效捕捉文本长距离上下文依赖；
- 输出每一个词语对应的全部时序隐藏特征序列，保留全文所有位置的语义信息。
相较于普通LSTM，该模型不直接采用最后时刻状态，保留完整时序特征用于注意力加权优化。

#### 3. Attention注意力机制层
普通LSTM仅使用末端隐藏状态，容易丢失前文关键信息。注意力机制通过自适应权重分配优化特征表达：
- 对LSTM输出的每个时序隐藏状态计算重要性得分；
- 通过Softmax归一化得到各词语对应的注意力权重；
- 对所有时序特征进行加权求和，自动强化关键词语义、弱化无效冗余信息。
有效解决长文本信息丢失问题，提升模型对核心语义的聚焦能力，同时具备良好的可解释性。

#### 4. 分类输出层
将注意力加权融合得到的全局最优文本特征送入全连接层，配合激活函数与Softmax，输出文本类别预测结果。

#### 5. 完整流程
1. 文本预处理：分词、去除停用词；
2. 词语映射为GloVe预训练词向量，组成时序向量序列；
3. LSTM逐时序处理向量序列，学习上下文时序语义特征；
4. 注意力机制自适应分配权重，加权聚合全局语义特征；
5. 融合特征传入全连接层，完成情感分类预测。

#### 6. 优缺点
**优点**
- GloVe词向量融合全局统计信息，语义表达质量较高；
- LSTM具备优秀的长距离时序建模能力，对文本语序敏感；
- 注意力机制筛选关键语义，解决传统LSTM尾部偏向、前文信息丢失问题；
- 特征融合更精准，大幅提升长文本分类效果。

**缺点**
- GloVe属于静态词向量，无法处理一词多义现象；
- LSTM串行逐词运算，训练速度相较于CNN模型更慢；
- 注意力机制增加模型参数量，提升了一定的训练开销。

| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | GloVe预训练词向量 + LSTM时序网络 + Attention注意力机制的增强文本分类模型 |
| 词向量表征层 | 采用融合全局词共现统计与局部上下文的GloVe静态词向量；分词词语映射为稠密向量，按语序组成时序序列送入网络，向量固定，无法区分一词多义 |
| 特征提取模块 | LSTM依靠三门控机制捕捉长距离上下文时序依赖；通过Attention机制为不同词语分配自适应权重，聚焦关键语义、过滤冗余信息，优化全局特征表达 |
| 分类输出模块 | 注意力加权融合后的全局文本特征送入全连接层，结合激活函数与Softmax实现文本分类预测 |
| 整体流程 | 文本预处理→词语转换为GloVe时序词向量序列→LSTM提取全时序上下文特征→Attention加权优化特征→全连接层完成情感分类预测 |
| 优势 | GloVe语义表征稳定；LSTM擅长长时序依赖建模；注意力机制解决信息丢失问题，精准捕捉核心语义，分类精度更高 |
| 局限性 | 静态词向量无法解决一词多义；LSTM串行运算训练效率低；注意力结构增加模型计算成本 |  

#### 结果  
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| attention_lstm.csv | 0.87208 |
| attention_lstm(1).csv | 0.89052 |
| attention_;stm.csv | 0.90864 |  

#### 结果分析  
##### 第一版代码  
1. Attention 实现缺陷（最关键）

```
class Attention(nn.Module):
    def forward(self, inputs):
        x = inputs
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        outputs = x * att_score
        return outputs
```

- `softmax(dim=1)`：**在序列维度做 softmax 逻辑没问题，但输出只是加权后的序列，没有求和压缩**
- 返回的依然是 `[seq_len,batch,dim]` 三维张量，后续强行拼接 `attention[0], attention[-1]`，本质还是在用首尾隐状态，**注意力完全没有起到聚合句子表征的作用**，相当于伪注意力。
- 参数初始化方式老旧，没有分离注意力权重与上下文向量。

 2. 模型 Forward 逻辑缺陷

```
states, _ = self.encoder(embeddings.permute(1, 0, 2))
attention = self.attention(states)
encoding = torch.cat([attention[0], attention[-1]], dim=1)
```

1. LSTM 输出 `states.shape=[seq_len,batch,hidden*2]`
2. Attention 只做加权、不求和，依旧保留序列长度
3. 强行取第 0 条与最后一条时间步拼接，**浪费注意力权重**，注意力权重没有聚合整句信息。

> 
> 正确思路：用注意力权重对全部时序加权求和，得到**单个句子向量 [batch,dim]**。

 3. 网络训练相关缺陷

1. `self.embedding.weight.requires_grad = False`：词向量冻结，无法微调 GloVe，表达能力受限；
2. LSTM 没有设置 dropout，缺少正则，容易过拟合；
3. 没有梯度裁剪、没有 L2 权重衰减；
4. 学习率 `lr=0.01` 偏大，容易震荡不收敛；
5. 无学习率调度策略；
6. 输出层只有单层 Linear，缺少激活与 Dropout。
###### 修改方案  
1.新版 Attention：加权求和，输出**固定维度句向量**  
2. 修改 embedding：`requires_grad=True` 开启词向量微调；
3. LSTM 增加 dropout：`dropout=dropout if num_layers>1 else 0`  
4.新增余弦学习率调度器 `CosineAnnealingLR`；  
5.Embedding 输出后增加 `self.dropout`；分类头内部增加 Dropout，增强正则。  
##### 第二版代码  
 1. Attention 模块缺陷

1）**没有 padding mask**
句子长短不一，padding 位置（0）依然参与注意力权重计算。模型会关注无效填充 token，干扰语义表征，降低效果。
2）接口不支持传入 mask，无法屏蔽 padding 位置。
 2. 模型 Decoder 分类头偏弱

原始 decoder：

```
nn.Sequential(
    nn.Linear(lstm_out_dim, lstm_out_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(lstm_out_dim, labels)
)
```

- 缺少归一化（BatchNorm）；
- 使用 ReLU，表达能力弱于 GELU；

3. 损失函数

标准`CrossEntropyLoss`，**没有标签平滑 label smoothing**，容易过拟合、对硬标签过分自信。

 4. 学习率调度策略简单

只用`CosineAnnealingLR`，**缺少 warmup**。训练初期学习率直接拉满，容易震荡、破坏预训练 GloVe 词向量。  
###### 修改方案  
步骤 1：修改超参数区域

新增 / 修改超参：

```
num_epochs = 8 → num_epochs = 12
dropout_rate = 0.3 → dropout_rate = 0.4
# 新增
patience = 3
warmup_epochs = 1
# 新增模型保存路径
BEST_MODEL_PATH = "/kaggle/working/result/best_model.pth"
# 修改输出csv文件名
RESULT_PATH = "/kaggle/working/result/attention_lstm_best.csv"
```

步骤 2：升级 Attention 类（支持 padding mask）

原始 forward：只接收`lstm_output`
新版改动：

1. 增加形参`mask=None`
2. 根据 mask 把 padding 位置分数设置 `-1e4`，softmax 后权重趋近于 0

```
def forward(self, lstm_output, mask=None):
    seq_len, batch, _ = lstm_output.shape
    attn_weights = torch.tanh(self.attn(lstm_output))
    attn_scores = self.v(attn_weights).squeeze(-1)

    # 新增mask逻辑
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask.T, -1e4)

    attn_dist = F.softmax(attn_scores, dim=0)
    weighted = lstm_output * attn_dist.unsqueeze(-1)
    output = torch.sum(weighted, dim=0)
    return output, attn_dist
```

步骤 3：升级 SentimentNet 网络

1. Embedding 增加 `padding_idx=0`

```
self.embedding = nn.Embedding.from_pretrained(weight, padding_idx=0)
```

2. Forward 中构造 padding mask

```
pad_mask = (inputs == 0)
```

3. 调用 attention 时传入 mask

```
attn_pool, _ = self.attention(lstm_out, pad_mask)
```

4. 重构 decoder 序列：

- 增加`nn.BatchNorm1d`
- ReLU → GELU

```
self.decoder = nn.Sequential(
    nn.BatchNorm1d(lstm_out_dim),
    nn.Linear(lstm_out_dim, lstm_out_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(lstm_out_dim, labels)
)
```

步骤 4：新增 warmup 自定义学习率调度函数

在模型定义外部新增函数：

```
def get_warmup_scheduler(optimizer, warmup_epoch, total_epoch):
    def lr_lambda(epoch):
        if epoch < warmup_epoch:
            return (epoch + 1) / warmup_epoch
        else:
            progress = (epoch - warmup_epoch) / (total_epoch - warmup_epoch)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

同时删除原来的`CosineAnnealingLR`，替换调度器：

```
# 删除
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# 修改为
scheduler = get_warmup_scheduler(optimizer, warmup_epochs, num_epochs)
```


### GloVe + CNN-LSTM 模型原理
GloVe+CNN-LSTM是将预训练静态词向量、卷积局部特征提取与循环时序建模相结合的混合文本分类模型，整体分为词向量表征、CNN局部特征提取、LSTM时序建模、分类输出四个阶段。

#### 1. GloVe词向量层
GloVe属于静态预训练词向量，融合全局词语共现统计与局部上下文信息训练得到。文本分词后，每个词语加载预训练GloVe权重转化为低维稠密词向量，按照文本语序依次排列，构成时序向量序列输入后续网络。GloVe词向量固定不变，无法区分一词多义，但具备稳定、可靠的全局语义表征能力，为模型提供基础语义信息。

#### 2. CNN局部特征提取层
CNN利用多尺度卷积核在词向量序列上滑动提取局部特征：
- 通过卷积操作捕捉相邻词汇构成的N-gram短语、局部搭配与关键语义特征；
- 对文本局部语义进行强化筛选，提取细粒度短语特征；
- 相比单纯LSTM，CNN能够快速挖掘局部语义关联，弥补循环网络局部特征捕捉较弱的缺陷。
该层输出富含短语信息的局部特征序列，送入LSTM层进行时序建模。

#### 3. LSTM时序特征提取层
LSTM优化了传统RNN梯度消失问题，通过遗忘门、输入门、输出门调控细胞状态：
- 接收CNN输出的局部特征序列，按语序逐时序更新隐藏状态；
- 选择性保留有效历史信息、丢弃冗余信息，建模长距离上下文依赖关系；
- 融合局部短语特征与全文时序信息，输出全局语义特征。
模型同时具备CNN局部感知能力与LSTM长序列建模能力，特征表达更加全面。

#### 4. 分类输出层
将LSTM输出的深度融合特征送入全连接层，配合激活函数与Softmax，输出最终文本类别预测结果。

#### 5. 完整流程
1. 文本预处理：分词、去停用词、清洗无效字符；
2. 将词语映射为预训练GloVe词向量，构建语义时序序列；
3. CNN卷积核滑动提取文本局部短语特征，生成特征序列；
4. LSTM对局部特征序列进行上下文时序建模，融合全局语义；
5. 特征向量传入全连接层，完成文本分类/情感预测。

#### 6. 优缺点
**优点**
- GloVe依托全局共现统计，词向量语义稳定性强；
- CNN高效捕捉局部短语、关键词特征，细粒度表征能力强；
- LSTM建模长距离时序依赖，充分利用文本语序信息；
- 混合结构兼顾**局部短语特征**与**全局上下文特征**，特征维度更丰富。

**缺点**
- GloVe为静态词向量，无法根据上下文动态解决一词多义问题；
- CNN只能提取固定窗口局部特征，缺乏长距离关联建模能力；
- LSTM串行计算速度慢，整体模型参数量与训练开销大于单一CNN模型。

| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | GloVe预训练词向量 + CNN局部特征提取 + LSTM时序建模的混合文本分类模型 |
| 词向量表征层 | 采用融合全局词共现统计与局部上下文的GloVe静态词向量；分词词语映射为低维稠密向量，按语序组成时序序列，向量固定，无法区分一词多义 |
| 特征提取模块 | CNN利用卷积核提取N-gram局部短语特征，挖掘细粒度语义；LSTM通过三门控机制优化长距离依赖，融合局部与时序全局特征 |
| 分类输出模块 | LSTM融合后的全局特征送入全连接层，结合激活函数与Softmax实现文本分类预测 |
| 整体流程 | 文本预处理→GloVe词向量映射→CNN提取局部短语特征→LSTM建模时序上下文→全连接层输出分类结果 |
| 优势 | 融合CNN局部特征优势与LSTM时序建模优势；词向量语义质量高；特征信息维度丰富，适配复杂文本任务 |
| 局限性 | 静态词向量不具备动态语义能力；模型结构复杂、训练速度慢；CNN窗口固定，灵活性有限 |  
#### 结果  
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| cnn_lstmcsv | 0.59392 |
| cnn_lstm(1).csv | 0.89180 |
| cnn_lstm(2).csv | 0.89976 |    
#### 结果分析  
##### 第一版代码  
1）**只有单一卷积核（filter_size=3）**，只能捕捉 3 词局部特征，无法获取长短不同的短语语义；没有多尺度 CNN。
2）CNN 输出全部下采样后送入 LSTM；**缺少独立全局 CNN 特征分支**；仅依靠 LSTM 首尾隐状态作为最终特征，表征单一。
3）词向量冻结：`embedding.weight.requires_grad = False`，无法微调 GloVe 预训练权重，适配 IMDB 语料能力弱。
4）无 Dropout，缺少正则，极易过拟合；LSTM 内部`dropout=0`。
5）输出维度计算硬编码：仅拼接 LSTM 首尾向量，特征维度低。  
6）优化器使用**SGD**：收敛速度慢，自适应能力弱；
7）学习率 `lr=0.01` 搭配 SGD 容易震荡；**无学习率调度器**；
8） 无梯度裁剪，存在梯度爆炸风险；
9） **不保存最优模型、无早停机制**：训练结束直接使用最后一轮权重预测，最后 epoch 大概率过拟合。  
###### 修改方案  
步骤 1：更新路径与超参

1. 新增最优模型保存路径常量

```
BEST_MODEL_PATH = "/kaggle/working/result/best_model.pth"
```

2. 修改、新增超参数

```
num_epochs = 10 → 15
num_hiddens = 64 → 128
lr = 0.01 → lr = 1e-3
# 单一卷积参数
filter_size = 3
# 修改为多尺度卷积列表
filter_sizes = [3, 4, 5]
# 新增
dropout_rate = 0.3
grad_clip = 5.0
patience = 3
```

3. 设备自动兼容写法

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

步骤 2：重构 SentimentNet 网络类

（1）**init** 参数变更

入参 `filter_size` → `filter_sizes`，新增 `dropout` 参数。

```
def __init__(self, embed_size, num_filter, filter_sizes, num_hiddens, num_layers, bidirectional, weight, labels, dropout=0.3, **kwargs)
```

（2）Embedding 模块升级

```
# 旧
self.embedding.weight.requires_grad = False
# 新
self.embedding.weight.requires_grad = True
self.drop_emb = nn.Dropout(dropout)
```

（3）搭建双分支结构（核心改动）

原始结构：Embedding → 单 Conv1d → MaxPool → LSTM → 首尾拼接 → Linear
新版两条并行分支：

- 分支 A：**多尺度 CNN（3/4/5 卷积核）+ 全局最大池**，提取全局短语特征；
- 分支 B：保留原单尺度卷积 + 池化，送入 LSTM 提取序列时序特征；

```
# 多尺度卷积模块列表
self.convs = nn.ModuleList([
    nn.Conv1d(embed_size, num_filter, fs, padding=fs // 2)
    for fs in filter_sizes
])
# 给LSTM提供序列输入的单尺度卷积
self.conv_for_lstm = nn.Conv1d(embed_size, num_filter, kernel_size=3, padding=1)
```

（4）LSTM 开启内部 dropout

```
dropout=dropout if num_layers > 1 else 0
```

（5）输出层维度修改

分类输入 = LSTM 首尾特征 + 多尺度 CNN 全局特征拼接

```
self.decoder = nn.Linear(lstm_out_dim * 2 + cnn_global_dim, labels)
```

（6）重写 forward 前向传播

1. embedding 后加入 dropout；
2. 循环计算多尺度卷积 + 全局池，拼接得到全局特征；
3. 保留原有卷积、池化送入 LSTM 逻辑；
4. 两路特征融合，融合后增加 dropout 送入分类头。

步骤 3：优化器与学习率调度替换

```
# 旧
optimizer = optim.SGD(net.parameters(), lr=lr)
# 新
optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
# 新增自适应学习率调度器（基于验证集精度调整）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
```
##### 第二版代码  
1. LSTM 时序特征仅简单拼接**首时刻、末时刻隐状态**，忽略句子中间关键情感片段；长文本信息丢失严重。
2. 没有时序注意力机制，无法自动赋予句子重点词语更高权重。
3. 时序特征来源单一，缺少全局平均时序特征作为补充。
4. Dropout 固定 0.3，embedding 层没有单独衰减控制；
5. 学习率 `1e-3`、早停阈值`patience=3`、学习率衰减系数有优化空间。
###### 修改方案  
步骤 1：修改路径与超参数

1. 修改输出文件名

```
RESULT_PATH = "/kaggle/working/result/cnnlstm_att.csv"
```

2. 更新超参

```
lr = 1e-3 → lr = 8e-4
dropout_rate = 0.3 → 0.35
patience = 3 → 4
```

步骤 2：新增独立 Attention 注意力模块

在 SentimentNet 类外部定义注意力类：

```
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: [seq_len, batch, hidden]
        attn_weights = torch.tanh(self.attn(lstm_output))
        score = self.v(attn_weights).squeeze(-1)    # [seq_len, batch]
        alpha = F.softmax(score, dim=0)
        weighted = lstm_output * alpha.unsqueeze(-1)
        attn_out = torch.sum(weighted, dim=0)       # [batch, hidden]
        return attn_out
```

步骤 3：改造 SentimentNet 网络结构

（1）Embedding dropout 调整

```
# 旧
self.drop_emb = nn.Dropout(dropout)
# 新
self.drop_emb = nn.Dropout(dropout * 0.6)
```

（2）注册注意力层

在 LSTM 定义之后添加：

```
self.attention = Attention(lstm_out_dim)
```

（3）修改分类层输入维度

旧：`lstm_out_dim * 2 + cnn_global_dim`
新：`lstm_out_dim * 3 + cnn_global_dim`

```
self.decoder = nn.Linear(lstm_out_dim * 3 + cnn_global_dim, labels)
```

（4）重写 forward 函数 LSTM 分支逻辑（核心改动）

旧逻辑：

```
states, _ = self.encoder(lstm_in)
lstm_feature = torch.cat([states[0], states[-1]], dim=1)
fusion = torch.cat([lstm_feature, cnn_global_pool], dim=1)
```

新版逻辑改动点：

1. 接收 LSTM 完整输出与隐状态 `states, (h_n, _) = self.encoder(lstm_in)`
2. 使用 Attention 对全部时序输出加权得到`attn_feature`
3. 双向 LSTM 最后一层隐状态拼接 `last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)`
4. 新增时序全局平均特征 `mean_seq = torch.mean(states, dim=0)`
5. **三路时序特征 + CNN 全局特征融合**

```
fusion = torch.cat([attn_feature, last_hidden, mean_seq, cnn_global_pool], dim=1)
```

> 
> 时序三路特征说明：
> attn_feature：注意力加权重点时序特征
> last_hidden：句子首尾隐状态
> mean_seq：整段序列平均特征

步骤 4：调整学习率调度器参数

```
# 旧
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
# 新
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=1)
```
## 完整理解

在训练阶段，Word2Vec和GloVe会遍历语料中所有包含该词的上下文窗口，通过不断迭代调整参数，最终把该词在所有不同语境中的共现信息压缩到一个固定的向量点上。训练完成后，词典中每个词语只对应唯一的一个静态词向量。

> **注意：** 这个"压缩"操作发生在训练过程中，由算法自动完成，而不是在使用时由人工操作。训练完成后，我们直接得到的就是那个唯一的静态向量，不存在"多个向量"让你去平均。

在使用阶段，当我们需要把句子输入到CNN、LSTM等深度神经网络时，我们直接去训练好的词向量表里，把每个词对应的唯一静态向量"查"出来，按句子中词的顺序排列成 `[序列长度, 嵌入维度]` 的矩阵。然后将多个这样的矩阵堆叠成一个Batch，形成 `[Batch_Size, 序列长度, 嵌入维度]` 的三维张量，再整体作为输入喂给CNN/LSTM进行有监督训练。   

---

### 总结一句话

Word2Vec/GloVe在训练阶段通过隐含的"上下文压缩"机制为每个词生成唯一的静态向量；在下游任务阶段，我们把一个Batch里所有句子的每个词都查表得到对应的静态向量，堆叠成 `[Batch_Size, 序列长度, 嵌入维度]` 的三维张量，再作为输入喂给CNN/LSTM进行特征提取和训练。  
### 数据流程

原始文本: "I love this movie"
    ↓
分词: ["I", "love", "this", "movie"]
    ↓
编码: [24, 583, 1024, 56]
    ↓
填充/截断: [24, 583, 1024, 56, 0, 0, ...] → 长度固定为512
    ↓
查表（Word Embedding）: 每个ID变成300维向量
    → 单个句子形状 [512, 300]
    ↓
堆叠成Batch: 多个句子堆叠
    → 形状 [Batch_Size, 512, 300]
    ↓
输入CNN/LSTM...模型
    → 模型处理
    → 输出 [Batch_Size, 标签数]（分类任务）        

### GloVe + GRU 模型原理

GloVe + GRU 是将预训练静态词向量与轻量级门控循环网络相结合的文本分类模型，兼顾 GloVe 的全局语义稳定性与 GRU 的高效时序建模能力。整体分为**词向量表征层、GRU 时序特征提取层、分类输出层**三个阶段。

#### 1. GloVe 词向量层

GloVe 融合全局词共现统计与局部上下文信息训练得到静态词向量。文本分词后，每个词语加载预训练 GloVe 权重转化为低维稠密向量，按语序排列构成时序序列输入 GRU 网络。词向量固定不变，无法区分一词多义，但具备稳定、可靠的全局语义表征能力。

#### 2. GRU 时序特征提取层

GRU 是 LSTM 的轻量级变体，将 LSTM 的三个门控精简为**重置门**和**更新门**，取消独立细胞状态，参数量减少约 1/3：

- **重置门**：决定历史信息的遗忘比例，帮助捕捉短期局部模式；
- **更新门**：决定新旧信息的融合比例，缓解梯度消失，建模长距离依赖。

GRU 接收 GloVe 词向量序列，按语序逐时序更新隐藏状态，输出富含上下文依赖的全局语义特征。

#### 3. 分类输出层

取 GRU 最后一个时间步的隐藏状态作为全文语义表征，送入全连接层，经 Softmax 输出各类别预测概率。

#### 4. 完整流程

文本预处理 → GloVe 词向量映射 → GRU 逐时序编码上下文 → 提取最终隐藏状态 → 全连接层分类输出。

#### 5. 优缺点

**优点**
- GloVe 语义稳定，小样本场景表现良好；
- GRU 结构精简，训练速度快，有效缓解梯度消失；
- 保留语序敏感性，适合文本分类任务。

**缺点**
- GloVe 静态向量无法解决一词多义；
- GRU 超长序列记忆能力弱于 LSTM；
- 串行计算，训练效率低于 Transformer。

| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | GloVe 静态词向量 + GRU 门控循环网络 |
| 词向量层 | GloVe 预训练向量，按语序构成时序序列，冻结训练 |
| 特征提取 | GRU 通过重置门与更新门逐时序编码上下文 |
| 分类输出 | 取最终隐藏状态，经全连接层与 Softmax 输出概率 |
| 优势 | 语义稳定、结构轻量、训练快速、缓解梯度消失 |
| 局限性 | 静态词向量无多义区分、超长序列记忆有限、串行计算 |

#### 结果  
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| gru.csv | 0.86216 |
| gru(1).csv | 0.89372 |
| gru(2).csv | 0.90296 |    


#### 结果分析  
##### 第一版代码    

###### 修改建议：  
1. 修正双向 GRU Linear 输入维度不匹配问题（本次报错根源）
2. 移除废弃 Variable
3. 统一使用`.to(device)`兼容 Kaggle GPU/CPU
4. 网络 forward 逻辑与 Linear 维度严格配对

##### 第二版代码   
1. 训练稳定性问题

1）**SGD 优化器 + lr=0.08 仍然容易震荡**，原始任务用 SGD 收敛慢，极易卡在局部最优；
2）没有梯度裁剪，GRU 循环网络容易出现**梯度爆炸**；
3）无学习率调度器，全程固定学习率，后期难以精细收敛；
4）缺少早停（EarlyStop），持续训练会过拟合，白白浪费 epoch。

2. 模型设计缺陷

1）GRU `dropout=0`，无正则，容易过拟合；

> 
> 注意：GRU 的 dropout 仅作用于**层间**，只有`num_layers>1`才生效
> 2）冻结 Embedding，无法微调词向量特征；可以提供开关；
> 3）仅使用最后时刻隐状态，没有 dropout 层、BN 等正则层；

###### 修改建议：   
## 1. 训练策略改动

-  `SGD → Adam`，收敛更快、不容易震荡；
-  **梯度裁剪 `clip_grad_norm_`**，解决 RNN/GRU 梯度爆炸经典问题；
-  学习率衰减 `ExponentialLR`；
-  **早停 Early Stopping**，防止过拟合；
-  保存**验证集最优权重**，预测时加载最优模型（原代码直接用最后 epoch 权重，经常精度更低）

## 2. 模型优化

-  加入随机种子，实验可复现；
-  GRU 支持 dropout 正则；
-  增加`use_emb_finetune`开关，可选微调 GloVe 词向量；



### GloVe + Capsule_LSTM 模型原理

GloVe+Capsule_LSTM是预训练静态词向量结合胶囊网络与长短时记忆网络的文本分类模型，旨在同时捕捉文本的序列依赖关系与局部特征之间的层次结构。模型分为词向量表征、序列上下文建模、胶囊特征路由、分类输出四个阶段。

#### 1. GloVe词向量层

GloVe属于静态预训练词向量，融合全局词语共现统计与局部上下文信息训练得到。

文本经过分词后，每个单词查询预训练GloVe权重，转化为低维稠密词向量；整段文本按词语顺序依次排列，形成词向量序列，作为LSTM的输入。

特点：单词向量固定，不随上下文变化，无法区分一词多义，但相比词袋模型具备基础语义信息，且保留了词语的原始顺序。

#### 2. LSTM上下文建模层

LSTM（长短时记忆网络）通过精巧的门控结构有效缓解传统RNN的梯度消失与梯度爆炸问题，适合建模长序列文本。LSTM在词向量序列上按时间步依次处理：

- **遗忘门**：决定前一时刻细胞状态中哪些信息需要丢弃；
- **输入门**：决定当前时刻的输入信息中哪些需要存入细胞状态；
- **输出门**：基于当前细胞状态决定当前时刻的隐藏状态输出；
- 每个时间步接收当前词语的词向量与上一时刻的隐藏状态和细胞状态，经过门控运算输出当前隐藏状态与更新后的细胞状态；
- 最终将各时间步的隐藏状态汇总（可采用最后一个时刻的输出或全局池化），形成整条文本的序列语义特征向量。

LSTM擅长捕捉长距离词语依赖关系，对语序变化敏感，适合情感信息分散在文本各处的场景；相比GRU，LSTM具有额外的细胞状态，能更精细地控制信息流动。

#### 3. 胶囊特征路由层

在LSTM提取的序列特征基础上，引入胶囊网络（Capsule Network）进行更高阶的特征组织：

- 将LSTM各时间步的隐藏状态视为底层胶囊（Primary Capsules），每个胶囊编码对应位置的局部上下文特征；
- 通过**动态路由算法（Dynamic Routing）**，底层胶囊向高层胶囊传递信息，计算耦合系数，迭代更新路由权重；
- 高层胶囊的输出向量模长表示对应类别特征存在的概率，向量方向编码该类别的实例化参数（如情感强度、语义属性等）；
- 胶囊网络能够捕捉特征之间的**部分-整体层次关系**，例如短语组合、情感极性在不同粒度上的层次结构，弥补LSTM对局部特征间内在关联建模能力的不足。

#### 4. 分类输出层

将胶囊网络输出的高层胶囊向量（通常取模长或拼接）送入全连接层，结合激活函数与Softmax运算，输出文本所属类别。

#### 5. 完整流程

1. 文本预处理：分词、去除停用词；
2. 词语映射为GloVe预训练词向量，构成词向量序列；
3. LSTM按时间步依次处理词向量，通过门控机制建模上下文依赖，输出各时间步隐藏状态序列；
4. 将隐藏状态序列作为底层胶囊，通过动态路由算法迭代计算高层胶囊特征；
5. 高层胶囊特征送入全连接层，完成情感分类预测。

#### 6. 优缺点

**优点**
- GloVe提供带有全局统计信息的词向量，语义表征优于Word2Vec；
- LSTM通过门控机制有效捕获长距离依赖，对语序变化敏感；
- 胶囊网络通过动态路由捕获特征间的层次结构与部分-整体关系，在细粒度情感分析、讽刺检测等任务上有优势；
- 胶囊网络的向量输出比标量输出包含更丰富的语义信息（如情感强度、方向）。

**缺点**
- GloVe为静态词向量，不能解决一词多义；
- LSTM按时间步串行计算，训练并行性较差，大规模数据上训练速度慢；
- 胶囊网络的动态路由算法计算开销较大，迭代过程增加训练和推理时间；
- 模型整体复杂度高，参数量大，对数据量要求较高，小数据集上易过拟合；
- 胶囊网络在文本领域的优势相比图像领域尚缺乏广泛验证，实现复杂度较高。

| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | GloVe预训练词向量 + LSTM循环网络 + 胶囊网络（Capsule Network），静态词向量搭配序列建模与层次特征路由的文本分类模型 |
| 词向量表征层 | 使用融合全局词共现统计与局部上下文信息的GloVe静态词向量；分词后的单词映射为稠密向量，按顺序构成词向量序列作为网络输入，向量固定，无法区分一词多义，但保留语序信息 |
| 序列特征提取 | LSTM通过遗忘门、输入门、输出门对词向量序列进行时间步递推建模，有效缓解梯度消失问题；输出各时间步隐藏状态序列，捕捉长距离依赖关系，对语序变化敏感 |
| 胶囊路由层 | 将LSTM隐藏状态序列作为底层胶囊，通过动态路由算法迭代计算高层胶囊特征；胶囊向量模长表示特征存在概率，方向编码实例化参数，捕获特征间部分-整体层次关系与细粒度语义差异 |
| 分类输出模块 | 高层胶囊特征经聚合后送入全连接层，结合激活函数与Softmax得到文本分类结果 |
| 整体流程 | 文本预处理→词语转换为GloVe词向量构成输入序列→LSTM建模上下文依赖输出隐藏状态序列→动态路由提取层次胶囊特征→全连接层完成情感分类预测 |
| 优势 | GloVe语义表征质量优于Word2Vec；LSTM擅长长距离依赖和语序建模；胶囊网络捕获层次结构与部分-整体关系，向量输出语义更丰富；适合细粒度情感分析等复杂任务 |
| 局限性 | 静态词向量无法处理一词多义；LSTM串行计算训练速度慢；动态路由计算开销大，训练推理时间成本高；模型复杂度高，小数据集易过拟合；胶囊网络在文本任务上的优势尚需更多验证 |    

#### 结果  
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| capsule_lstm.csv | 0.50000 |
| capsule_lstm(1).csv | 0.88576 |
| capsule_lstm(2).csv | 0.89096 |      
| capsule_lstm(3).csv | 0.89864 |   


#### 结果分析
##### 第一版代码   
1. **修复 Capsule 层致命维度错误**
   - LSTM 输出 shape 是 `(seq_len,batch,dim)`，capsule 需要 `(batch,seq_len,dim)`，增加`permute`
   - 修正 W 权重矩阵维度，原来写死 num_hiddens*2 会在双向开关切换时报错
   - 修复解码层输入维度：把 capsule 输出 flatten，不再错误取`capsule[0],capsule[-1]`拼接（原代码逻辑错误）
   - dim_capsule 调整为 64 适配
2. **移除废弃 Variable**，直接使用`.to(device)`，兼容现代 PyTorch
3. 区分`net.train()` / `net.eval()`，验证测试关闭 dropout 相关行为
4. loss 使用`.item()`，不再存 tensor 累加，防止显存缓慢上涨

##### 第二版代码  
1. **文本预处理增强**：数字、特殊符号清洗；不做粗暴只保留 a‑z，保留有用缩写；小写；过滤极短噪声词
2. **Embedding 层优化**：padding 位置置零梯度；加入 Scale；GloVe OOV 初始化高斯分布，不再全部 0
3. **模型结构升级**：BiLSTM 输出后加一层 Projection；Capsule 增加层归一化；残差连接；多头池化（max/avg/min）；Dropout 改为 Dropout1D 针对时序；加入 LayerNorm 抑制内部协变量偏移
4. **损失与训练策略**：标签平滑缓解过拟合；更强梯度裁剪；warmup 学习率；CosineAnnealingWarmRestarts+ReduceLROnPlateau 双调度；混合精度加速；权重初始化精细控制
5. **数据集**：固定随机种子全部模块完全可复现；增加验证集指标打印；最优模型保存；早停逻辑加固

##### 第三版代码  
1. embedding**允许微调**（原版冻结 embedding 是分数低主要原因！设置`requires_grad=True`）
2. LSTM 增加 dropout，加入 Dropout 层防过拟合
3. 胶囊网络路由逻辑修复，增加 mask，降低 capsule 输出维度
4. 加入**早停**，保存最优模型，不是最后一轮
5. 学习率调度 ReduceLROnPlateau，梯度裁剪防止梯度爆炸
6. AdamW 优化器，权重衰减
7. 混合池化：capsule 输出 + 时序最大 / 平均池化拼接，提升特征

### Transformer 模型原理

Transformer 是一个完全基于**自注意力（Self-Attention）机制**的序列到序列模型，彻底摒弃了传统的循环（RNN）和卷积（CNN）结构。其核心创新在于通过“全局注意力”一次性捕捉序列中任意两个位置之间的依赖关系，解决了 RNN 难以并行计算和 CNN 感受野受限的痛点。整体分为**输入嵌入层、编码器（Encoder）堆栈、解码器（Decoder）堆栈**以及**输出生成层**四个核心阶段。

#### 1. 输入嵌入与位置编码层
与 RNN 天然具备时序敏感度不同，Transformer 的注意力机制是“无序”的（对位置不敏感）。因此，模型在输入阶段进行双重处理：
- **词嵌入（Embedding）**：将输入文本的每个 Token（词或子词）映射为高维稠密向量（类似静态词向量，但在训练中会动态微调）。
- **位置编码（Positional Encoding）**：使用正弦/余弦函数生成与词向量维度相同的“位置向量”，并**直接叠加**到词嵌入上。这使得模型在计算注意力时，能明确区分“第一个词”和“第二个词”的语序信息，弥补了注意力机制对位置不敏感的缺陷。

#### 2. 编码器（Encoder）堆栈层
编码器负责将输入序列（如“我/爱/中国”）转化为富含上下文语义的“隐藏表征”。它通常由 N 个（原论文为 6 个）结构完全相同的子层堆叠而成，每个子层包含两个主要模块：
- **多头自注意力（Multi-Head Self-Attention）**：这是 Transformer 的核心。对于序列中的每一个词，模型通过 Q（查询）、K（键）、V（值）矩阵的映射和点积计算，计算出该词与序列中**所有其他词**的注意力权重。多头机制让模型从多个不同的子空间进行注意力计算，捕捉词语之间多种维度的关系（如语法关系、指代关系、语义相关性）。
- **前馈神经网络（FFN）**：一个简单的全连接前馈网络，对每个位置的注意力输出进行独立的非线性变换，增强模型的表达能力。
- **残差连接与层归一化（Add & Norm）**：每个子层前后都包裹了残差连接（防止梯度消失）和层归一化（加速收敛），确保深层网络的训练稳定性。

#### 3. 解码器（Decoder）堆栈层
解码器负责根据编码器输出的“记忆”和已经生成的目标端内容，逐步生成输出序列（如将“我/爱/中国”翻译为英文）。它由 N 个结构相似的子层堆叠，但在编码器的基础上**多了一个交叉注意力子层**：
- **掩码多头自注意力（Masked Self-Attention）**：与编码器相同，但加上了掩码（Mask），防止解码器在预测第 i 个词时“偷看”到第 i 个词之后的位置（即未来信息），确保预测只依赖于已经生成的前文。
- **编码器-解码器交叉注意力（Cross-Attention）**：该层的 Q 来自解码器上一层的输出，而 K 和 V 来自**编码器最终的输出**。这使得解码器在生成每一个词时，都能主动“查询”并聚焦于输入源端序列中最相关的部分（类似人类翻译时回头看原文的习惯）。

#### 4. 输出生成层
将解码器堆栈输出的最终特征向量，通过一个线性全连接层（Linear）映射到目标词汇表大小的维度，再配合 Softmax 激活函数，将分数转化为概率分布，从而输出当前步最可能的目标单词（如输出 "I"、"love"、"China"）。

#### 5. 完整工作流程（以机器翻译为例）
1. **输入源文本**：将输入序列（中文）进行嵌入并叠加位置编码；
2. **编码双向语义**：编码器通过多头自注意力，生成携带全文双向信息的上下文表征矩阵；
3. **初始化解码**：将目标端起始符 `[SOS]` 输入解码器，结合编码器记忆，预测第一个目标词（如 "I"）；
4. **循环自回归生成**：将已生成的 "I" 作为新输入，重复送入解码器，结合编码器记忆预测 "love"，再拼接预测 "China"，直到预测出终止符 `[EOS]`；
5. **输出结果**：最终得到完整的翻译序列。

#### 6. 优缺点
**优点**
- 全局依赖捕捉：自注意力机制直接建模序列中任意距离的词语关系，完美解决了 RNN 的长距离遗忘问题；
- 高度并行化：训练时无需像 RNN 那样逐时序串行计算，所有位置可同时计算注意力，极大地利用 GPU 算力，训练速度远快于 LSTM；
- 可解释性强：注意力权重矩阵可视化后可以清晰展示模型在进行预测时关注了输入的哪些部分，具备良好的可解释性。

**缺点**
- 计算复杂度极高：自注意力的复杂度为序列长度 L 的平方级 O(L²)，在处理极长文档时显存占用和计算开销骤增；
- 位置编码局限性：尽管加入了位置编码，但相比 RNN 的天然递归结构，其对绝对/相对位置的建模能力仍不够“内生化”；
- 缺乏归纳偏置（Inductive Bias）：CNN 天然具备局部性先验，RNN 天然具备时序先验，而 Transformer 完全依赖数据去学习结构，极度依赖海量数据和大算力，在小数据集上容易过拟合。

| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | 基于纯注意力机制的 Seq2Seq 模型，由多头自注意力、前馈网络及编码器-解码器结构组成 |
| 输入特征层 | 词嵌入结合正余弦位置编码，为静态无序的注意力机制注入语序信息 |
| 特征提取模块 | 编码器利用多头自注意力提取全局双向语义依赖；解码器利用掩码自注意力和交叉注意力，结合已生成内容与源端信息进行自回归生成 |
| 输出生成模块 | 解码器最终输出经过线性层与 Softmax 映射，生成目标词汇的概率分布，实现文本生成或翻译 |
| 整体流程 | 源端嵌入与位置编码 → 编码器生成语义表征 → 解码器交叉注意力结合源端语义与目标端前缀 → 自回归逐词生成输出结果 |
| 优势 | 完美解决长距离依赖；训练高并行度、速度快；特征提取能力强，为后续 BERT、GPT 等大模型奠基 |
| 局限性 | 长文本计算量呈平方级增长；位置编码不如 RNN 结构自然；极度依赖大规模语料和硬件资源 |   

#### 结果  
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| transformer.csv | 0.70180 |
| transformer(1).csv | 0.84016 |
| transformer(2).csv | 0.84184 |    

#### 结果分析  
##### 第一版代码  
1. 取`hidden_states[0,:,:]`直接拿**第一个时间步**做分类，这是错误用法，Transformer 没有`<cls>`标记，第一个位置只是普通 token，语义代表性差；改为**句子长度均值池化**。
2. 词汇表`min_freq=1`，低频词过多，参数量大、噪声多，调大`min_freq=5`过滤罕见词。
3. 缺少 embedding 缩放，原始 Transformer 论文 embedding 需要乘 \(\sqrt{d_{model}}\)。
4. 学习率固定，增加**学习率预热 + 衰减**，Transformer 对 lr 非常敏感。
5. 增加权重衰减 weight_decay，防止过拟合；加入梯度裁剪防止梯度爆炸。
###### 修改建议：  
1. **移除 log_softmax 输出**：`CrossEntropyLoss = log‑softmax + NLLLoss`，之前模型输出 log_softmax，相当于两次 log，loss 完全错误。现在直接返回 logits。
2. **均值池化代替取第 0 个 token**：Transformer 没有`<cls>`符号，取第一个 token 语义无效，用真实非 padding 位置求平均，这是最大精度提升点。
3. embedding 乘 \(\sqrt{d_{model}}\)，严格遵循 Transformer 原始论文。
4. 增加 AdamW 权重衰减、梯度裁剪，防止过拟合、梯度爆炸。
5. 学习率 warmup，Transformer 必须预热，否则训练震荡不收敛。
##### 第二版代码  
1. 纯 Transformer 小数据集 IMDB 很容易**过拟合**；3 层 encoder 容量偏大，加上 warmup 学习率策略对小语料不稳定，验证集反而震荡下跌。
2. 简单停用词过滤会删掉部分情感关键副词，损伤特征，不要强行删 stopword。
3. 均值池化虽比取第 0 位好，但单纯均值会被大量无关 token 稀释；改用**最大 + 均值拼接池化**效果更强。
4. AdamW+warmup 是大模型策略，25k 样本小数据集，warmup 容易前期学习率过低收敛慢。
5. 缺少权重初始化；positional encoding+embedding 后 dropout 力度不足。
###### 修改建议：  
1. **删除停用词过滤**：IMDB 情感大量依赖否定词 (not, never, don’t)，删停用词直接破坏语义，分数暴跌元凶。
2. **均值 + 最大拼接池化**：mean 抓全局语义，max 抓强情感关键词，比单纯均值显著提升。
3. 弃用 Transformer 论文式 warmup；改用`ReduceLROnPlateau`，验证集不涨自动降 lr，小数据集更稳定。
4. 改回`num_layers=2`，降低模型容量，对抗过拟合；提升 dropout=0.3，增大 weight_decay。
5. 增加 xavier 权重初始化；全套随机种子锁死，结果可复现。
6. mask‑max 池化正确处理 padding，padding 位置设置‑inf，不会被 max 取到 padding 零向量。


## bert_native 模型原理

bert_native 是基于 BERT 预训练语言模型并采用**手动训练循环**的文本分类实现，整体分为 BERT 动态词向量编码、分类输出、手动训练控制三个阶段。

#### 1. BERT 词向量编码层

BERT（Bidirectional Encoder Representations from Transformers）基于 Transformer Encoder 架构，通过多层双向自注意力机制对输入文本进行深度编码。文本经 WordPiece 分词后，使用 `BertTokenizerFast` 将词语转换为 `input_ids` 和 `attention_mask`，送入预训练的 `bert-base-uncased` 模型。BERT 的核心优势在于生成**动态上下文词向量**——同一词语在不同语境下获得不同的表征，有效解决一词多义问题。模型输出 `[CLS]` 位置的向量作为整个句子的语义摘要，该向量融合了全文双向上下文信息。

#### 2. 分类输出层

`BertForSequenceClassification` 在 BERT 主体之上自动添加了分类头：取 `[CLS]` 向量经 Dropout 正则化后，送入全连接线性层将隐藏维度映射到二分类输出空间（`num_labels=2`），并通过交叉熵损失函数计算分类损失。整个过程由官方模型封装，无需手动实现分类器构造与损失计算。

#### 3. 手动训练流程

与使用高级 API 不同，本实现**完全手动编写训练循环**。自定义 `torch.utils.data.Dataset` 封装编码后的数据，使用 `DataLoader` 按批次加载（训练 batch_size=8，验证 batch_size=16），采用 `AdamW` 优化器（学习率 5e-5），训练 3 个 epoch。每个 epoch 内手动执行：前向传播 → 损失计算 → 反向传播 → 梯度更新。通过 `tqdm` 在每个 batch 和 epoch 结束时实时打印训练/验证损失与准确率，训练结束后手动遍历测试集生成预测结果并保存为 CSV。

#### 4. 完整流程

1. 文本数据读取与预处理，按 8:2 划分训练集与验证集；
2. 使用 BERT Tokenizer 对文本进行 WordPiece 分词、编码、填充与截断；
3. 自定义 Dataset 类封装编码后的数据；
4. 手动循环 3 个 epoch，逐批次执行前向/反向传播更新参数；
5. 每个 epoch 结束后在验证集上评估性能；
6. 训练完成后对测试集进行预测，输出 CSV 结果文件。

#### 5. 优缺点

**优点**
- 完全控制训练细节，便于插入自定义逻辑（如梯度裁剪、自定义学习率调度）；
- 适合调试、教学或对训练过程有特殊需求的场景；
- BERT 动态词向量能力优于静态 GloVe，可有效处理一词多义。

**缺点**
- 代码量大，需手动管理设备转移、梯度清零、损失累加等；
- 缺乏高级 API 的自动化优化（如混合精度训练、分布式支持、早停等）；
- 训练效率较低，易引入人为编码错误。


| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | 预训练 BERT 编码器 + 官方分类头 + 手动训练循环 |
| 词向量表征层 | 采用 BERT 动态上下文词向量，同一词语在不同语境下表征不同；使用 WordPiece 子词分词，有效解决一词多义及未登录词问题 |
| 特征提取模块 | BERT 多层双向 Transformer Encoder 通过自注意力机制同时捕捉双向上下文依赖，输出 `[CLS]` 向量作为句子级表征 |
| 分类输出模块 | `[CLS]` 向量经 Dropout 后送入线性分类层，通过交叉熵损失优化二分类任务 |
| 整体流程 | 文本分词编码 → BERT 提取动态上下文表征 → 分类头输出预测 → 手动循环训练优化 |
| 优势 | BERT 动态词向量语义表征能力强；手动训练灵活可控；适合学习 Transformer 训练底层流程 |
| 局限性 | 代码冗余度高；缺乏高级 API 自动化优化；大规模实验效率偏低 |    

#### 结果  
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| bert_native.csv | 0.87996 |
| bert_native(1).csv | 0.88540 |
| bert_native(2).csv | 0.91692 |   


## bert_scratch 模型原理

bert_scratch 的核心特点是**从零自定义 BERT 分类模型结构**（继承 `BertPreTrainedModel`），配合 Hugging Face 的 `Trainer` 高级训练 API 完成训练。整体分为自定义 BERT 分类模型构建、数据预处理与 Trainer 自动训练三个阶段。

#### 1. 自定义 BERT 分类模型（BertScratch）

与直接使用 `BertForSequenceClassification` 不同，本模型手动构建了分类头部。继承自 `BertPreTrainedModel`，保留预训练 BERT 的权重加载能力。在 `__init__` 方法中显式定义：
- `self.bert = BertModel(config)` —— BERT 主体编码器；
- `self.dropout = nn.Dropout(classifier_dropout)` —— 防止过拟合；
- `self.classifier = nn.Linear(config.hidden_size, config.num_labels)` —— 线性分类层。

在 `forward` 方法中手动实现前向逻辑：调用 `self.bert` 获取输出，取 `pooled_output`（即 `[CLS]` 向量），经 Dropout 后送入分类器得到 `logits`，显式调用 `nn.CrossEntropyLoss` 计算损失，并封装为 `SequenceClassifierOutput` 返回。

#### 2. BERT 编码层

与 bert_native 一致，使用预训练的 `bert-base-uncased` 权重初始化 BERT 主体，通过多层双向自注意力生成上下文相关的动态词向量。文本经 WordPiece 分词后编码为 `input_ids` 和 `attention_mask`，BERT 输出 `[CLS]` 向量作为句子级表征供分类头使用。

#### 3. Trainer 高级训练流程

本实现使用 Hugging Face 的 `Trainer` API 自动化管理训练过程。使用 `datasets.Dataset` 封装数据，配合 `map` 函数批量分词；采用 `DataCollatorWithPadding` 动态填充批次序列以提升效率；通过 `TrainingArguments` 配置训练参数（epochs=3，batch_size=6/12，权重衰减等）。`Trainer` 自动执行训练循环、梯度更新、验证评估、日志记录与指标计算。自定义 `compute_metrics` 函数使用准确率评估验证集性能，训练完成后直接调用 `trainer.predict()` 生成测试集预测。

#### 4. 完整流程

1. 数据读取并按 8:2 划分训练/验证集；
2. 将 Pandas DataFrame 转换为 `datasets.Dataset` 格式；
3. 使用 BERT Tokenizer 批量分词编码；
4. 自定义 `BertScratch` 类，手动构建 BERT 分类模型结构；
5. 配置 `TrainingArguments`，初始化 `Trainer`，传入模型、数据集、数据整理器、评估函数；
6. 调用 `trainer.train()` 自动完成 3 个 epoch 的训练与验证；
7. 调用 `trainer.predict()` 生成测试预测并保存结果。

#### 5. 优缺点

**优点**
- 清晰展示 BERT 分类模型内部的构造细节（Dropout、分类器、损失计算），适合学习；
- `Trainer` API 大幅简化训练流程，自动处理设备转移、批次迭代、日志保存等；
- 代码量适中，兼顾可读性与工程效率。

**缺点**
- 自定义分类头与官方 `BertForSequenceClassification` 功能完全一致，实际项目无必要；
- 相比直接使用官方模型，增加了维护成本和出错风险；
- 未充分利用 `Trainer` 的全部高级功能（如回调、超参数搜索等）。


| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | 自定义继承 `BertPreTrainedModel` 的分类模型 + Trainer 训练 API |
| 词向量表征层 | BERT 动态上下文词向量，基于 WordPiece 分词，支持一词多义与未登录词处理 |
| 特征提取模块 | BERT 主体（`BertModel`）提取双向上下文特征，输出 `pooled_output` 作为句子级表征 |
| 分类输出模块 | 手动构建 `Dropout` + `Linear` 分类头，显式计算交叉熵损失，封装为标准输出格式 |
| 整体流程 | 文本分词编码 → BERT 提取动态表征 → 手动构建分类头输出 → Trainer 自动训练优化 |
| 优势 | 揭示 BERT 分类模型内部实现细节；`Trainer` 提升训练效率与自动化程度；适合教学理解 |
| 局限性 | 自定义分类头冗余，与官方实现功能重复；工程实用性低于直接使用官方模型 |    

#### 结果  
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| bert_scratch.csv | 0.91948 |
之后的改动均为分数降低，不再记录


## bert_trainer 模型原理

bert_trainer 是**最简洁、最推荐**的实现方式，直接使用 Hugging Face 官方提供的 `BertForSequenceClassification` 预训练模型，配合 `Trainer` 高级训练 API，以最少代码完成高质量文本分类。整体分为 BERT 编码层、官方分类头、Trainer 标准化训练三个阶段。

#### 1. BERT 编码层

与上述两个模型一致，采用 `bert-base-uncased` 预训练权重。通过 `BertTokenizerFast` 对文本进行 WordPiece 分词、编码、填充与截断；BERT 的多层双向自注意力机制生成**动态上下文词向量**，每个 token 的表示随语境动态变化；输出 `[CLS]` 位置的向量作为整个句子的语义摘要。

#### 2. 官方分类头

直接加载 `BertForSequenceClassification.from_pretrained('bert-base-uncased')`，该模型包含 BERT 主体编码器、预置分类头（`Dropout` → `Linear(hidden_size, num_labels)`）以及内置的交叉熵损失计算逻辑，无需手动实现分类器构造与损失计算。

#### 3. Trainer 标准化训练

采用 `Trainer` API 自动化训练：使用 `datasets.Dataset` 和 `DataCollatorWithPadding` 高效管理数据；通过 `TrainingArguments` 配置超参数（batch_size=16/32，较前两者更大）；自动执行多轮训练、验证评估、指标计算、日志保存；自定义 `compute_metrics` 计算准确率用于验证集评估。所有训练过程由 `Trainer` 统一管理，无需编写循环代码。

#### 4. 完整流程

1. 数据读取、划分训练/验证集；
2. 转换为 `datasets.Dataset` 格式；
3. 批量分词编码；
4. **一步加载**官方 `BertForSequenceClassification` 预训练模型；
5. 配置 `TrainingArguments`，初始化 `Trainer`；
6. 调用 `trainer.train()` 完成 3 个 epoch 训练；
7. 调用 `trainer.predict()` 生成测试集预测并输出结果。

#### 5. 优缺点

**优点**
- **代码最简洁**，仅需约 60 行即可完成完整训练与预测；
- 官方 `BertForSequenceClassification` 经大量验证，稳定可靠；
- `Trainer` 内置混合精度训练、分布式训练、梯度累积等高级特性，可无缝扩展；
- 最适合生产环境、竞赛、快速原型验证。

**缺点**
- 封装程度高，对初学者来说"黑盒"感较强，不利于理解底层细节；
- 自定义灵活性相对较低（如需修改损失函数或模型结构，需额外处理）。


| 项目 | 内容 |
| ---- | ---- |
| 模型架构 | 官方 `BertForSequenceClassification` 预训练模型 + Trainer 高级训练 API |
| 词向量表征层 | BERT 动态上下文词向量，通过 WordPiece 分词和双向自注意力实现语义感知 |
| 特征提取模块 | 多层 Transformer Encoder 提取双向上下文特征，`[CLS]` 向量聚合为句子级表征 |
| 分类输出模块 | 官方内置分类头（Dropout + Linear）；损失计算内置于模型前向过程 |
| 整体流程 | 文本分词编码 → BERT 提取动态表征 → 官方分类头输出预测 → Trainer 全自动训练优化 |
| 优势 | 代码量最少，开发效率最高；充分利用官方模型稳定性与 `Trainer` 高级优化功能；适合快速部署与生产落地 |
| 局限性 | 高度封装不利于教学理解；自定义损失函数或模型结构需额外操作 |     

#### 结果   
| 实验版本 | 测试集准确率 |
| ---- | ---- |
| bert_trainer.csv | 0.93236 |
| bert_trainer(1).csv | 0.93884 |
  



## 三个模型的综合对比

| 维度 | bert_native | bert_scratch | bert_trainer |
|------|-------------|--------------|---------------|
| **模型来源** | 官方 `BertForSequenceClassification` | 自定义 `BertScratch`（继承 `BertPreTrainedModel`） | 官方 `BertForSequenceClassification` |
| **分类头实现** | 官方内置 | 手动编写（Linear + Dropout + CrossEntropyLoss） | 官方内置 |
| **训练流程** | 手动 `for` 循环 | Hugging Face `Trainer` | Hugging Face `Trainer` |
| **数据格式** | 自定义 `torch.utils.data.Dataset` | `datasets.Dataset` | `datasets.Dataset` |
| **批次填充** | 统一 padding（固定长度） | `DataCollatorWithPadding`（动态填充） | `DataCollatorWithPadding`（动态填充） |
| **代码复杂度** | 高（约 150 行） | 中（约 100 行） | 低（约 60 行） |
| **适用场景** | 教学 / 需要精细控制训练细节 | 学习 BERT 内部结构 | 生产 / 竞赛 / 快速实验 |
| **底层原理** | BERT 动态词向量 + 分类微调 | BERT 动态词向量 + 分类微调 | BERT 动态词向量 + 分类微调 |

## 总结

三个模型的 BERT 底层原理完全一致，差异仅在于：

1. **如何构建分类头**——官方内置 vs 手动编写；
2. **如何管理训练过程**——手动循环 vs Trainer API。

推荐优先使用 **bert_trainer** 方式，因其兼顾开发效率与模型可靠性，是 Hugging Face 生态的标准实践。



    


  
