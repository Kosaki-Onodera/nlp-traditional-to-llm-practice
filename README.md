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
[32, 512, 300] ← 最终输入模型的形状

### 一、CNN 文本卷积神经网络
#### 1. 核心功能
TextCNN 是适配文本任务的轻量化卷积网络，由图像CNN优化改造而来，核心作用是高效提取文本局部组合特征，聚焦词语搭配、短句语义，不依赖上下文记忆与语序逻辑，适配文本短语级特征挖掘，多用于情感分类任务。
#### 2. 详细工作机制
- 输入结构：以 GloVe 预训练词向量构成的二维矩阵为输入，矩阵行数为句子单词数，列数为词向量维度。
- 多尺度卷积核滑动采样：设置 2、3、4 等不同尺寸的卷积核，分别对应二元、三元、多元词语组合，模拟人工提取 n-gram 语法特征，可精准捕捉 “not happy”“very wonderful” 等关键情感短语。
- 卷积特征计算：卷积核在词向量矩阵上逐行滑动，对窗口内局部词向量加权运算，提取相邻词语的组合语义特征。

$$
c_i=\text{ReLU}(W\cdot E_{i:i+k-1}+b)
$$

$W$ 代表卷积核权重，$E_{i:i+k-1}$ 为窗口内词向量，经过ReLU激活得到一组局部特征。
- 最大池化降维筛选：对卷积生成的特征图进行全局最大池化，保留每组特征里数值最大的有效信息，过滤冗余特征，同时完成维度压缩。

$$
\hat{c}=\max\{c\}
$$

- 特征融合分类：将2、3、4尺度卷积核池化后的特征拼接融合，输入全连接层，完成情感正负分类。
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


### 三、CNN-LSTM 混合特征提取模型
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


### 四、Attention-LSTM 注意力增强时序模型
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

### 五、Transformer 自注意力序列建模网络

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




    


  
