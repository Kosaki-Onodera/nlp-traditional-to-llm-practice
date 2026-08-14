# nlp-traditional-to-llm-practice
本项目基于IMDB电影评论情感二分类任务，复现NLP发展早期经典模型，完整演示**词袋模型 → Word2Vec → GloVe预训练词向量 + 深度学习网络**的技术路线，梳理静态Embedding发展脉络，为后续学习大模型LLM动态嵌入打下基础。
实验环境：Kaggle Notebook（GPU加速），Python3，代码由原始Python2.7教程迁移改造。

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
###  一、TextCNN 文本卷积神经网络
####  1. 核心功能
TextCNN 是适配文本任务的轻量化卷积网络，由图像 CNN 优化改造而来，核心作用是高效提取文本局部组合特征，聚焦词语搭配、短句语义，不依赖上下文记忆与语序逻辑，适配文本短语级特征挖掘。
####  2. 详细工作机制
- 输入结构：以 GloVe 预训练词向量构成的二维矩阵为输入，矩阵行数为句子单词数，列数为词向量维度。
- 多尺度卷积核滑动采样：设置 2、3、4 等不同尺寸的卷积核，分别对应二元、三元、多元词语组合，模拟人工提取 n-gram 语法特征，可精准捕捉“not happy”“very wonderful”等关键情感短语。
- 卷积特征计算：卷积核在词向量矩阵上逐行滑动，对局部词向量做加权运算，提取每一段相邻词语的组合语义特征，生成多维特征图。
- 最大池化降维筛选：对卷积生成的特征图进行全局最大池化操作，自动保留每类特征中响应最强、语义贡献最大的关键特征，过滤冗余无效信息，同时实现特征降维。
- 特征融合分类：将多尺度卷积核提取的特征拼接融合，输入全连接层，完成情感正负分类。
####  3. 优缺点分析
- 优点：网络结构简单，全程并行计算，训练速度快、模型收敛稳定；对短句、固定情感短语的识别精度高，泛化能力优秀。
- 缺点：卷积核采样范围固定，仅能捕捉局部相邻词语的关联特征，无法建模长距离语义依赖，对长文本、逻辑跨度大、前后呼应的句子建模效果较差，且完全不关注词语语序信息。
| 项目 | 内容 |
| ---- | ---- |
| 核心功能 | 面向文本任务的轻量化卷积网络，专注抽取局部词语组合特征，捕获短语层面语义信息。 |
| 输入形式 | 词向量拼接形成二维矩阵，矩阵维度为句子长度 × 词向量维度。 |
| 特征提取流程 | 通过尺寸2、3、4的卷积核滑动提取多元词组特征；卷积得到特征图后利用全局最大池化筛选有效特征、压缩维度；融合多尺度特征送入全连接层完成分类。 |
| 计算特性 | 支持并行运算，训练效率高。 |
| 优势 | 模型结构简洁，收敛稳定；擅长识别固定情感短语，短句分类效果较好。 |
| 局限性 | 感受野有限，难以捕捉远距离词语关联；对语序不敏感，长文本语义建模能力不足。 |
### 二、STM 长短期记忆网络（RNN 优化变体）
#### 1. 核心功能
LSTM 是为解决原生 RNN 梯度消失、长文本记忆失效问题专门设计的改进型循环网络，在保留 RNN 时序建模、语序记忆优势的基础上，通过可控门控机制，同时兼顾短期实时语义与长期关键语义，稳定适配长短文本的特征提取任务。
#### 2. 核心结构与详细工作机制
LSTM 摒弃 RNN 单一隐藏状态结构，引入细胞状态（长期记忆载体）和三重门控机制，精准控制信息的留存、更新与输出，全程保护长距离语义信息：
- 遗忘门：核心作用是过滤冗余旧信息。通过sigmoid激活函数生成0-1的权重系数，自动判断前文记忆中无用的虚词、冗余语义、无效上下文信息，并选择性丢弃，保留核心长期记忆。
- 输入门：核心作用是更新实时语义。一方面筛选当前时刻新词的有效语义特征，另一方面通过tanh函数生成候选记忆信息，将有效新词语义更新到细胞状态中，实现记忆迭代。
- 输出门：核心作用是融合输出特征。结合当前细胞状态的长期记忆与当前时刻的短期语义，筛选出适配当前语境的最优特征，输出当前时刻的隐藏状态，用于后续特征融合与分类。
整套门控机制协同工作，让网络自主学习“该记什么、该忘什么、该输出什么”，彻底解决 RNN 长文本记忆衰减问题。
#### 3. 优缺点分析
- 优点：完美解决 RNN 梯度消失问题，具备强大的长距离语义依赖捕捉能力，可同时适配长短文本分类任务；语义特征提取精准、模型鲁棒性强，是传统时序文本模型中综合效果最优的结构。
- 缺点：延续 RNN 串行逐词计算模式，无法并行运算，训练耗时远高于 TextCNN；网络结构复杂、参数更多，训练成本更高，收敛速度相对较慢。

| 项目 | 内容 |
| ---- | ---- |
| 核心功能 | 面向序列数据的循环网络，依靠门控机制保存时序信息，建模上下文长距离依赖。 |
| 输入形式 | 按照文本语序依次输入每个单词对应的词向量。 |
| 特征提取流程 | 通过遗忘门、输入门、输出门调控细胞状态，选择性保留、更新时序语义；持续迭代隐藏状态累积全文信息，最终输出时序特征用于分类。 |
| 计算特性 | 串行逐时刻计算，无法并行。 |
| 优势 | 重视词语顺序，能够捕捉长距离上下文关联，长短文本均可适配。 |
| 局限性 | 训练速度较慢，参数量更大，训练开销高于TextCNN。 |

##     
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
  
