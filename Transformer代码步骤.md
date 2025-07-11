# Transformer 核心模块结构索引
```
# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
```

# 一.Input
---

## 1. Embedding(词嵌入层)

- **类名**：`Embedding`
- **构造函数**：`__init__(vocab_size, d_model)`
- **前向传播函数**: `forward(self, x)`
- **注意**：`forward` 输出需乘以 $\sqrt{d_\text{model}}$

---

## 2. PositionEncoding(位置编码)
作用：为 Transformer 提供位置编码（Positional Encoding），让模型知道序列中每个 token 的位置信息。
- **类名**：`PositionEncoding`
- **构造函数**：`__init__(d_model, dropout_p, max_len=...)`
- **前向传播函数**: `forward(self, x)`
- **公式**:<img src="https://github.com/user-attachments/assets/f6ba8457-bbf8-4314-8e12-91e2b6d58aab" alt="位置编码计算公式" width="500"/>
- **步骤**:
```
1.初始化参数
2.构造位置编码矩阵
  - 创建全零张量pe [max-len, d_model] -> [60, 512]
  - 构造位置索引索引position再升维 [max_len, 1] ->[60, 1] 每行是一个位置 pos，从 0 到 59。
  - 计算频率缩放向量(就是公式中的被除数计算)  ->[256]
3.填充偶数&奇数维度
  - 计算角度矩阵pos_vec=position * div_term     [max_len, d_model // 2] ->[60, 256]
  - 拆分pe sin/cos
4.注册为缓冲区(register_buffer)
  - pe升维,便于后续广播
  - 使用register_buffer将其加入模型，但不作为可训练参数
5.前向融合语义向量和位置编码
  - 将输入 x（形状 [batch, seq_len, d_model]）与对应的前 seq_len 位位置编码相加，再做随机失活
    只取前 𝐿个位置的编码，避免将超过实际句长的“多余”位置编码加进来。
```

---

# 二.Encoder

## x.mask(掩码张量)
- **函数定义**: `subsequent_mask(size)`
```
return 1 - torch.triu(torch.ones(1, size, size, dtype=torch.int), diagonal=1)
```

## 3. Attention(注意力计算)

- **函数定义**：`attention(query, key, value, mask, dropout_p)`
- **公式**: <img src="https://github.com/user-attachments/assets/e43b185b-bcfb-4635-a279-ae358e2647a5" alt="注意力机制公式" width="320"/>
- **步骤**:
```
1.计算维度标量 dₖ
  - 取最后一维长度，后面用来做缩放因子。
2.点积并缩放
  - 将 Query 和 Key 做矩阵乘法，得到相似度分数矩阵。除以𝑑𝑘防止数值过大导致 softmax 梯度消失。
3.应用掩码（可选）
  - if  ?  is not None:
4.Softmax 归一化
  - softmax的代码记得dim=-1,得到注意力权重
5.dropout(可选)
6.加权求和并返回
  - 得到中间上下文向量C
```

---
## x.工具函数clones
```
module, N
return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```
---

## 4. MultiHeadAttention(多头注意力机制层)

- **类名**：`MultiHeadAttention`
- **构造函数**：`__init__(head, d_model, dropout_p=...)`
- **前向传播函数**: `forward(self, query, key, value, mask=None)`
- **公式**:<img src="https://github.com/user-attachments/assets/ea220db3-595a-4ff0-8181-64bc12180c34" alt="多头注意力公式" width="400"/>
- **步骤**:
```
init
1. 断言整除判断 assert
2. 多头拆分维度
3. 线性层(q, k, v, 融合多头后的输出)
  - clones()
4. Dropout

forward
1. 掩码升维
2. 获取 batch_size
3. 线性变换 + 切分多头
    #    [batch, seq_len, embed_dim] 
    # → 线性 → [batch, seq_len, embed_dim]
    # → view → [batch, seq_len, head, dₖ]
    # → transpose → [batch, head, seq_len, dₖ]
4. 调用单头注意力函数计算
    - 返回 shape: [batch, head, seq_len, dₖ]
5. 拼回多头输出
    -    [batch, head, seq_len, dₖ] 
    - → transpose → [batch, seq_len, head, dₖ]
    - → contiguous.view → [batch, seq_len, embed_dim]
6. return最后一层线性映射
```

---

## 5. FeedForward(前馈全连接层)

- **类名**：`FeedForward`
- **构造函数**：`__init__(d_model, d_ff, dropout_p=...)`
- **前向传播函数**: `forward(self, x)`
- **公式**:<img src="https://github.com/user-attachments/assets/a9aa7e20-4063-4268-a502-ca331506bd8d" alt="前馈全连接层公式" width="320"/>
- **步骤**:
```
1. 初始化超参数
2. 定义两层线性变换
3. 前向计算流程 linear1 -> Rule -> dropout -> linear2
```

---

## 6. LayerNorm(规范化层)

- **类名**：`LayerNorm`
- **构造函数**：`__init__(features, eps=1e-6)`
- **前向传播函数**: `forward(self, x)`
- **公式**: <img src="https://github.com/user-attachments/assets/b80ff74f-b5d6-4fdd-8d74-138f103bac04" alt="层归一化公式" width="280"/>
- **步骤**:
```
1. 初始化超参数
  - 初始化为 全 1 的缩放向量 γ 和 全 0 的偏移向量 β。 eps（ε） nn.Parameter()
2. 计算均值与标准差
  - torch.mean/std   keepdim=Ture
3. 标准化：减均值、除标准差（加 eps 防止除 0）
  - 再用 γ（self.a）缩放、加 β（self.b）平移
```

---

## 7. SubLayerConnection(子层连接结构)

- **类名**：`SubLayerConnection`
- **构造函数**：`__init__(size, dropout_p=...)`
- **前向传播函数**: `forward(self, x, sublayer)`
- **公式**: <img src="https://github.com/user-attachments/assets/cf3910bb-59d1-4c6d-b61e-b257fc04dbb3" alt="残差连接和归一化" width="400"/>
- **步骤**:
```
1. 初始化超参数 & LayerNorm模块
2. 前向计算流程
    # 1）归一化：先对输入 x 做 LayerNorm
    # 2）子层计算：将归一化结果输入到子层（如 Attention 或 FeedForward）
    # 3）Dropout：对子层输出做随机失活
    # 4）残差连接：将原始 x 与处理后结果相加
```

---

## 8. EncoderLayer(编码器层)

- **类名**：`EncoderLayer`
- **构造函数**：`__init__(size, self_attention, feed_forward, dropout_p)`
- **前向传播函数**: `forward(self, x, mask)`
- **步骤**:
```
1. 初始化子模块
  -  # 克隆两个 SubLayerConnection（残差+LayerNorm）实例
2. 前向计算流程 sub_layer[N]
  - # 1）第一子层：多头自注意力
  - # 2）第二子层：前馈网络
```

---

## 9. Encoder(编码器)

- **类名**：`Encoder`
- **构造函数**：`__init__(layer, N)`
- **前向传播函数**: `forward(self, x, mask)`
- **步骤**:
```
1. 初始化模块 clones N & 调用LayerNorm
  - layer：已构造的 EncoderLayer 实例（包含自注意力＋前馈＋残差＋规范化）。
  - 用 clones 复制 layer N 份，生成 ModuleList。
  - 构造最终用的层归一化 LayerNorm，维度和单层保持一致。
2. 前向计算流程
  - 循环堆叠：依次将输入 x 和同一 mask 喂入每一个子 EncoderLayer，输出作为下一层输入。
  - 归一化：所有层处理完毕后，对最终输出做一次 LayerNorm，增强信息稳定性。
```

---

# 三.Decoder

---

## 10. DecoderLayer(解码器层)

- **类名**：`DecoderLayer`
- **构造函数**：`__init__(size, self_attention, src_attention, feed_forward, dropout_p)`
- **前向传播函数**: `forward(self, y, encoder_output, source_mask, target_mask)`
- **步骤**:
```
1. 初始化子模块
  - clones SubLayerConnection
  - 目标序列自注意力层
  - 编码器-解码器交互注意力层
2. 前向计算流程 N=3个子层
  - sub_layers[?]
```

---

## 11. Decoder(解码器)

- **类名**：`Decoder`
- **构造函数**：`__init__(layer, N)`
- **前向传播函数**: `forward(self, y, memory, source_mask, target_mask)`
- **步骤**:
```
1. 初始化解码器层列表 clones layer N=6
2. 定义最终规范化 LayerNorm
3. 前向计算流程
  -  1）循环堆叠每一层解码器
  -  2）最终 LayerNorm 归一化
```

---

## 12. Generator(输出)

- **类名**：`Generator`
- **构造函数**：`__init__(d_model, vocab_size)`
- **前向传播函数**: `forward(self, x)`
- **步骤**:
```
1. 初始化超参数
2. 定义线性层
3. 前向计算
  - 记得dim=-1,log_softmax
```
---

# 四. 整合Transformer全流程

- **类名**：`EncoderToDecoder`
- **构造函数**：`___init__(self, encoder, decoder, source_embed, target_embed, generator)`
- **前向传播函数**: `forward(self, source_x, target_y, source_mask1, source_mask2, target_mask)`
- **步骤**:
```
1. 初始化子模块
2. 前向计算流程
        #  source_x:代表编码器的输入：[batch_size, seq_len]-->[2, 4]
        #  target_y:代表解码器的输入：[batch_size, seq_len]-->[2, 6]
        #  source_mask1:代表编码器部分的padding mask：[head, source_seq_len, source_seq_len]-->[8, 4, 4]
        #  source_mask2:代表解码器（第二子层）部分的padding mask：[head, target_seq_len, source_seq_len]-->[8, 6, 4]
        #  target_mask:代表解码器部分的sentence mask：[head, target_seq_len, target_seq_len]-->[8, 6, 6]
```

---
---
