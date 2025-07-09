# Transformer 核心模块结构索引

# 一.Input
---

## 1. Embedding

- **类名**：`Embedding`
- **构造函数**：`__init__(vocab_size, d_model)`
- **前向传播函数**: `forward(self, x)`
- **注意**：`forward` 输出需乘以 $\sqrt{d_\text{model}}$

---

## 2. PositionEncoding

- **类名**：`PositionEncoding`
- **构造函数**：`__init__(d_model, dropout_p, max_len=...)`
- **前向传播函数**: `forward(self, x)`
- **公式**:<img src="https://github.com/user-attachments/assets/92d7e48a-14f9-48ab-a8c2-7bd099b3b7ce" alt="位置编码计算公式" width="320"/>

- **步骤**:
```
1.初始化参数
2.构造位置编码矩阵
3.填充偶数&奇数维度
4.注册为缓冲区(register_buffer)
5.前向融合语义向量和位置编码
```

---

# 二.Encoder

## 3. Attention 计算

- **函数定义**：`Attention(query, key, value, mask, dropout_p)`
- **公式**: <img src="https://github.com/user-attachments/assets/546f5a83-b56a-4dbd-81c7-edfa95b63168" alt="注意力机制公式" width="320"/>
- **步骤**:
```
1.计算维度标量 dₖ
2.点积并缩放
3.应用掩码（可选）
4.Softmax 归一化
5.dropout(可选)
6.加权求和并返回
```

---
## x.工具函数clones
```
return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```
---

## 4. MultiHeadAttention

- **类名**：`MultiHeadAttention`
- **构造函数**：`__init__(head, d_model, dropout_p=...)`
- **前向传播函数**: `forward(self, query, key, value, mask=None)`
- **步骤**:
```
init
1. 断言整除判断 assert
2. 多头拆分维度
3. 线性层(q, k, v, 融合多头后的输出)
4. Dropout

forward
1. 掩码升维
2. 获取 batch_size
3. 线性变换 + 切分多头
4. 调用单头注意力函数计算
5. 拼回多头输出
6. return最后一层线性映射
```

---

## 5. FeedForward

- **类名**：`FeedForward`
- **构造函数**：`__init__(d_model, d_ff, dropout_p=...)`
- **前向传播函数**: `forward(self, x)`
- **公式**:<img src="https://github.com/user-attachments/assets/e256c18e-d2a8-43b6-b968-e7f0679c7b19" alt="前馈全连接层公式" width="320"/>

- **步骤**:
```
1. 初始化超参数
2. 定义两层线性变换
3. 前向计算流程 linear1 -> Rule -> dropout -> linear2
```

---

## 6. LayerNorm

- **类名**：`LayerNorm`
- **构造函数**：`__init__(features, eps=1e-6)`
- **前向传播函数**: `forward(self, x)`
- **步骤**:
```
1. 初始化超参数
2. 计算均值与标准差
3. 归一化并仿射变换
```

---

## 7. SubLayerConnection

- **类名**：`SubLayerConnection`
- **构造函数**：`__init__(size, dropout_p=...)`
- **前向传播函数**: `forward(self, x, sublayer)`
- **步骤**:
```
1. 初始化超参数 & LayerNorm模块
2. 前向计算流程 
```

---

## 8. EncoderLayer

- **类名**：`EncoderLayer`
- **构造函数**：`__init__(size, self_attention, feed_forward, dropout_p)`
- **前向传播函数**: `forward(self, x, mask)`
- **步骤**:
```
1. 初始化子模块 clones2层SubLayerConnection
2. 前向计算流程 sub_layer[N] 
```

---

## 9. Encoder

- **类名**：`Encoder`
- **构造函数**：`__init__(layer, N)`
- **前向传播函数**: `forward(self, x, mask)`
- **步骤**:
```
1. 初始化模块 clones N & 调用LayerNorm
2. 前向计算流程
```

---

## 10. DecoderLayer

- **类名**：`DecoderLayer`
- **构造函数**：`__init__(size, self_attention, src_attention, feed_forward, dropout_p)`
- **前向传播函数**: `forward(self, y, encoder_output, source_mask, target_mask)`
- **步骤**:
```
1. 初始化子模块 clones SubLayerConnection
2. 前向计算流程 N=3
```

---

## 11. Decoder

- **类名**：`Decoder`
- **构造函数**：`__init__(layer, N)`
- **前向传播函数**: `forward(self, y, memory, source_mask, target_mask)`
- **步骤**:
```
1. 初始化解码器层列表 clones layer N=6
2. 定义最终规范化 LayerNorm
3. 前向计算流程
```

---

## 12. Generator

- **类名**：`Generator`
- **构造函数**：`__init__(d_model, vocab_size)`

---

