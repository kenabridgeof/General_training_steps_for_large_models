# Transformer 核心模块结构索引

---

## 1. Embedding

- **类名**：`Embedding`
- **构造函数**：`__init__(vocab_size, d_model)`
- **注意**：`forward` 输出需乘以 $\sqrt{d_\text{model}}$

---

## 2. PositionEncoding

- **类名**：`PositionEncoding`
- **构造函数**：`__init__(d_model, dropout_p, max_len=...)`

---

## 3. Attention 计算

- **函数定义**：`attention(query, key, value, mask, dropout_p)`

---

## 4. MultiHeadAttention

- **类名**：`MultiHeadAttention`
- **构造函数**：`__init__(head, d_model, dropout_p=...)`

---

## 5. FeedForward

- **类名**：`FeedForward`
- **构造函数**：`__init__(d_model, d_ff, dropout_p=...)`

---

## 6. LayerNorm

- **类名**：`LayerNorm`
- **构造函数**：`__init__(features, eps=1e-6)`

---

## 7. SubLayerConnection

- **类名**：`SubLayerConnection`
- **构造函数**：`__init__(size, dropout_p=...)`

---

## 8. EncoderLayer

- **类名**：`EncoderLayer`
- **构造函数**：`__init__(size, self_attention, feed_forward, dropout_p)`

---

## 9. Encoder

- **类名**：`Encoder`
- **构造函数**：`__init__(layer, N)`

---

## 10. DecoderLayer

- **类名**：`DecoderLayer`
- **构造函数**：`__init__(size, self_attention, src_attention, feed_forward, dropout_p)`

---

## 11. Decoder

- **类名**：`Decoder`
- **构造函数**：`__init__(layer, N)`

---

## 12. Generator

- **类名**：`Generator`
- **构造函数**：`__init__(d_model, vocab_size)`

---

