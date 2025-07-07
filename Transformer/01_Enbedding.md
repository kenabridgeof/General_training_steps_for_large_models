### 

```python
# DemoEmbedding 继承自 nn.Module，是一个“Token→向量”映射模块，通常用于 NLP 模型的词嵌入（word embedding）层。

"""
构造函数 __init__

1. vocab
    词表大小（token 数量），例如如果有 50,000 个不同单词/子词，就传入 vocab=50000。
2. embed_dim
    要把每个 token 映射到的向量维度（比如 512、768 等）。
3. self.embed = nn.Embedding(vocab, embed_dim)
    PyTorch 内置的查表式嵌入层：
        它内部维护一个形状为 [vocab, embed_dim] 的可训参数矩阵。
        当调用 self.embed(x) 时，x 是一个整型张量（token 索引），层就会把每个索引映射到对应行（向量），输出一个浮点向量。
"""
"""
前向函数 forward
目的: 
    将“原始的”嵌入向量的数值尺度放大到和位置编码（positional encoding）相当的量级 。
为什么要这样:
    如果直接把未经放缩的 embedding（它们通常是均值为 0、方差小于 1 的随机向量）加到位置编码上，位置编码反而会主导模型学习。
乘以 √𝑑ₖ 后，这两个加数方差差不多，模型能更好地同时利用“语义信息”（token embedding）和“位置信息”（positional encoding）。
"""

class DemoEmbedding(nn.Module):
    def __init__(self, vocab, embed_dim):
        super().__init__()
        # vocab
        self.vocab = vocab
        self.d_model = embed_dim

        self.embed = nn.Embedding(self.vocab, self.d_model)
        

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
```



### 
