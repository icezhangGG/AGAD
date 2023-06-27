import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

# 加性模型
class attention1(nn.Module):
    def __init__(self, q_size, k_size, v_size, seq_len):
        # q、k、v的维度，seq_len每句话中词的数量
        super(attention1, self).__init__()
        self.linear_v = nn.Linear(v_size, seq_len)
        self.linear_W = nn.Linear(k_size, k_size)
        self.linear_U = nn.Linear(q_size, q_size)
        self.tanh = nn.Tanh()

    def forward(self, query, key, value, dropout=None):
        key = self.linear_W(key)
        query = self.linear_U(query)
        k_q = self.tanh(query + key)
        alpha = self.linear_v(k_q)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(0)
        value=value.unsqueeze(0)
        out = th.bmm(alpha, value)
        alpha = alpha.squeeze(0)
        out = out.squeeze(0)
        return out, alpha
# 点积模型
class attention2(nn.Module):
    def __init__(self):
        super(attention2, self).__init__()
    def forward(self, query, key, value, dropout=None):
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        alpha = th.bmm(query, key.transpose(-1, -2))
        alpha = F.softmax(alpha, dim=-1)
        value = value.unsqueeze(0)
        out = th.bmm(alpha, value)
        alpha = alpha.squeeze(0)
        value = value.squeeze(0)
        return out, alpha

# 缩放点积模型
class attention3(nn.Module):
    def __init__(self):
        # q、k、v的维度，seq_len每句话中词的数量
        super(attention3, self).__init__()
    def forward(self, query, key, value, dropout=None):
        d = key.size(-1)
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        alpha = th.bmm(query, key.transpose(-1, -2)) / math.sqrt(d)
        alpha = F.softmax(alpha, dim=-1)
        # print("===========",alpha.shape)
        # print("===========",value.shape)
        out = th.bmm(alpha, value)
        query = query.squeeze(1)
        key = key.squeeze(1)
        value = value.squeeze(1)
        out = out.squeeze(1)
        alpha = alpha.squeeze(1)
        return out, alpha
#多头注意力
class MultiheadAttention(nn.Module):
    def __init__(self, head, embedding_size, dropout=0.2):
        super(MultiheadAttention, self).__init__()
        assert embedding_size % head == 0 # 得整分
        self.head = head
        self.W_K = nn.Linear(embedding_size, embedding_size)
        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.W_V = nn.Linear(embedding_size, embedding_size)
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.d_k = embedding_size // head
        self.attention = attention3()
    def forward(self, query, key, value):
        batch_size = query.size(0)
        # 转换成多头，一次矩阵乘法即可完成
        query = self.W_Q(query).view(batch_size, self.head, -1, self.d_k)
        key = self.W_K(key).view(batch_size, self.head, -1, self.d_k)
        value = self.W_V(value).view(batch_size, self.head, -1, self.d_k)
        out, alpha = self.attention(query, key, value, self.dropout)
        out = out.view(batch_size, -1, self.d_k * self.head)
        out = self.fc(out)
        return out, alpha

#测试失败的注意力层
class AttentionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttentionLayer, self).__init__()
        self.input_proj = nn.Linear(input_size, output_size, bias=False)
        self.output_proj = nn.Linear(output_size, output_size, bias=False)

    def forward(self, inputs, outputs):
        inputs = self.input_proj(inputs).unsqueeze(inputs,0)  # (batch_size, n, input_size) -> (batch_size, n, output_size)
        outputs = self.output_proj(outputs).t.unsqueeze(outputs,0) # (batch_size, m, output_size) -> (batch_size, m, output_size)

        scores = th.bmm(inputs, outputs.transpose(1,2))  # (batch_size, n, output_size) * (batch_size, output_size, m) -> (batch_size, n, m)
        weights = F.softmax(scores, dim=1)  # (batch_size, n, m)
        context = th.bmm(weights.transpose(1, 2),inputs)  # (batch_size, m, n) * (batch_size, n, output_size) -> (batch_size, m, output_size)
        return context

