import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class GraphAwareTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 图增强的Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, activation='gelu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.action_head = nn.Linear(d_model, 3)  # 3个动作
        
        # 初始化参数
        self.init_weights()
        self.d_model = d_model

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.action_head.bias.data.zero_()
        self.action_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, graph_mask=None, src_key_padding_mask=None):
        """
        输入:
        src: [seq_len, batch_size]
        graph_mask: [seq_len, seq_len] 图结构注意力掩码
        src_key_padding_mask: [batch_size, seq_len] 填充掩码
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 融合图注意力掩码
        if graph_mask is not None:
            attn_mask = graph_mask.to(src.device)
        else:
            attn_mask = None
        
        output = self.transformer_encoder(
            src, 
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 全局平均池化
        pooled = output.mean(dim=0)
        action_scores = self.action_head(pooled)
        return action_scores

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x