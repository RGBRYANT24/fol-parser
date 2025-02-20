# models/common.py

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    通用位置编码器，将位置信息加入输入特征中。
    输入 x 的形状为 [B, seq_len, d_model]。
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用 cos
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [B, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]