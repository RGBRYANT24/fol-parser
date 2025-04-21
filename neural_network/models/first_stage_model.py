# models/first_stage_model.py

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .common import PositionalEncoding

class GlobalEncoder(nn.Module):
    """
    全局状态编码器，对 SLI 树和图数据进行编码。
    输入应为 [seq_len, B]（Transformer 要求这种维度），输出为 [B, seq_len, d_model]。
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super(GlobalEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, graph_mask=None, src_key_padding_mask=None):
        # src: [seq_len, B]
        src = self.embedding(src) * math.sqrt(self.d_model)  # [seq_len, B, d_model]
        src = src.transpose(0, 1)  # 转成 [B, seq_len, d_model]
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # 转回 [seq_len, B, d_model]
        output = self.transformer_encoder(
            src,
            mask=graph_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        output = output.transpose(0, 1)  # [B, seq_len, d_model]
        return output

class FirstStageModel(nn.Module):
    """
    第一阶段模型：
      1. 使用 GlobalEncoder 对状态（SLI 树 + 图数据）进行编码；
      2. 采用全局平均池化得到状态摘要；
      3. 通过全连接层预测三个操作（Extension, Factoring, Ancestry）的分数。
    """
    def __init__(self, global_encoder, d_model, num_actions=3):
        super(FirstStageModel, self).__init__()
        self.global_encoder = global_encoder
        self.action_head = nn.Linear(d_model, num_actions)
    
    def forward(self, src, graph_mask=None, src_key_padding_mask=None):
        # src: [seq_len, B]
        global_state_seq = self.global_encoder(src, graph_mask, src_key_padding_mask)  # [B, seq_len, d_model]
        global_state = global_state_seq.mean(dim=1)  # 全局平均池化得到 [B, d_model]
        action_scores = self.action_head(global_state)  # [B, num_actions]
        # 先用sigmoid将值映射到(0,1)区间，再用clamp确保包含边界值0和1
        action_scores = torch.clamp(torch.sigmoid(action_scores), 0.0, 1.0)
        return action_scores