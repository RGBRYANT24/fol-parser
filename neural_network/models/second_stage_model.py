# models/second_stage_model.py

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from .common import PositionalEncoding

class GlobalEncoder(nn.Module):
    """
    全局状态编码器，与 first_stage_model 中的实现类似。
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
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src.transpose(0, 1)  # [B, seq_len, d_model]
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # [seq_len, B, d_model]
        output = self.transformer_encoder(src, mask=graph_mask, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)  # [B, seq_len, d_model]
        return output

class CandidateEncoder(nn.Module):
    """
    候选操作参数编码器。
    输入形状：[B, num_candidates, param_seq_length]
    输出形状：[B, num_candidates, d_model]
    """
    def __init__(self, vocab_size, d_model, max_param_seq_length, nhead=4, num_layers=2, dropout=0.1):
        super(CandidateEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_param_seq_length)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_model*4, dropout=dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.max_param_seq_length = max_param_seq_length
        self.d_model = d_model
    
    def forward(self, candidate_ids):
        # candidate_ids: [B, num_candidates, param_seq_length]
        B, num_candidates, seq_length = candidate_ids.shape
        candidates = candidate_ids.view(B * num_candidates, seq_length)
        x = self.embedding(candidates) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [seq_length, B*num_candidates, d_model]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # 全局平均池化 [B*num_candidates, d_model]
        candidate_repr = x.view(B, num_candidates, self.d_model)
        return candidate_repr

class CrossTransformer(nn.Module):
    """
    交叉 Transformer 模块，用于将候选参数表示（Query）和全局状态（Key/Value）进行交叉注意力融合。
    """
    def __init__(self, d_model, nhead=4, num_layers=1, dropout=0.1):
        super(CrossTransformer, self).__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_model*4, dropout=dropout, activation='gelu')
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
    
    def forward(self, candidate_repr, global_state):
        # candidate_repr: [B, num_candidates, d_model]
        # global_state: [B, seq_len, d_model]
        candidate_repr = candidate_repr.transpose(0, 1)  # [num_candidates, B, d_model]
        global_state = global_state.transpose(0, 1)      # [seq_len, B, d_model]
        refined = self.decoder(candidate_repr, global_state)  # [num_candidates, B, d_model]
        refined = refined.transpose(0, 1)  # [B, num_candidates, d_model]
        return refined

class SecondStageModel(nn.Module):
    """
    第二阶段模型：
      1. 使用 GlobalEncoder 编码状态（SLI树+图数据）；
      2. 使用 CandidateEncoder 编码候选操作参数；
      3. 通过 CrossTransformer 模块将候选表示和全局状态进行交互；
      4. 根据候选参数里的操作类型（假设第 2 个 token 为 [ACTION_xxx]）分别进行分支路由，
         并与全局状态摘要拼接，最后经过 MLP 输出候选评分。
    """
    def __init__(self, global_encoder, candidate_encoder, cross_transformer,
                 d_model, branch_hidden_dim, fusion_hidden_dim, tokenizer):
        super(SecondStageModel, self).__init__()
        self.global_encoder = global_encoder
        self.candidate_encoder = candidate_encoder
        self.cross_transformer = cross_transformer
        
        # 不同操作类型采用不同评分头（分支）
        self.ext_head = nn.Linear(d_model, branch_hidden_dim)
        self.fact_head = nn.Linear(d_model, branch_hidden_dim)
        self.ances_head = nn.Linear(d_model, branch_hidden_dim)
        
        # 融合层，将候选分支特征与全局状态摘要拼接，输出单个候选评分
        self.fusion_mlp = nn.Sequential(
            nn.Linear(branch_hidden_dim + d_model, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 1)
        )
        
        self.tokenizer = tokenizer
        # 获取操作类型 token id，假设候选参数序列中第 2 个 token 为操作标识
        self.ext_token_id = self.tokenizer.vocab.get("[ACTION_Extension]", -1)
        self.fact_token_id = self.tokenizer.vocab.get("[ACTION_Factoring]", -1)
        self.ances_token_id = self.tokenizer.vocab.get("[ACTION_Ancestry]", -1)
    
    def forward(self, global_input, candidate_param_ids, graph_mask=None, src_key_padding_mask=None):
        # global_input: [seq_len, B]
        global_state_seq = self.global_encoder(
            global_input, graph_mask, src_key_padding_mask
        )  # [B, seq_len, d_model]
        candidate_repr = self.candidate_encoder(candidate_param_ids)  # [B, num_candidates, d_model]
        fused_candidates = self.cross_transformer(candidate_repr, global_state_seq)  # [B, num_candidates, d_model]
        
        # 根据候选参数序列中第 2 个 token 判断操作类型
        candidate_types = candidate_param_ids[:, :, 1]  # [B, num_candidates]
        B, num_candidates, _ = fused_candidates.shape
        branch_features = torch.zeros(B, num_candidates, self.ext_head.out_features,
                                      device=fused_candidates.device)
        ext_mask = (candidate_types == self.ext_token_id)
        fact_mask = (candidate_types == self.fact_token_id)
        ances_mask = (candidate_types == self.ances_token_id)
        if ext_mask.sum() > 0:
            branch_features[ext_mask] = self.ext_head(fused_candidates[ext_mask])
        if fact_mask.sum() > 0:
            branch_features[fact_mask] = self.fact_head(fused_candidates[fact_mask])
        if ances_mask.sum() > 0:
            branch_features[ances_mask] = self.ances_head(fused_candidates[ances_mask])
        
        # 采用全局平均池化得到状态摘要
        global_summary = global_state_seq.mean(dim=1)  # [B, d_model]
        global_summary_expanded = global_summary.unsqueeze(1).expand(-1, num_candidates, -1)
        fused_features = torch.cat([branch_features, global_summary_expanded], dim=-1)
        scores = self.fusion_mlp(fused_features).squeeze(-1)  # [B, num_candidates]
        return scores