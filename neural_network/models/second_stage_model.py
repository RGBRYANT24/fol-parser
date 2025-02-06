import math
import torch
import torch.nn as nn
from neural_network.models.common import PositionalEncoding

class SecondStageModel(nn.Module):
    """
    改进后的第二阶段模型：
      - 使用共享的 Transformer 对全局状态和候选操作参数进行编码
      - 全局输入经过 Transformer 编码后，平均池化得到全局状态摘要
      - 候选参数序列同样经过 Transformer 编码后得到候选表示
      - 对候选表示和全局摘要分别进行投影（投影到 branch_hidden_dim），再进行拼接与融合降维（fusion_hidden_dim）
      - 根据候选参数中第 2 个 token 表示的候选操作类型，通过不同的 head 输出候选得分
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                 branch_hidden_dim=256, fusion_hidden_dim=128, tokenizer=None):
        super(SecondStageModel, self).__init__()
        self.tokenizer = tokenizer
        self.d_model = d_model
        
        # 共享嵌入层与位置编码模块
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分别对候选和全局表示进行投影
        self.candidate_proj = nn.Linear(d_model, branch_hidden_dim)
        self.global_proj = nn.Linear(d_model, branch_hidden_dim)
        
        # 融合层：将拼接（[候选;全局]，维度为 2*branch_hidden_dim）后的特征降维到 fusion_hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * branch_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim)
        )
        
        # 分类型候选得分 head（输出标量得分）
        self.ext_head = nn.Linear(fusion_hidden_dim, 1)
        self.fact_head = nn.Linear(fusion_hidden_dim, 1)
        self.ances_head = nn.Linear(fusion_hidden_dim, 1)
        
        # 利用 tokenizer 获取候选类型特殊 token 对应的 token id（假设候选参数中的第 2 个 token 表示操作类型）
        if self.tokenizer:
            self.ext_token_id = self.tokenizer.vocab.get("[ACTION_Extension]", -1)
            self.fact_token_id = self.tokenizer.vocab.get("[ACTION_Factoring]", -1)
            self.ances_token_id = self.tokenizer.vocab.get("[ACTION_Ancestry]", -1)
        else:
            self.ext_token_id = -1
            self.fact_token_id = -1
            self.ances_token_id = -1

    def encode(self, token_ids, mask=None, src_key_padding_mask=None):
        """
        对输入 token 序列进行编码
        参数:
          token_ids: [seq_len, batch_size]
          mask: 可选的后续遮罩
          src_key_padding_mask: 可选的键填充遮罩
        返回:
          编码后的张量，形状为 [seq_len, batch_size, d_model]
        """
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        # 由于 PositionalEncoding 期望的形状为 [batch, seq_len, d_model]，因此先转置
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        encoded = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return encoded
    
    def forward(self, global_input, candidate_param_ids, graph_mask=None, src_key_padding_mask=None):
        """
        参数:
          global_input: [seq_len, B] —— 全局输入序列（例如 SLI 树和图数据）
          candidate_param_ids: [B, num_candidates, param_seq_length] —— 候选操作参数 token 序列
          graph_mask: Transformer 后续遮罩，用于全局输入编码
          src_key_padding_mask: 填充遮罩，用于全局输入编码
        返回:
          scores: [B, num_candidates] —— 每个候选操作参数对应的连续得分
        """
        device = global_input.device
        
        # 1. 全局输入编码（传入 mask 与填充 mask）
        global_encoded = self.encode(global_input, mask=graph_mask, src_key_padding_mask=src_key_padding_mask)
        global_encoded = global_encoded.transpose(0, 1)  # [B, seq_len, d_model]
        global_summary = global_encoded.mean(dim=1)      # [B, d_model]
        
        # 2. 候选参数编码（候选序列通常固定，无需额外 mask）
        B, num_candidates, cand_seq_len = candidate_param_ids.size()
        candidate_flat = candidate_param_ids.view(B * num_candidates, cand_seq_len)  # [B*num_candidates, cand_seq_len]
        candidate_flat = candidate_flat.transpose(0, 1)  # [cand_seq_len, B*num_candidates]
        candidate_encoded = self.encode(candidate_flat)  # [cand_seq_len, B*num_candidates, d_model]
        candidate_encoded = candidate_encoded.transpose(0, 1)  # [B*num_candidates, cand_seq_len, d_model]
        candidate_repr = candidate_encoded.mean(dim=1)  # [B*num_candidates, d_model]
        candidate_repr = candidate_repr.view(B, num_candidates, self.d_model)  # [B, num_candidates, d_model]
        
        # 3. 分别对候选表示与全局摘要进行投影
        candidate_proj = self.candidate_proj(candidate_repr)            # [B, num_candidates, branch_hidden_dim]
        global_proj = self.global_proj(global_summary)                  # [B, branch_hidden_dim]
        global_proj_expanded = global_proj.unsqueeze(1).expand(-1, num_candidates, -1)  # [B, num_candidates, branch_hidden_dim]
        
        # 4. 融合：拼接投影后的候选与全局特征，并通过融合网络降维
        fused_features = torch.cat([candidate_proj, global_proj_expanded], dim=-1)  # [B, num_candidates, 2*branch_hidden_dim]
        fused_features = self.fusion_layer(fused_features)  # [B, num_candidates, fusion_hidden_dim]
        
        # 5. 根据候选参数中第 2 个 token（代表候选类型）选择不同 head 进行得分预测
        candidate_types = candidate_param_ids[:, :, 1]  # [B, num_candidates]
        scores = torch.zeros(B, num_candidates, device=device)
        
        if self.ext_token_id != -1:
            ext_mask = (candidate_types == self.ext_token_id)
            if ext_mask.any():
                ext_scores = self.ext_head(fused_features[ext_mask]).squeeze(-1)
                scores[ext_mask] = ext_scores
        if self.fact_token_id != -1:
            fact_mask = (candidate_types == self.fact_token_id)
            if fact_mask.any():
                fact_scores = self.fact_head(fused_features[fact_mask]).squeeze(-1)
                scores[fact_mask] = fact_scores
        if self.ances_token_id != -1:
            ances_mask = (candidate_types == self.ances_token_id)
            if ances_mask.any():
                ances_scores = self.ances_head(fused_features[ances_mask]).squeeze(-1)
                scores[ances_mask] = ances_scores
        
        return scores