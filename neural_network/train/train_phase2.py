#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 假设你已有 read_data 模块
from neural_network.data.read_data import GraphSLIDataset, collate_fn

# 从第二阶段模型模块中导入
from neural_network.models.second_stage_model import GlobalEncoder, CandidateEncoder, CrossTransformer, SecondStageModel

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train_phase2():
    config = {
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 15,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'max_param_seq_length': 30,
        'branch_hidden_dim': 256,
        'fusion_hidden_dim': 128,
        'save_path': 'second_stage_model.pth'
    }
    
    train_dataset = GraphSLIDataset(
        sli_file="data/training_data.json",
        graph_file="data/k3_graph.json",
        max_seq_length=512,
        max_param_seq_length=config['max_param_seq_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, train_dataset.tokenizer)
    )
    
    global_encoder = GlobalEncoder(
        vocab_size=len(train_dataset.tokenizer.vocab),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    )
    
    candidate_encoder = CandidateEncoder(
        vocab_size=len(train_dataset.tokenizer.vocab),
        d_model=config['d_model'],
        max_param_seq_length=config['max_param_seq_length']
    )
    
    cross_transformer = CrossTransformer(d_model=config['d_model'], nhead=4, num_layers=1)
    
    model = SecondStageModel(
        global_encoder,
        candidate_encoder,
        cross_transformer,
        d_model=config['d_model'],
        branch_hidden_dim=config['branch_hidden_dim'],
        fusion_hidden_dim=config['fusion_hidden_dim'],
        tokenizer=train_dataset.tokenizer
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    # 损失函数改为 MSELoss 用于回归
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)              # [B, global_seq_length]
            attention_mask = batch['attention_mask'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            candidate_param_ids = batch['candidate_param_ids'].to(device)  # [B, num_candidates, param_seq_length]
            # 使用连续回归标签
            candidate_q_values = batch['candidate_q_values'].to(device)      # [B, num_candidates]
            
            optimizer.zero_grad()
            scores = model(
                input_ids.transpose(0, 1),  # 转换为 [seq_len, B]
                candidate_param_ids,
                graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(device),
                src_key_padding_mask=(attention_mask == 0)
            )  # scores 的形状为 [B, num_candidates]，连续数值输出
            
            loss = criterion(scores, candidate_q_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Phase2 Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['save_path'])
        print(f"Phase2 Epoch {epoch} | Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_phase2()