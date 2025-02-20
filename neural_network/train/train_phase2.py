#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 假设你已有数据读取模块
from neural_network.data.read_data import GraphSLIDataset, collate_fn

# 导入改进后的第二阶段模型
from neural_network.models.second_stage_model import SecondStageModel

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf'))\
                .masked_fill(mask == 1, float(0.0))
    return mask

def train_phase2():
    config = {
        'batch_size': 2,
        'lr': 1e-4,
        'epochs': 15,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 6,
        'max_param_seq_length': 30,
        'branch_hidden_dim': 256,
        'fusion_hidden_dim': 128,
        'save_path': 'second_stage_model.pth'
    }
    
    # 构建数据集，注意数据文件路径及 tokenizer 的初始化由 GraphSLIDataset 内部实现
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
    
    # 实例化模型：注意这里采用了改进后的融合分支结构
    model = SecondStageModel(
        vocab_size=len(train_dataset.tokenizer.vocab),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        branch_hidden_dim=config['branch_hidden_dim'],
        fusion_hidden_dim=config['fusion_hidden_dim'],
        tokenizer=train_dataset.tokenizer
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    # 候选参数得分作为连续回归输出，这里使用 MSELoss
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # 全局输入 [B, global_seq_length]
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            # 候选操作参数 [B, num_candidates, param_seq_length]
            candidate_param_ids = batch['candidate_param_ids'].to(device)
            # 连续回归标签 [B, num_candidates]
            candidate_q_values = batch['candidate_q_values'].to(device)
            
            optimizer.zero_grad()
            scores = model(
                input_ids.transpose(0, 1),  # 转换为 [seq_len, B]
                candidate_param_ids,
                graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(device),
                src_key_padding_mask=(attention_mask == 0)
            )  # scores 的形状为 [B, num_candidates]
            
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