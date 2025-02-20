#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# read_data 模块
from neural_network.data.read_data import GraphSLIDataset, collate_fn

# 从第一阶段模型模块中导入
from neural_network.models.first_stage_model import GlobalEncoder, FirstStageModel

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train_phase1():
    config = {
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 20,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'save_path': 'first_stage_model.pth'
    }
    
    # 加载数据集（read_data 模块已实现）
    train_dataset = GraphSLIDataset(
        sli_file="data/training_data.json",
        graph_file="data/k3_graph.json",
        max_seq_length=512
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
    
    model = FirstStageModel(global_encoder, d_model=config['d_model'], num_actions=3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()  # 可根据标签形式改为 CrossEntropyLoss
    
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)         # [B, seq_len]
            attention_mask = batch['attention_mask'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            labels = batch['labels'].to(device)                 # [B, 3]
            
            optimizer.zero_grad()
            outputs = model(
                input_ids.transpose(0, 1),  # 转换为 [seq_len, B]
                graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(device),
                src_key_padding_mask=(attention_mask == 0)
            )
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Phase1 Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['save_path'])
        print(f"Phase1 Epoch {epoch} | Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_phase1()