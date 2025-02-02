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
    criterion = nn.BCEWithLogitsLoss()
    
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)              # [B, global_seq_length]
            attention_mask = batch['attention_mask'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            candidate_param_ids = batch['candidate_param_ids'].to(device)  # [B, num_candidates, param_seq_length]
            
            # 获取每个样本的 ground truth 操作
            gt_actions = []
            if "raw_data" in batch:
                for sample in batch["raw_data"]:
                    gt_actions.append(sample.get("selected_op", {}).get("action", "UNK"))
            else:
                gt_actions = ["UNK"] * input_ids.size(0)
            
            gt_action_ids = []
            for action in gt_actions:
                token = f"[ACTION_{action}]"
                gt_action_ids.append(train_dataset.tokenizer.vocab.get(token, -1))
            gt_action_ids = torch.tensor(gt_action_ids, device=device)  # [B]
            
            # 候选参数中第 2 个 token 为操作类型
            candidate_types = candidate_param_ids[:, :, 1]  # [B, num_candidates]
            B, num_candidates = candidate_types.shape
            gt_labels = torch.zeros(B, num_candidates, device=device, dtype=torch.float)
            for i in range(B):
                gt_labels[i] = (candidate_types[i] == gt_action_ids[i]).float()
            
            optimizer.zero_grad()
            scores = model(
                input_ids.transpose(0, 1),  # 转换为 [seq_len, B]
                candidate_param_ids,
                graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(device),
                src_key_padding_mask=(attention_mask == 0)
            )
            loss = criterion(scores, gt_labels)
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