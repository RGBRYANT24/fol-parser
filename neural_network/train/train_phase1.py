#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os
import glob

# read_data 模块
from data.read_data import GraphSLIDataset, collate_fn

# 从第一阶段模型模块中导入
from models.first_stage_model import GlobalEncoder, FirstStageModel

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train_phase1():
    config = {
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 1,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'save_path': 'first_stage_model.pth',
        'data_dir': '../data',  # 指定数据目录
        'data_pattern': 'training_data_*.json'  # 指定数据文件匹配模式
    }
    
    # 寻找所有匹配的数据文件
    data_files = glob.glob(os.path.join(config['data_dir'], config['data_pattern']))
    print(f"Found {len(data_files)} data files: {data_files}")
    
    # 为每个数据文件创建一个数据集
    datasets = []
    for file_path in data_files:
        try:
            dataset = GraphSLIDataset(unified_file=file_path, max_seq_length=512)
            print(f"Loaded dataset from {file_path} with {len(dataset)} samples")
            datasets.append(dataset)
        except Exception as e:
            print(f"Error loading dataset from {file_path}: {e}")
    
    if not datasets:
        raise ValueError("No valid datasets found!")
    
    # 如果只有一个数据集，直接使用它；否则合并所有数据集
    if len(datasets) == 1:
        train_dataset = datasets[0]
    else:
        # 检查所有数据集是否共享同一个分词器
        first_tokenizer = datasets[0].tokenizer
        for i, ds in enumerate(datasets[1:], 1):
            if ds.tokenizer.vocab != first_tokenizer.vocab:
                print(f"Warning: Dataset {i} has a different tokenizer vocabulary!")
                # 可以选择合并词汇表或使用第一个数据集的词汇表
                # 这里我们简单地使用第一个数据集的词汇表
                ds.tokenizer = first_tokenizer
        
        train_dataset = ConcatDataset(datasets)
        # 为合并后的数据集设置分词器（用于collate_fn）
        train_dataset.tokenizer = first_tokenizer
    
    print(f"Total dataset size: {len(train_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, train_dataset.tokenizer)
    )
    
    # 使用第一个数据集的分词器作为模型的词汇表基础
    vocab_size = len(datasets[0].tokenizer.vocab)
    
    global_encoder = GlobalEncoder(
        vocab_size=vocab_size,
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
            
            # 添加检查代码
            if batch_idx == 0:  # 只检查第一个批次
                if batch_idx == 0:  # 仅第一个批次
                    print("\n===== 标签与模型输出匹配检查 =====")
                    print(f"标签形状: {labels.shape}")
                    print(f"标签内容: {labels[:2]}")  # 打印前两个样本的标签
                
                # 前向传播获取模型输出但不计算梯度
                with torch.no_grad():
                    test_outputs = model(
                        input_ids.transpose(0, 1),
                        graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(device),
                        src_key_padding_mask=(attention_mask == 0)
                    )
                    print(f"模型输出形状: {test_outputs.shape}")
                    print(f"模型输出内容: {test_outputs[:2]}")
                    print(f"损失函数: {criterion}")
                    print(f"损失值: {criterion(test_outputs, labels).item()}")

                for i in range(min(2, input_ids.size(0))):  # 检查前两个样本
                    tokens = datasets[0].tokenizer.convert_ids_to_tokens(input_ids[i].cpu().tolist())
                    sep_idx = tokens.index('[TREE_OP_SEP]') if '[TREE_OP_SEP]' in tokens else -1
                    
                    print(f"Sample {i} tokens up to [TREE_OP_SEP]:")
                    if sep_idx != -1:
                        print(tokens[:sep_idx])
                        print("\nTokens after [TREE_OP_SEP]:")
                        print(tokens[sep_idx:])
                        
                        # 检查attention_mask是否正确处理了[TREE_OP_SEP]之后的token
                        print("\nAttention mask around [TREE_OP_SEP]:")
                        print(attention_mask[i, max(0, sep_idx-5):min(attention_mask.size(1), sep_idx+5)])
                    else:
                        print("No [TREE_OP_SEP] found in tokens")
            
            # 只检查第一个epoch的第一个batch
            if epoch == 0 and batch_idx == 0:
                print("\n===== First Batch Input Verification =====")
                # 获取第一个样本作为示例
                sample_idx = 0
                sample_input = input_ids[sample_idx].cpu().tolist()
                sample_mask = attention_mask[sample_idx].cpu().tolist()
                sample_tokens = datasets[0].tokenizer.convert_ids_to_tokens(sample_input)
                
                # 找到TREE_OP_SEP的位置
                try:
                    sep_idx = sample_tokens.index('[TREE_OP_SEP]')
                    print(f"Found [TREE_OP_SEP] at position {sep_idx}")
                    
                    # 检查注意力掩码是否正确掩盖了TREE_OP_SEP之后的token
                    after_sep_masks = sample_mask[sep_idx:]
                    if all(m == 0 for m in after_sep_masks):
                        print("VERIFICATION PASSED: All tokens after [TREE_OP_SEP] are properly masked (attention_mask=0)")
                    else:
                        print("VERIFICATION FAILED: Some tokens after [TREE_OP_SEP] are not masked!")
                        # 打印未被正确mask的位置
                        unmasked = [i+sep_idx for i, m in enumerate(after_sep_masks) if m != 0]
                        print(f"Unmasked positions after [TREE_OP_SEP]: {unmasked}")
                        if unmasked:
                            print(f"Tokens at these positions: {[sample_tokens[i] for i in unmasked[:10]]}")
                    
                    # 显示TREE_OP_SEP前后的一些token和它们的掩码值
                    window = 5
                    start = max(0, sep_idx - window)
                    end = min(len(sample_tokens), sep_idx + window + 1)
                    print("\nTokens and masks around [TREE_OP_SEP]:")
                    for i in range(start, end):
                        print(f"{i}: {sample_tokens[i]} - Mask: {sample_mask[i]}")
                        
                except ValueError:
                    print("No [TREE_OP_SEP] found in the first sample")
            
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