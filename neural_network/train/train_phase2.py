#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os
import glob
import json
import pickle
from collections import Counter

# 导入数据读取模块
from data.read_data import GraphSLIDataset, collate_fn, EnhancedTokenizer

# 导入第二阶段模型
from models.second_stage_model import SecondStageModel

def check_gpu():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"找到 {device_count} 个GPU设备:")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {device_name} (CUDA能力: {device_capability[0]}.{device_capability[1]})")
        current_device = torch.cuda.current_device()
        print(f"当前使用的GPU索引: {current_device}")
        print(f"GPU是否可用: {'是' if torch.cuda.is_available() else '否'}")
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU内存最大分配: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        return True
    else:
        print("警告：没有可用的GPU。训练将在CPU上进行，这会非常慢。")
        return False

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def analyze_candidate_tokens(dataset, tokenizer, num_samples=5):
    """分析候选操作参数中是否有未知token"""
    unk_id = tokenizer.vocab['[UNK]']
    total_candidates = 0
    total_tokens = 0
    total_unks = 0
    
    print("\n===== 候选操作参数Token分析 =====")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        candidate_ids = sample['candidate_param_ids']
        
        # 统计每个候选参数中的UNK数量
        for j in range(candidate_ids.size(0)):
            cand_tokens = tokenizer.convert_ids_to_tokens(candidate_ids[j].tolist())
            unk_count = cand_tokens.count('[UNK]')
            total_candidates += 1
            total_tokens += len([t for t in cand_tokens if t != '[PAD]'])
            total_unks += unk_count
            
            if unk_count > 0:
                print(f"样本 {i}, 候选参数 {j} 包含 {unk_count} 个UNK:")
                print(cand_tokens)
                
                # 如果发现UNK，尝试找出是哪些原始token被转为UNK
                if hasattr(dataset, 'samples') and i < len(dataset.samples):
                    raw_op = dataset.samples[i]['raw_data'].get('available_ops', [])[j]
                    print(f"原始操作数据: {raw_op}")
    
    print(f"\n总计: {total_candidates} 个候选参数, {total_tokens} 个token, {total_unks} 个UNK")
    print(f"UNK占比: {total_unks/total_tokens*100:.2f}% (如果>0，需要更新分词器)")

def load_unified_tokenizer(tokenizer_path):
    """加载统一分词器"""
    if os.path.exists(tokenizer_path):
        print(f"加载已有的统一分词器: {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"未找到统一分词器: {tokenizer_path}")
        return None

def train_phase2():
    # 设置使用第二个GPU (索引为1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = {
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 500,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'branch_hidden_dim': 256,
        'fusion_hidden_dim': 128,
        'save_path': 'second_stage_model.pth',
        'data_dir': '/home/jiangguifei01/aiderun/fol-parser/fol-parser/data',  # 指定数据目录
        'data_pattern': 'training_data_*.json',  # 指定数据文件匹配模式
        'tokenizer_path': 'unified_tokenizer.pkl',  # 统一分词器保存路径
        'first_stage_model_path': 'first_stage_model.pth',  # 第一阶段模型路径
        'check_candidates': True,  # 是否检查候选操作参数
    }
    
    check_gpu()
    
    # 寻找所有匹配的数据文件
    data_files = glob.glob(os.path.join(config['data_dir'], config['data_pattern']))
    print(f"找到 {len(data_files)} 个数据文件: {data_files}")
    
    # 1. 加载统一分词器
    unified_tokenizer = load_unified_tokenizer(config['tokenizer_path'])
    if not unified_tokenizer:
        print("无法找到统一分词器，将创建新的分词器")
        unified_tokenizer = EnhancedTokenizer()
    
    # 2. 检查第一阶段模型配置，保持一致性
    if os.path.exists(config['first_stage_model_path']):
        checkpoint = torch.load(config['first_stage_model_path'], map_location='cpu')
        if isinstance(checkpoint, dict) and 'd_model' in checkpoint:
            # 更新配置以匹配第一阶段模型
            config['d_model'] = checkpoint['d_model']
            config['nhead'] = checkpoint['nhead']
            config['num_layers'] = checkpoint['num_layers']
            print(f"从第一阶段模型加载配置: d_model={config['d_model']}, nhead={config['nhead']}, num_layers={config['num_layers']}")
    
    # 3. 为每个数据文件创建一个数据集
    datasets = []
    for file_path in data_files:
        try:
            # 创建一个使用预定义统一分词器的数据集
            class PresetTokenizerDataset(GraphSLIDataset):
                def __init__(self, unified_file, max_seq_length=768, max_param_seq_length=30, preset_tokenizer=None):
                    with open(unified_file, "r", encoding="utf-8") as f:
                        unified_data = json.load(f)
                    self.graph_data = unified_data.get("graph", {})
                    self.max_param_seq_length = max_param_seq_length
                    
                    self.tokenizer = preset_tokenizer
                    self._build_graph_vocab()
                    
                    raw_data = unified_data.get("search_path", [])
                    self.max_seq_length = max_seq_length * 2
                    self.samples = []
                    self._build_vocab(raw_data)
                    self._process_samples(raw_data)
            
            dataset = PresetTokenizerDataset(
                unified_file=file_path, 
                max_seq_length=512,
                preset_tokenizer=unified_tokenizer
            )
            
            print(f"从 {file_path} 加载了数据集，包含 {len(dataset)} 个样本")
            datasets.append(dataset)
        except Exception as e:
            print(f"加载 {file_path} 的数据集时出错: {e}")
    
    if not datasets:
        raise ValueError("没有找到有效的数据集!")
    
    # 4. 合并数据集
    if len(datasets) == 1:
        train_dataset = datasets[0]
    else:
        train_dataset = ConcatDataset(datasets)
        train_dataset.tokenizer = unified_tokenizer
    
    print(f"总数据集大小: {len(train_dataset)} 个样本")
    
    # 5. 检查候选操作参数中是否有未知token
    if config['check_candidates']:
        if isinstance(train_dataset, ConcatDataset):
            # 对于ConcatDataset，我们分析每个子数据集
            for i, ds in enumerate(datasets):
                print(f"\n分析数据集 {i+1}:")
                analyze_candidate_tokens(ds, unified_tokenizer)
        else:
            analyze_candidate_tokens(train_dataset, unified_tokenizer)
    
    # 6. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, unified_tokenizer)
    )
    
    # 7. 实例化第二阶段模型
    vocab_size = len(unified_tokenizer.vocab)
    print(f"词汇表大小: {vocab_size}")
    
    model = SecondStageModel(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        branch_hidden_dim=config['branch_hidden_dim'],
        fusion_hidden_dim=config['fusion_hidden_dim'],
        tokenizer=unified_tokenizer
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    # 8. 训练模型
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 全局输入 [B, global_seq_length]
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            # 候选操作参数 [B, num_candidates, param_seq_length]
            candidate_param_ids = batch['candidate_param_ids'].to(device)
            # 连续回归标签 [B, num_candidates]
            candidate_q_values = batch['candidate_q_values'].to(device)
            
            # 检查第一个batch的输入和输出
            if epoch == 0 and batch_idx == 0:
                print("\n===== 第二阶段模型输入输出检查 =====")
                print(f"输入IDs形状: {input_ids.shape}")
                print(f"注意力掩码形状: {attention_mask.shape}")
                print(f"图掩码形状: {graph_mask.shape}")
                print(f"候选参数IDs形状: {candidate_param_ids.shape}")
                print(f"候选得分形状: {candidate_q_values.shape}")
                
                # 检查候选得分中的非零值数量
                nonzero_scores = (candidate_q_values != 0).sum(dim=1)
                print(f"每个样本中非零候选得分数量: {nonzero_scores}")
                
                # 检查输入序列的token
                for i in range(min(2, input_ids.size(0))):
                    sample_tokens = unified_tokenizer.convert_ids_to_tokens(input_ids[i].cpu().tolist())
                    print(f"\n样本 {i} 的前50个token:")
                    print(sample_tokens[:50])
                    
                    # 检查候选参数
                    print(f"样本 {i} 的第一个候选参数:")
                    cand_tokens = unified_tokenizer.convert_ids_to_tokens(
                        candidate_param_ids[i, 0].cpu().tolist()
                    )
                    print(cand_tokens)
                    print(f"对应的候选得分: {candidate_q_values[i, 0].item()}")
                
                # 前向传播检查
                with torch.no_grad():
                    test_scores = model(
                        input_ids.transpose(0, 1),
                        candidate_param_ids,
                        graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(device),
                        src_key_padding_mask=(attention_mask == 0)
                    )
                    print(f"\n模型输出形状: {test_scores.shape}")
                    print(f"模型输出示例 (前2个样本的前5个得分):")
                    print(test_scores[:2, :5])
                    
                    # 验证模型是否只对有效候选参数输出得分
                    for i in range(min(2, test_scores.size(0))):
                        valid_cands = (candidate_param_ids[i, :, 0] != unified_tokenizer.vocab['[PAD]']).sum().item()
                        print(f"样本 {i} 有 {valid_cands} 个有效候选参数")
                        print(f"前 {valid_cands+2} 个得分: {test_scores[i, :valid_cands+2]}")
            
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
            batch_count += 1
            
            if batch_idx % 50 == 0:
                print(f"第二阶段 Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存模型时同时保存配置信息
            save_dict = {
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'd_model': config['d_model'],
                'nhead': config['nhead'],
                'num_layers': config['num_layers'],
                'branch_hidden_dim': config['branch_hidden_dim'],
                'fusion_hidden_dim': config['fusion_hidden_dim'],
                'epoch': epoch,
                'loss': best_loss,
            }
            torch.save(save_dict, config['save_path'])
            print(f"模型已保存到 {config['save_path']}")
        
        print(f"第二阶段 Epoch {epoch} | 平均损失: {avg_loss:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练第二阶段模型')
    parser.add_argument('--no-check', action='store_true', help='跳过候选参数检查')
    args = parser.parse_args()
    
    train_phase2()