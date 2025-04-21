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

# read_data 模块
from data.read_data import GraphSLIDataset, collate_fn, EnhancedTokenizer

# 从第一阶段模型模块中导入
from models.first_stage_model import GlobalEncoder, FirstStageModel

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

def analyze_tokenizer_coverage(datasets):
    """分析所有数据集的词汇表覆盖情况"""
    # 获取第一个数据集的词汇表
    first_tokenizer = datasets[0].tokenizer
    first_vocab = first_tokenizer.vocab
    
    # 对每个数据集进行分析
    for i, dataset in enumerate(datasets):
        # 如果是第一个数据集，跳过
        if i == 0:
            continue
            
        # 统计数据集中的词汇
        all_tokens = []
        unk_tokens = []
        
        # 仅采样部分数据以加快分析
        sample_size = min(len(dataset), 100)
        for idx in range(sample_size):
            sample = dataset[idx]
            tokens = first_tokenizer.convert_ids_to_tokens(sample['input_ids'].tolist())
            all_tokens.extend([t for t in tokens if t != '[PAD]'])
            
            # 使用原数据集的分词器获取原始token
            orig_tokens = dataset.tokenizer.convert_ids_to_tokens(
                dataset.tokenizer.convert_tokens_to_ids(tokens)
            )
            
            # 比较两个分词器的结果，找出变成UNK的词
            for orig_t, new_t in zip(orig_tokens, tokens):
                if new_t == '[UNK]' and orig_t != '[UNK]':
                    unk_tokens.append(orig_t)
        
        # 计算统计信息
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        unk_count = all_tokens.count('[UNK]')
        unique_unks = len(set(unk_tokens))
        
        # print(f"\n数据集 {i+1} 词汇表覆盖分析:")
        # print(f"  - 总token数: {total_tokens}")
        # print(f"  - 唯一token数: {unique_tokens}")
        # print(f"  - UNK token数: {unk_count} ({unk_count/total_tokens*100:.2f}%)")
        # print(f"  - 唯一UNK token数: {unique_unks}")
        
        # 如果有UNK，显示最常见的几个
        if unk_tokens:
            counter = Counter(unk_tokens)
            print("  - 最常见的UNK tokens:")
            for token, count in counter.most_common(10):
                print(f"    {token}: {count}次")

def build_unified_tokenizer(data_files, save_path='unified_tokenizer.pkl'):
    """从所有数据文件中构建统一的分词器"""
    print("构建统一分词器...")
    unified_tokenizer = EnhancedTokenizer()
    
    # 从所有数据文件中收集词汇
    all_tokens = set()
    for file_path in data_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 临时创建数据集以提取词汇
            temp_dataset = GraphSLIDataset(unified_file=file_path, max_seq_length=512)
            
            # 将这个数据集的词汇添加到统一集合中
            all_tokens.update(temp_dataset.tokenizer.vocab.keys())
            print(f"从 {file_path} 收集了 {len(temp_dataset.tokenizer.vocab)} 个tokens")
        except Exception as e:
            print(f"处理 {file_path} 时出错: {e}")
    
    # 向统一分词器添加所有收集到的词汇
    unified_tokenizer.add_tokens(all_tokens)
    print(f"统一分词器包含 {len(unified_tokenizer.vocab)} 个tokens")
    
    # 保存统一分词器
    with open(save_path, 'wb') as f:
        pickle.dump(unified_tokenizer, f)
    print(f"统一分词器已保存到 {save_path}")
    
    return unified_tokenizer

# 添加这个新函数用于分析数据集分布
def analyze_dataset_distribution(dataset, file_path):
    """分析数据集中操作类型的分布情况"""
    # print(f"\n===== 数据集分布分析: {os.path.basename(file_path)} =====")
    
    # 统计不同操作类型的计数
    action_counts = {"ANCESTRY": 0, "EXTENSION": 0, "FACTORING": 0}
    positive_action_counts = {"ANCESTRY": 0, "EXTENSION": 0, "FACTORING": 0}
    
    for i in range(len(dataset.samples)):
        sample = dataset.samples[i]
        raw_sample = sample['raw_data']
        
        # 统计所有操作数量
        for op in raw_sample.get('available_ops', []):
            action = op.get('action', '')
            if action in action_counts:
                action_counts[action] += 1
        
        # 统计正例操作数量（global_reward > 0）
        global_reward = raw_sample.get("global_reward", {})
        if isinstance(global_reward, dict) and "expected_by_type" in global_reward:
            exp_reward = global_reward["expected_by_type"]
            for action, reward in exp_reward.items():
                if action in positive_action_counts and reward > 0:
                    positive_action_counts[action] += 1
    
    total_ops = sum(action_counts.values())
    actual_ratios = {k: v/total_ops if total_ops > 0 else 0 for k, v in action_counts.items()}
    
    total_positive_ops = sum(positive_action_counts.values())
    positive_ratios = {k: v/total_positive_ops if total_positive_ops > 0 else 0 
                      for k, v in positive_action_counts.items()}
    
    # print("全部操作分布:")
    # print(f"操作计数: {action_counts}")
    # print(f"操作比例: {actual_ratios}")
    
    # print("\n正例操作分布 (reward > 0):")
    # print(f"正例操作计数: {positive_action_counts}")
    # print(f"正例操作比例: {positive_ratios}")
    
    # 分析标签分布
    all_labels = []
    for i in range(len(dataset.samples)):
        sample = dataset.samples[i]
        if 'labels' in sample and sample['labels'] is not None:
            try:
                # 确保标签是tensor对象
                if isinstance(sample['labels'], list):
                    label_tensor = torch.tensor(sample['labels'], dtype=torch.float)
                else:
                    label_tensor = sample['labels']
                
                all_labels.append(label_tensor)
            except Exception as e:
                print(f"处理样本 {i} 的标签时出错: {e}")
    
    # if all_labels:
    #     try:
    #         all_labels_tensor = torch.stack(all_labels)
    #         print("\n标签分布统计:")
    #         for i, action in enumerate(["EXTENSION", "FACTORING", "ANCESTRY"]):
    #             labels = all_labels_tensor[:, i]
    #             positive_count = (labels > 0).sum().item()
    #             print(f"{action}: 正例({labels > 0}): {positive_count}, 比例: {positive_count/len(all_labels):.4f}")
    #             print(f"  - 平均值: {labels.mean().item():.4f}, 最大值: {labels.max().item():.4f}")
    #     except Exception as e:
    #         print(f"分析标签分布时出错: {e}")

def train_phase1(use_unified_tokenizer=True):
    config = {
        'batch_size': 16,
        'lr': 1e-4,
        'epochs': 1000,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'save_path': 'first_stage_model.pth',
        'data_dir': '/home/jiangguifei01/aiderun/fol-parser/fol-parser/data',  # 指定数据目录
        'data_pattern': 'training_data_*.json',  # 指定数据文件匹配模式
        'tokenizer_path': '/home/jiangguifei01/aiderun/fol-parser/fol-parser/neural_network/unified_tokenizer.pkl',  # 统一分词器保存路径
        'vocab_analysis': True,  # 是否分析词汇表覆盖情况
    }
    
    # 寻找所有匹配的数据文件
    data_files = glob.glob(os.path.join(config['data_dir'], config['data_pattern']))
    print(f"找到 {len(data_files)} 个数据文件: {data_files}")
    
    # 决定是否使用统一分词器
    unified_tokenizer = None
    if use_unified_tokenizer:
        # 检查是否存在已保存的统一分词器
        if os.path.exists(config['tokenizer_path']):
            print(f"加载已有的统一分词器: {config['tokenizer_path']}")
            with open(config['tokenizer_path'], 'rb') as f:
                unified_tokenizer = pickle.load(f)
        else:
            # 构建新的统一分词器
            print('未加载已有分词器')
            unified_tokenizer = build_unified_tokenizer(data_files, config['tokenizer_path'])
    
    # 为每个数据文件创建一个数据集
    datasets = []
    print('os path ', os.path)
    print('file path ',os.path.join(config['data_dir'], config['data_pattern']))
    for file_path in data_files:
        try:
            # 添加此行以设置平衡比例
            balance_ratio = {
                "ANCESTRY": 0.5,
                "EXTENSION": 0.45,
                "FACTORING": 12.14
            }
            # 如果使用统一分词器，传递给数据集构造函数
            if unified_tokenizer:
                # 创建一个自定义的GraphSLIDataset子类实例，它使用预定义的分词器
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
                    preset_tokenizer=unified_tokenizer,
                    balance_ratio=balance_ratio  # 添加平衡比例参数
                )
            else:
                dataset = GraphSLIDataset(
                    unified_file=file_path, 
                    max_seq_length=512,
                    balance_ratio=balance_ratio  # 添加平衡比例参数
                )
            
            # print(f"从 {file_path} 加载了数据集，包含 {len(dataset)} 个样本")
            # 添加此代码分析数据集分布
            analyze_dataset_distribution(dataset, file_path)
            datasets.append(dataset)
        except Exception as e:
            print(f"加载 {file_path} 的数据集时出错: {e}")
    
    if not datasets:
        raise ValueError("没有找到有效的数据集!")
    
    # 如果只有一个数据集，直接使用它；否则合并所有数据集
    if len(datasets) == 1:
        train_dataset = datasets[0]
    else:
        # 如果要使用统一分词器但尚未创建，则从第一个数据集获取
        if not unified_tokenizer:
            print("使用第一个数据集的分词器作为基准")
            first_tokenizer = datasets[0].tokenizer
            # 分析词汇表覆盖情况
            if config['vocab_analysis']:
                analyze_tokenizer_coverage(datasets)
            
            # 更新其他数据集的分词器为第一个数据集的分词器
            for i, ds in enumerate(datasets[1:], 1):
                ds.tokenizer = first_tokenizer
        
        # 使用ConcatDataset合并所有数据集
        train_dataset = ConcatDataset(datasets)
        # 为合并后的数据集设置分词器（用于collate_fn）
        if unified_tokenizer:
            train_dataset.tokenizer = unified_tokenizer
        else:
            train_dataset.tokenizer = datasets[0].tokenizer
    
    print(f"总数据集大小: {len(train_dataset)} 个样本")

    # 添加这段代码分析合并数据集的分布
    print("\n===== 合并后的数据集分布分析 =====")
    overall_action_counts = {"ANCESTRY": 0, "EXTENSION": 0, "FACTORING": 0}
    overall_positive_counts = {"ANCESTRY": 0, "EXTENSION": 0, "FACTORING": 0}

    for dataset in datasets:
        for i in range(len(dataset.samples)):
            sample = dataset.samples[i]
            raw_sample = sample['raw_data']
            
            # 统计操作
            for op in raw_sample.get('available_ops', []):
                action = op.get('action', '')
                if action in overall_action_counts:
                    overall_action_counts[action] += 1
            
            # 统计正例
            global_reward = raw_sample.get("global_reward", {})
            if isinstance(global_reward, dict) and "expected_by_type" in global_reward:
                exp_reward = global_reward["expected_by_type"]
                for action, reward in exp_reward.items():
                    if action in overall_positive_counts and reward > 0:
                        overall_positive_counts[action] += 1

    total_ops = sum(overall_action_counts.values())
    overall_ratios = {k: v/total_ops if total_ops > 0 else 0 for k, v in overall_action_counts.items()}

    total_positive = sum(overall_positive_counts.values())
    positive_ratios = {k: v/total_positive if total_positive > 0 else 0 for k, v in overall_positive_counts.items()}

    print("全部操作分布:")
    print(f"操作计数: {overall_action_counts}")
    print(f"操作比例: {overall_ratios}")

    print("\n正例操作分布 (reward > 0):")
    print(f"正例操作计数: {overall_positive_counts}")
    print(f"正例操作比例: {positive_ratios}")
        
    # 保存最终使用的分词器，以便在测试阶段使用
    final_tokenizer = unified_tokenizer if unified_tokenizer else datasets[0].tokenizer
    if not os.path.exists(config['tokenizer_path']) or not unified_tokenizer:
        with open(config['tokenizer_path'], 'wb') as f:
            pickle.dump(final_tokenizer, f)
        print(f"分词器已保存到 {config['tokenizer_path']}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, final_tokenizer)
    )
    
    # 使用最终分词器的词汇表大小
    vocab_size = len(final_tokenizer.vocab)
    print(f"词汇表大小: {vocab_size}")
    
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
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)         # [B, seq_len]
            attention_mask = batch['attention_mask'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            labels = batch['labels'].to(device)                 # [B, 3]
            
            # 添加检查代码
            if epoch == 0 and batch_idx == 0:  # 只检查第一个epoch的第一个批次
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
                    tokens = final_tokenizer.convert_ids_to_tokens(input_ids[i].cpu().tolist())
                    sep_idx = tokens.index('[TREE_OP_SEP]') if '[TREE_OP_SEP]' in tokens else -1
                    
                    print(f"样本 {i} 的tokens直到 [TREE_OP_SEP]:")
                    if sep_idx != -1:
                        print(tokens[:sep_idx])
                        print("\n[TREE_OP_SEP]之后的tokens:")
                        print(tokens[sep_idx:])
                        
                        # 检查attention_mask是否正确处理了[TREE_OP_SEP]之后的token
                        print("\n[TREE_OP_SEP]周围的attention mask:")
                        print(attention_mask[i, max(0, sep_idx-5):min(attention_mask.size(1), sep_idx+5)])
                    else:
                        print("在tokens中未找到 [TREE_OP_SEP]")
                
                # 检查UNK token
                unk_id = final_tokenizer.vocab['[UNK]']
                unk_positions = (input_ids == unk_id).nonzero(as_tuple=True)
                if unk_positions[0].size(0) > 0:
                    print(f"\n在第一个batch中发现 {unk_positions[0].size(0)} 个UNK tokens")
                    unk_counts = []
                    for b in range(input_ids.size(0)):
                        batch_unks = (input_ids[b] == unk_id).sum().item()
                        if batch_unks > 0:
                            unk_counts.append((b, batch_unks))
                    
                    print("每个样本中UNK的数量:")
                    for b, count in unk_counts:
                        print(f"  样本 {b}: {count} 个UNK tokens")
                else:
                    print("\n在第一个batch中没有发现UNK tokens")
            
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
            batch_count += 1
            
            if batch_idx % 50 == 0:
                print(f"第1阶段 Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
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
                'epoch': epoch,
                'loss': best_loss,
            }
            torch.save(save_dict, config['save_path'])
            print(f"模型已保存到 {config['save_path']}")
        
        print(f"第1阶段 Epoch {epoch} | 平均损失: {avg_loss:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练第一阶段模型')
    parser.add_argument('--unified', action='store_true', help='使用统一的分词器')
    args = parser.parse_args()
    
    train_phase1(use_unified_tokenizer=args.unified)