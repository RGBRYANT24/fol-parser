#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import pickle
import glob
import json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# 导入相同的模块
from data.read_data import GraphSLIDataset, collate_fn, EnhancedTokenizer
from models.first_stage_model import GlobalEncoder, FirstStageModel

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def test_phase1(model_path='first_stage_model.pth', tokenizer_path='unified_tokenizer.pkl', save_results=True):
    # 检查模型和分词器文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"分词器文件不存在: {tokenizer_path}")
    
    # 加载分词器
    print(f"加载分词器: {tokenizer_path}")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # 加载模型配置和权重
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # 从保存的配置中恢复模型参数
    vocab_size = checkpoint['vocab_size']
    d_model = checkpoint['d_model']
    nhead = checkpoint['nhead']
    num_layers = checkpoint['num_layers']
    
    print(f"模型配置: vocab_size={vocab_size}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    print(f"保存时的 epoch: {checkpoint['epoch']}, 损失: {checkpoint['loss']}")
    
    # 创建模型实例
    global_encoder = GlobalEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    model = FirstStageModel(global_encoder, d_model=d_model, num_actions=3)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为评估模式
    
    print(f"模型已加载并设置为评估模式，使用设备: {device}")
    
    # 加载测试数据
    config = {
        'batch_size': 16,
        'data_dir': '/home/jiangguifei01/aiderun/fol-parser/fol-parser/data',  # 与训练相同路径
        'data_pattern': 'test_data_*.json',  # 替换为您的测试数据模式
    }
    
    # 寻找测试数据文件
    test_files = glob.glob(os.path.join(config['data_dir'], config['data_pattern']))
    print(f"找到 {len(test_files)} 个测试数据文件")
    
    if not test_files:
        print("警告：没有找到测试数据文件，将尝试使用训练数据文件进行测试")
        test_files = glob.glob(os.path.join(config['data_dir'], 'training_data_*.json'))
        print(f"找到 {len(test_files)} 个训练数据文件用于测试")
    
    # 加载测试数据集
    test_datasets = []
    for file_path in test_files:
        try:
            # 使用保存的分词器创建一个自定义数据集
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
                    
                    # 保存原始样本ID或索引，以便跟踪结果
                    self.sample_ids = []
                    for i, sample in enumerate(raw_data):
                        sample_id = sample.get("id", f"sample_{i}")
                        self.sample_ids.append(sample_id)
            
            dataset = PresetTokenizerDataset(
                unified_file=file_path, 
                max_seq_length=512,
                preset_tokenizer=tokenizer
            )
            
            print(f"从 {file_path} 加载了测试数据集，包含 {len(dataset)} 个样本")
            test_datasets.append(dataset)
        except Exception as e:
            print(f"加载 {file_path} 的数据集时出错: {e}")
    
    if not test_datasets:
        raise ValueError("没有找到有效的测试数据集!")
    
    # 如果只有一个数据集，直接使用它；否则使用第一个
    test_dataset = test_datasets[0]
    print(f"使用测试数据集，大小: {len(test_dataset)} 个样本")
    
    # 创建数据加载器，不要使用shuffle，以保持样本顺序
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # 定义操作名称映射
    action_names = ["Extension", "Factoring", "Ancestry"]
    
    # 评估模型并保存所有样本的结果
    print("开始评估模型...")
    criterion = nn.MSELoss()  # 与训练中使用的损失函数相同
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # 用于保存所有结果的列表
    all_results = []
    sample_idx = 0
    
    with torch.no_grad():  # 不计算梯度
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_mask = batch['graph_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids.transpose(0, 1),  # 转换为 [seq_len, B]
                graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(device),
                src_key_padding_mask=(attention_mask == 0)
            )
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 计算预测准确率（假设最高分数对应的动作是预测）
            _, predicted = torch.max(outputs, 1)
            _, true_actions = torch.max(labels, 1)
            batch_correct = (predicted == true_actions).cpu().numpy()
            correct_predictions += batch_correct.sum()
            total_samples += labels.size(0)
            
            # 保存这个批次中每个样本的结果
            for i in range(input_ids.size(0)):
                # 尝试获取样本ID（如果数据集中存在）
                try:
                    sample_id = test_dataset.sample_ids[sample_idx] if hasattr(test_dataset, 'sample_ids') else f"sample_{sample_idx}"
                except IndexError:
                    sample_id = f"sample_{sample_idx}"
                
                pred_action = predicted[i].item()
                true_action = true_actions[i].item()
                is_correct = batch_correct[i]
                
                # 提取样本的原始输入（转换为token）
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().tolist())
                input_text = " ".join([t for t in input_tokens if t != '[PAD]'])
                
                # 保存结果
                result = {
                    'sample_id': sample_id,
                    'sample_idx': sample_idx,
                    'predicted_action': pred_action,
                    'predicted_action_name': action_names[pred_action] if pred_action < len(action_names) else f"Unknown_{pred_action}",
                    'true_action': true_action,
                    'true_action_name': action_names[true_action] if true_action < len(action_names) else f"Unknown_{true_action}",
                    'is_correct': is_correct,
                    'loss': loss.item(),
                    'predicted_scores': outputs[i].cpu().numpy().tolist(),
                    'true_scores': labels[i].cpu().numpy().tolist(),
                    'input_text': input_text[:100] + "..." if len(input_text) > 100 else input_text  # 保存部分输入文本
                }
                all_results.append(result)
                sample_idx += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(test_loader)} | Loss: {loss.item():.4f}")
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    print("\n评估结果:")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"准确率: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    # 分析每个操作的准确率
    results_df = pd.DataFrame(all_results)
    action_accuracy = {}
    
    for action in range(len(action_names)):
        action_samples = results_df[results_df['true_action'] == action]
        if len(action_samples) > 0:
            action_correct = action_samples['is_correct'].sum()
            action_accuracy[action_names[action]] = action_correct / len(action_samples)
            print(f"{action_names[action]}操作准确率: {action_accuracy[action_names[action]]:.4f} ({action_correct}/{len(action_samples)})")
    
    # 保存详细结果
    if save_results:
        results_file = 'test_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"已将详细测试结果保存到: {results_file}")
        
        # 创建一个更简洁的结果摘要
        summary_file = 'test_results_summary.csv'
        summary_df = results_df[['sample_id', 'predicted_action_name', 'true_action_name', 'is_correct']]
        summary_df.to_csv(summary_file, index=False)
        print(f"已将测试结果摘要保存到: {summary_file}")
    
    # 显示错误预测的样本
    print("\n错误预测的样本（最多显示10个）:")
    incorrect_samples = results_df[~results_df['is_correct']].head(10)
    for _, row in incorrect_samples.iterrows():
        print(f"样本ID: {row['sample_id']}")
        print(f"预测: {row['predicted_action_name']}, 真实: {row['true_action_name']}")
        print(f"预测分数: {row['predicted_scores']}")
        print(f"真实分数: {row['true_scores']}")
        print(f"输入文本片段: {row['input_text']}")
        print("-" * 80)
    
    return model, tokenizer, results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试第一阶段模型')
    parser.add_argument('--model', default='first_stage_model.pth', help='模型文件路径')
    parser.add_argument('--tokenizer', default='unified_tokenizer.pkl', help='分词器文件路径')
    parser.add_argument('--no-save', action='store_true', help='不保存测试结果')
    args = parser.parse_args()
    
    model, tokenizer, results = test_phase1(
        model_path=args.model, 
        tokenizer_path=args.tokenizer,
        save_results=not args.no_save
    )
    print("测试完成")