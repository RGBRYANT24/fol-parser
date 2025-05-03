#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import glob
import json
import pickle
import argparse
import random
import numpy as np
from tqdm import tqdm

# 从您的代码中引入需要的模块
from data.read_data import GraphSLIDataset, collate_fn, EnhancedTokenizer

# 导入模型相关的模块
from models.first_stage_model import GlobalEncoder, FirstStageModel

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class FactoringDatasetExtractor:
    """提取并保存仅包含FACTORING奖励大于0的样本"""
    def __init__(self, input_dir, output_file=None, data_pattern="*.json"):
        self.input_dir = input_dir
        self.output_file = output_file
        self.data_pattern = data_pattern
        
    def extract(self):
        """从文件夹中的所有JSON文件提取factoring样本并返回提取的数据"""
        print(f"从目录 {self.input_dir} 读取JSON文件...")
        
        # 查找所有匹配的JSON文件
        data_files = glob.glob(os.path.join(self.input_dir, self.data_pattern))
        print(f"找到 {len(data_files)} 个JSON文件")
        
        if not data_files:
            print(f"错误: 在 {self.input_dir} 中没有找到匹配 {self.data_pattern} 的文件")
            return None
        
        # 用来存储所有factoring样本和第一个有效的图结构
        all_factoring_samples = []
        first_graph = None
        
        # 处理每个文件
        for file_path in tqdm(data_files, desc="处理文件"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 保存第一个有效的图结构
                if first_graph is None and "graph" in data:
                    first_graph = data["graph"]
                
                # 提取搜索路径
                search_path = data.get("search_path", [])
                if not search_path:
                    print(f"警告: {file_path} 中没有search_path字段")
                    continue
                
                # 筛选出global_reward中FACTORING值大于0的样本
                for sample in search_path:
                    global_reward = sample.get("global_reward", {})
                    if isinstance(global_reward, dict) and "expected_by_type" in global_reward:
                        factoring_reward = global_reward["expected_by_type"].get("FACTORING", 0.0)
                        if factoring_reward > 0:
                            all_factoring_samples.append(sample)
            
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
        
        print(f"从所有文件中找到 {len(all_factoring_samples)} 个FACTORING奖励大于0的样本")
        
        if len(all_factoring_samples) == 0:
            print("警告: 未找到FACTORING奖励大于0的样本")
            return None
        
        # 构建新的数据结构
        filtered_data = {
            "graph": first_graph,
            "search_path": all_factoring_samples
        }
        
        # 如果提供了输出文件路径，保存筛选后的数据
        if self.output_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, indent=2)
            print(f"筛选后的数据已保存到 {self.output_file}")
        
        return filtered_data

class FactoringModelTester:
    """测试已训练模型在factoring样本上的表现"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        set_seed(config.get('seed', 42))
        
        # 提取factoring数据
        self.extract_factoring_data()
        
        # 加载模型及相关组件
        self.load_components()
    
    def extract_factoring_data(self):
        """提取factoring样本数据"""
        factoring_file = self.config.get('factoring_file')
        input_dir = self.config.get('input_dir')
        
        # 如果factoring文件已存在且不强制重新提取，则直接使用
        if os.path.exists(factoring_file) and not self.config.get('force_extract', False):
            print(f"使用已有的factoring数据文件: {factoring_file}")
        else:
            if not input_dir:
                raise ValueError("必须提供输入目录路径")
                
            print(f"从目录 {input_dir} 提取factoring样本到 {factoring_file}")
            extractor = FactoringDatasetExtractor(input_dir, factoring_file)
            extractor.extract()
            
            
        # 检查文件是否存在
        if not os.path.exists(factoring_file):
            raise FileNotFoundError(f"factoring数据文件不存在: {factoring_file}")
    
    def load_components(self):
        """加载分词器、预训练模型和数据集"""
        # 加载分词器
        tokenizer_path = self.config.get('tokenizer_path')
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"分词器文件不存在: {tokenizer_path}")
            
        print(f"加载分词器: {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # 创建数据集
        factoring_file = self.config.get('factoring_file')
        max_seq_length = self.config.get('max_seq_length', 512)
        max_param_seq_length = self.config.get('max_param_seq_length', 30)

        # print('load_components factoring file', factoring_file)
        
        print(f"创建factoring测试数据集... (max_seq_length={max_seq_length}, max_param_seq_length={max_param_seq_length})")
        self.dataset = GraphSLIDataset(
            unified_file=factoring_file,
            max_seq_length=max_seq_length,
            max_param_seq_length=max_param_seq_length
        )
        self.dataset.tokenizer = self.tokenizer
        print(f"数据集创建完成，包含 {len(self.dataset)} 个样本")
        
        # 创建数据加载器
        batch_size = self.config.get('batch_size', 16)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer)
        )
        
        # 加载预训练模型
        pretrained_model_path = self.config.get('pretrained_model_path')
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"预训练模型文件不存在: {pretrained_model_path}")
            
        print(f"加载预训练模型: {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location=self.device)
        
        # 获取模型参数
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            vocab_size = checkpoint.get('vocab_size', len(self.tokenizer.vocab))
            d_model = checkpoint.get('d_model', 512)
            nhead = checkpoint.get('nhead', 8)
            num_layers = checkpoint.get('num_layers', 6)
            model_state_dict = checkpoint['model_state_dict']
            print(f"使用checkpoint中的模型参数: vocab_size={vocab_size}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        else:
            # 如果没有保存参数，使用默认值或配置值
            vocab_size = len(self.tokenizer.vocab)
            d_model = self.config.get('d_model', 512)
            nhead = self.config.get('nhead', 8)
            num_layers = self.config.get('num_layers', 6)
            model_state_dict = checkpoint
            print(f"使用默认/配置的模型参数: vocab_size={vocab_size}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        
        # 创建模型
        self.global_encoder = GlobalEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
        self.model = FirstStageModel(
            self.global_encoder, 
            d_model=d_model, 
            num_actions=3
        )
        
        # 加载模型权重
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
    
    def run_test(self):
        """在factoring样本上测试模型并打印详细信息"""
        # 设置操作名称
        action_names = ["EXTENSION", "FACTORING", "ANCESTRY"]
        factoring_index = 1  # FACTORING在操作列表中的索引
        
        # 输出数量限制 (避免输出过多)
        max_samples_to_print = self.config.get('max_samples_to_print', 10)
        print_step = max(1, len(self.dataset) // max_samples_to_print)
        
        # 用于统计的变量
        results = []
        correct_counts = [0, 0, 0]  # 每个操作类型的正确预测数
        actual_counts = [0, 0, 0]   # 每个操作类型的实际样本数
        predicted_counts = [0, 0, 0]  # 每个操作类型的预测数
        total_correct = 0
        total_samples = 0
        print_counter = 0
        
        print("\n=== 开始样本测试，展示神经网络输出和原始global reward ===\n")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="测试进度")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                graph_mask = batch['graph_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 获取原始样本数据
                raw_samples = batch.get('raw_samples', [None] * input_ids.size(0))
                
                # 模型推理
                outputs = self.model(
                    input_ids.transpose(0, 1),
                    graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(self.device),
                    src_key_padding_mask=(attention_mask == 0)
                )

                print('------- 调试输出 -------')
                print('神经网络输出:', outputs.shape)
                if raw_samples:
                    print('原始样本数量:', len(raw_samples))
                    print('第一个原始样本类型:', type(raw_samples[0]))
                    print('第一个原始样本内容:')
                    if raw_samples[0]:
                        print('graph_node_id:', raw_samples[0].get('graph_node_id', '未知'))
                        print('global_reward:', raw_samples[0].get('global_reward', {}))
                else:
                    print('没有原始样本数据!')
                print('------------------------')
                
                # 应用softmax获取概率分布
                softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
                
                # 获取预测结果和实际标签
                predicted_actions = outputs.argmax(dim=1)
                actual_actions = labels.argmax(dim=1)
                
                # 处理每个样本
                for i in range(len(input_ids)):
                    pred_action = predicted_actions[i].item()
                    actual_action = actual_actions[i].item()
                    is_correct = (pred_action == actual_action)
                    
                    # 更新统计信息
                    predicted_counts[pred_action] += 1
                    actual_counts[actual_action] += 1
                    if is_correct:
                        correct_counts[actual_action] += 1
                        total_correct += 1
                    total_samples += 1
                    
                    # 构建详细结果
                    sample_result = {
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "input_token_ids": [token_id.item() for token_id in input_ids[i] if token_id.item() != 0],  # 移除padding
                        "predicted_action": pred_action,
                        "predicted_action_name": action_names[pred_action],
                        "actual_action": actual_action,
                        "actual_action_name": action_names[actual_action],
                        "is_correct": is_correct,
                        "network_outputs": outputs[i].cpu().tolist(),
                        "network_probabilities": softmax_outputs[i].cpu().tolist(),
                    }
                    
                    # 添加原始global reward信息
                    # 修改 run_test 方法中处理每个样本的部分
                    # 添加原始global reward信息
                    if raw_samples[i]:
                        global_reward = raw_samples[i].get("global_reward", {})
                        expected_rewards = {}
                        
                        # 处理不同格式的 global_reward
                        if isinstance(global_reward, dict):
                            if "expected_by_type" in global_reward:
                                expected_rewards = global_reward["expected_by_type"]
                            else:
                                expected_rewards = global_reward
                        
                        sample_result.update({
                            "graph_node_id": raw_samples[i].get("graph_node_id", "未知"),
                            "global_reward_expected": expected_rewards,
                            "factoring_reward": expected_rewards.get("FACTORING", 0.0),
                            "expand_reward": expected_rewards.get("EXTENSION", expected_rewards.get("EXPAND", 0.0)),
                            "distribute_reward": expected_rewards.get("ANCESTRY", expected_rewards.get("DISTRIBUTE", 0.0))
                        })
                    
                    results.append(sample_result)
                    
                    # 定期打印样本详情
                    if print_counter % print_step == 0 and print_counter < max_samples_to_print:
                        self._print_sample_details(sample_result)
                    
                    print_counter += 1
        
        # 计算总体准确率和FACTORING操作的准确率
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        factoring_accuracy = correct_counts[factoring_index] / actual_counts[factoring_index] if actual_counts[factoring_index] > 0 else 0
        
        # 打印总结
        print("\n=== 测试结果总结 ===")
        print(f"总样本数: {total_samples}")
        print(f"总体准确率: {overall_accuracy:.4f} ({total_correct}/{total_samples})")
        
        print("\n各操作类型的统计:")
        print("操作类型\t实际数量\t预测数量\t正确数量\t准确率")
        for i, action_name in enumerate(action_names):
            accuracy = correct_counts[i] / actual_counts[i] if actual_counts[i] > 0 else 0
            print(f"{action_name}\t{actual_counts[i]}\t\t{predicted_counts[i]}\t\t{correct_counts[i]}\t\t{accuracy:.4f}")
        
        # 保存详细结果
        results_file = self.config.get('results_file', 'factoring_test_results.json')
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "overall_accuracy": overall_accuracy,
                "factoring_accuracy": factoring_accuracy,
                "total_samples": total_samples,
                "total_correct": total_correct,
                "action_stats": {
                    action_names[i]: {
                        "actual": actual_counts[i],
                        "predicted": predicted_counts[i],
                        "correct": correct_counts[i],
                        "accuracy": correct_counts[i] / actual_counts[i] if actual_counts[i] > 0 else 0
                    } for i in range(len(action_names))
                },
                "detailed_results": results
            }, f, indent=2)
        
        print(f"\n详细测试结果已保存到: {results_file}")
        
        # 返回测试结果
        return {
            "overall_accuracy": overall_accuracy,
            "factoring_accuracy": factoring_accuracy,
            "results_file": results_file
        }
    
    def _print_sample_details(self, sample):
        """打印单个样本的详细信息，并直观对比神经网络输出与原始global reward"""
        print("\n" + "="*80)
        print(f"样本 ID: {sample.get('graph_node_id', '未知')}")
        
        # 1. 打印神经网络输出
        print("\n[神经网络输出]:")
        action_names = ["EXTENSION", "FACTORING", "ANCESTRY"]
        print("原始输出值:", sample['network_outputs'])
        print("概率分布:", [f"{p:.4f}" for p in sample['network_probabilities']])
        print(f"预测操作: {sample['predicted_action_name']} (索引: {sample['predicted_action']})")
        
        # 2. 打印原始奖励值
        print("\n[原始Global Reward]:")
        if 'global_reward_expected' in sample:
            rewards = sample['global_reward_expected']
            for action, reward in rewards.items():
                print(f"  {action}: {reward}")
        elif 'factoring_reward' in sample:
            # 使用已经提取的特定奖励值
            print(f"  FACTORING: {sample.get('factoring_reward', 0.0)}")
            print(f"  EXPAND: {sample.get('expand_reward', 0.0)}")
            print(f"  DISTRIBUTE: {sample.get('distribute_reward', 0.0)}")
        else:
            print("  (无Global Reward信息)")
        
        # 3. 添加直观对比
        print("\n[神经网络输出与原始奖励对比]:")
        
        # 创建一个映射，将action_names映射到可能的global_reward键
        action_to_reward_key = {
            "EXTENSION": ["EXTENSION", "EXPAND"],
            "FACTORING": ["FACTORING"],
            "ANCESTRY": ["ANCESTRY", "DISTRIBUTE"]
        }
        
        # 获取原始奖励值
        reward_values = {}
        if 'global_reward_expected' in sample:
            for action_name, possible_keys in action_to_reward_key.items():
                reward_values[action_name] = 0.0
                for key in possible_keys:
                    if key in sample['global_reward_expected']:
                        reward_values[action_name] = max(reward_values[action_name], sample['global_reward_expected'][key])
        elif 'factoring_reward' in sample:
            reward_values = {
                "EXTENSION": sample.get('expand_reward', 0.0),
                "FACTORING": sample.get('factoring_reward', 0.0),
                "ANCESTRY": sample.get('distribute_reward', 0.0)
            }
        
        # 打印对比表格
        print("操作类型\t神经网络概率\t原始奖励值\t差异")
        for i, action_name in enumerate(action_names):
            network_prob = sample['network_probabilities'][i]
            reward_value = reward_values.get(action_name, 0.0)
            diff = network_prob - reward_value
            print(f"{action_name}\t{network_prob:.4f}\t\t{reward_value:.4f}\t\t{diff:+.4f}")
        
        # 4. 打印预测结果与实际标签比较
        print("\n[预测结果]:")
        print(f"实际操作: {sample['actual_action_name']} (索引: {sample['actual_action']})")
        print(f"预测操作: {sample['predicted_action_name']} (索引: {sample['predicted_action']})")
        print(f"预测正确: {'✓' if sample['is_correct'] else '✗'}")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='测试已训练模型在factoring样本上的表现')
    
    # 数据相关参数
    parser.add_argument('--input_dir', type=str, default='/home/jiangguifei01/aiderun/fol-parser/fol-parser/data/newreward',
                        help='输入JSON文件所在目录')
    parser.add_argument('--factoring_file', type=str, default='factoring_samples.json',
                        help='存储factoring样本的JSON文件路径')
    parser.add_argument('--tokenizer_path', type=str, default='/home/jiangguifei01/aiderun/fol-parser/fol-parser/neural_network/unified_tokenizer.pkl',
                        help='预训练分词器路径')
    parser.add_argument('--force_extract', action='store_true',
                        help='强制重新提取factoring样本')
    
    # 模型相关参数
    parser.add_argument('--pretrained_model_path', type=str, default='/home/jiangguifei01/aiderun/fol-parser/fol-parser/neural_network/first_stage_model.pth',
                        help='预训练模型路径')
    parser.add_argument('--d_model', type=int, default=512,
                        help='模型维度(仅当模型checkpoint中没有此信息时使用)')
    parser.add_argument('--nhead', type=int, default=8,
                        help='注意力头数(仅当模型checkpoint中没有此信息时使用)')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Transformer层数(仅当模型checkpoint中没有此信息时使用)')
    
    # 测试相关参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--max_param_seq_length', type=int, default=30,
                        help='最大参数序列长度')
    parser.add_argument('--results_file', type=str, default='factoring_test_results.json',
                        help='测试结果保存路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--max_samples_to_print', type=int, default=10,
                        help='最多打印的样本数量')
    
    args = parser.parse_args()
    config = vars(args)
    
    # 运行测试
    tester = FactoringModelTester(config)
    test_results = tester.run_test()
    
    print("\n测试完成！")
    print(f"总体准确率: {test_results['overall_accuracy']:.4f}")
    print(f"FACTORING操作准确率: {test_results['factoring_accuracy']:.4f}")
    print(f"详细结果保存在: {test_results['results_file']}")

if __name__ == "__main__":
    main()