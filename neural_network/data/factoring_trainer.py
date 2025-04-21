#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import glob
import json
import pickle
import argparse
import random
import numpy as np
from tqdm import tqdm

# 从你的代码中引入需要的模块
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
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, indent=2)
            print(f"筛选后的数据已保存到 {self.output_file}")
        
        return filtered_data

# 创建自定义Dataset类，用于划分训练集和验证集
class FactoringDataset(Dataset):
    """包装GraphSLIDataset以便于划分训练集和验证集"""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class FactoringSpecificTrainer:
    """专门针对factoring样本进行训练的类"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_loss = float('inf')
        
        # 设置随机种子
        set_seed(config.get('seed', 42))
        
        # 加载或提取factoring样本
        self._prepare_factoring_data()
        
        # 初始化模型
        self._initialize_model()
        
    def _prepare_factoring_data(self):
        """准备factoring数据集"""
        factoring_file = self.config.get('factoring_file')
        
        if factoring_file and os.path.exists(factoring_file):
            print(f"使用已有的factoring数据文件: {factoring_file}")
        else:
            input_dir = self.config.get('input_dir')
            if not input_dir:
                raise ValueError("必须提供输入目录路径")
                
            factoring_file = self.config.get('factoring_file', 'factoring_samples.json')
            print(f"从目录 {input_dir} 提取factoring样本到 {factoring_file}")
            extractor = FactoringDatasetExtractor(input_dir, factoring_file)
            extractor.extract()
        
        # 加载分词器
        tokenizer_path = self.config.get('tokenizer_path')
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"加载分词器: {tokenizer_path}")
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            print("未找到预训练分词器，将使用新的分词器")
            self.tokenizer = EnhancedTokenizer()
        
        # 创建数据集
        print(f"创建factoring训练数据集...")
        self.dataset = GraphSLIDataset(
            unified_file=factoring_file,
            max_seq_length=self.config.get('max_seq_length', 512),
            max_param_seq_length=self.config.get('max_param_seq_length', 30)
        )
        
        # 使用预训练分词器（如果有）
        if hasattr(self, 'tokenizer'):
            self.dataset.tokenizer = self.tokenizer
            
        print(f"数据集创建完成，包含 {len(self.dataset)} 个样本")
        
        # 划分训练集和验证集
        dataset_size = len(self.dataset)
        validation_split = self.config.get('validation_split', 0.2)
        validation_size = int(dataset_size * validation_split)
        
        # 创建随机索引
        indices = list(range(dataset_size))
        random.shuffle(indices)
        
        # 划分训练集和验证集索引
        train_indices = indices[validation_size:]
        val_indices = indices[:validation_size]
        
        # 将原始数据集分成训练集和验证集
        train_samples = [self.dataset[i] for i in train_indices]
        val_samples = [self.dataset[i] for i in val_indices]
        
        print(f"数据集已划分为训练集 ({len(train_indices)} 样本) 和验证集 ({len(val_indices)} 样本)")
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_samples,
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.dataset.tokenizer)
        )
        
        self.val_loader = DataLoader(
            val_samples,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.dataset.tokenizer)
        )
    
    def _initialize_model(self):
        """初始化模型"""
        # 获取或计算词汇表大小
        vocab_size = len(self.dataset.tokenizer.vocab)
        d_model = self.config.get('d_model', 512)
        nhead = self.config.get('nhead', 8)
        num_layers = self.config.get('num_layers', 6)
        
        print(f"初始化模型 (vocab_size={vocab_size}, d_model={d_model}, nhead={nhead}, num_layers={num_layers})")
        
        # 创建编码器和模型
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
        
        # 加载预训练模型权重（如果有）
        pretrained_model_path = self.config.get('pretrained_model_path')
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"加载预训练模型: {pretrained_model_path}")
            checkpoint = torch.load(pretrained_model_path, map_location=self.device)
            
            # 检查是否有模型状态字典
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"成功加载预训练模型 (Epoch {checkpoint.get('epoch', 'unknown')}, Loss {checkpoint.get('loss', 'unknown')})")
            else:
                # 假设是直接的状态字典
                self.model.load_state_dict(checkpoint)
                print("成功加载预训练模型权重")
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.get('lr', 1e-4)
        )
        self.criterion = nn.MSELoss()
    
    def train(self):
        """专注于factoring样本的训练过程"""
        epochs = self.config.get('epochs', 100)
        save_path = self.config.get('save_path', 'factoring_model.pth')
        
        print(f"开始训练 ({epochs} epochs)...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_train_loss = 0.0
            train_batch_count = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"[训练] Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                graph_mask = batch['graph_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids.transpose(0, 1),
                    graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(self.device),
                    src_key_padding_mask=(attention_mask == 0)
                )
                
                # 计算损失 - 我们重点关注FACTORING操作（索引为1）
                # 方法1: 简单的MSE损失
                loss = self.criterion(outputs, labels)
                
                # 方法2: 加权损失 - 给FACTORING更高的权重（索引为1）
                factoring_weight = 5.0  # 给FACTORING更高的权重
                weights = torch.ones_like(labels)
                weights[:, 1] = factoring_weight  # 设置FACTORING列的权重
                weighted_loss = self.criterion(outputs * weights, labels * weights)
                loss = weighted_loss / weights.mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                total_train_loss += loss.item()
                train_batch_count += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })
            
            avg_train_loss = total_train_loss / train_batch_count if train_batch_count > 0 else float('inf')
            
            # 验证阶段
            val_accuracy = self._evaluate_factoring_performance(self.val_loader, "验证集")
            
            print(f"Epoch {epoch+1}/{epochs} | 训练损失: {avg_train_loss:.4f} | 验证集Factoring准确率: {val_accuracy:.2%}")
            
            # 保存模型（如果是最佳的）
            if avg_train_loss < self.best_loss:
                self.best_loss = avg_train_loss
                save_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'vocab_size': len(self.dataset.tokenizer.vocab),
                    'd_model': self.config.get('d_model', 512),
                    'nhead': self.config.get('nhead', 8),
                    'num_layers': self.config.get('num_layers', 6),
                    'epoch': epoch,
                    'loss': self.best_loss,
                    'val_accuracy': val_accuracy
                }
                torch.save(save_dict, save_path)
                print(f"模型已保存到 {save_path}")
    
    def _evaluate_factoring_performance(self, data_loader, dataset_name="测试集"):
        """评估模型在factoring样本上的表现"""
        self.model.eval()
        correct_factoring = 0
        total_samples = 0
        factoring_index = 1  # FACTORING在操作列表中的索引
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                graph_mask = batch['graph_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids.transpose(0, 1),
                    graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(self.device),
                    src_key_padding_mask=(attention_mask == 0)
                )
                
                # 检查模型是否预测factoring操作
                predicted_actions = outputs.argmax(dim=1)
                correct_factoring += (predicted_actions == factoring_index).sum().item()
                total_samples += input_ids.size(0)
        
        accuracy = correct_factoring / total_samples if total_samples > 0 else 0
        print(f"{dataset_name} Factoring预测准确率: {accuracy:.2%} ({correct_factoring}/{total_samples})")
        
        return accuracy

    def test_model_on_original_data(self):
        """测试训练好的模型在原始数据集中FACTORING样本上的表现"""
        input_dir = self.config.get('input_dir')
        if not input_dir:
            print("未指定输入目录，无法测试原始数据")
            return
        
        print(f"正在测试模型在原始数据中FACTORING样本上的表现...")
        
        # 查找所有匹配的JSON文件
        data_files = glob.glob(os.path.join(input_dir, "*.json"))
        print(f"找到 {len(data_files)} 个JSON文件")
        
        if not data_files:
            print(f"错误: 在 {input_dir} 中没有找到JSON文件")
            return
        
        # 记录结果
        total_factoring_samples = 0
        total_correct_factoring = 0
        
        # 处理每个文件
        for file_path in tqdm(data_files, desc="测试原始文件"):
            try:
                # 创建临时数据集
                test_dataset = GraphSLIDataset(
                    unified_file=file_path,
                    max_seq_length=self.config.get('max_seq_length', 512),
                    max_param_seq_length=self.config.get('max_param_seq_length', 30)
                )
                
                # 使用当前分词器
                test_dataset.tokenizer = self.dataset.tokenizer
                
                # 找到具有FACTORING奖励大于0的样本索引
                factoring_samples = []
                for i in range(len(test_dataset.samples)):
                    sample = test_dataset.samples[i]
                    raw_sample = sample['raw_data']
                    global_reward = raw_sample.get("global_reward", {})
                    if isinstance(global_reward, dict) and "expected_by_type" in global_reward:
                        factoring_reward = global_reward["expected_by_type"].get("FACTORING", 0.0)
                        if factoring_reward > 0:
                            factoring_samples.append(sample)
                
                if not factoring_samples:
                    continue
                
                # 创建数据加载器
                test_loader = DataLoader(
                    factoring_samples,
                    batch_size=self.config.get('batch_size', 16),
                    shuffle=False,
                    collate_fn=lambda batch: collate_fn(batch, test_dataset.tokenizer)
                )
                
                # 评估模型表现
                self.model.eval()
                correct_factoring = 0
                factoring_index = 1  # FACTORING在操作列表中的索引
                
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        graph_mask = batch['graph_mask'].to(self.device)
                        
                        outputs = self.model(
                            input_ids.transpose(0, 1),
                            graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(self.device),
                            src_key_padding_mask=(attention_mask == 0)
                        )
                        
                        # 检查模型是否预测factoring操作
                        predicted_actions = outputs.argmax(dim=1)
                        correct_factoring += (predicted_actions == factoring_index).sum().item()
                
                # 更新总计数
                num_factoring_samples = len(factoring_samples)
                total_factoring_samples += num_factoring_samples
                total_correct_factoring += correct_factoring
                
                print(f"文件 {os.path.basename(file_path)}: {correct_factoring}/{num_factoring_samples} 正确 ({correct_factoring/num_factoring_samples:.2%})")
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
        
        # 计算总体准确率
        overall_accuracy = total_correct_factoring / total_factoring_samples if total_factoring_samples > 0 else 0
        print(f"\n原始数据中FACTORING样本的整体准确率: {overall_accuracy:.2%} ({total_correct_factoring}/{total_factoring_samples})")

def main():
    parser = argparse.ArgumentParser(description='专门针对factoring样本进行训练')
    
    # 数据相关参数
    parser.add_argument('--input_dir', type=str, default='/home/jiangguifei01/aiderun/fol-parser/fol-parser/data/',
                        help='输入JSON文件所在目录')
    parser.add_argument('--factoring_file', type=str, default='/home/jiangguifei01/aiderun/fol-parser/fol-parser/data/factoring_samples.json',
                        help='存储factoring样本的JSON文件路径')
    parser.add_argument('--tokenizer_path', type=str, default='/home/jiangguifei01/aiderun/fol-parser/fol-parser/neural_network/unified_tokenizer.pkl',
                        help='预训练分词器路径')
    
    # 模型相关参数
    parser.add_argument('--pretrained_model_path', type=str, default='first_stage_model.pth',
                        help='预训练模型路径')
    parser.add_argument('--d_model', type=int, default=512,
                        help='模型维度')
    parser.add_argument('--nhead', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Transformer层数')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--save_path', type=str, default='factoring_model.pth',
                        help='模型保存路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--extract_only', action='store_true',
                        help='仅提取factoring样本，不训练')
    parser.add_argument('--test_original', action='store_true',
                        help='训练后测试模型在原始数据中factoring样本上的表现')
    
    args = parser.parse_args()
    
    # 检查是否仅提取样本
    if args.extract_only:
        print("仅提取factoring样本...")
        extractor = FactoringDatasetExtractor(args.input_dir, args.factoring_file)
        extractor.extract()
        return
    
    # 配置参数
    config = vars(args)
    
    # 训练模型
    trainer = FactoringSpecificTrainer(config)
    trainer.train()
    
    # 如果需要，测试模型在原始数据上的表现
    if args.test_original:
        trainer.test_model_on_original_data()
    
    print("训练完成！")

if __name__ == "__main__":
    main()