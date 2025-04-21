#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import glob
import json
import pickle
from tqdm import tqdm
import argparse
import numpy as np
import random

# 从你的代码中引入需要的模块
from data.read_data import GraphSLIDataset, collate_fn, EnhancedTokenizer
from models.first_stage_model import GlobalEncoder, FirstStageModel
from torch.utils.data import DataLoader

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
        
        # 用来存储所有factoring样本
        all_factoring_samples = []
        
        # 处理每个文件
        for file_path in tqdm(data_files, desc="处理文件"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 获取该文件的图结构
                graph = data.get("graph", {})
                if not graph:
                    print(f"警告: {file_path} 中没有graph字段")
                    continue
                
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
                            # 为每个样本保存对应的图结构
                            factoring_sample = {
                                "graph": graph,
                                "sample": sample
                            }
                            all_factoring_samples.append(factoring_sample)
            
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
        
        print(f"从所有文件中找到 {len(all_factoring_samples)} 个FACTORING奖励大于0的样本")
        
        if len(all_factoring_samples) == 0:
            print("警告: 未找到FACTORING奖励大于0的样本")
            return None
        
        # 构建新的数据结构 - 每个样本都有自己的图结构
        filtered_data = {
            "samples": all_factoring_samples
        }
        
        # 如果提供了输出文件路径，保存筛选后的数据
        if self.output_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, indent=2)
            print(f"筛选后的数据已保存到 {self.output_file}")
        
        return filtered_data

class FactoringDataset(torch.utils.data.Dataset):
    """专门用于处理factoring样本数据集，确保每个样本都有自己的图结构"""
    def __init__(self, data, tokenizer, max_seq_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        self._process_samples()
    
    def _linearize_graph(self, graph_data):
        """将图结构线性化为token序列"""
        tokens = ["[GRAPH_START]"]
        for edge in graph_data.get("edges", []):
            for lit in edge.get("literals", []):
                args = lit.get("arguments", [])
                if len(args) >= 2:
                    predicate = lit.get("predicate", "UNK")
                    # 只处理E谓词
                    if predicate != "E":
                        continue
                    
                    token_pred = f"[PRED_{predicate}]"
                    token_arg1 = f"[{args[1]}]"  # 第二个节点
                    token_arg0 = f"[{args[0]}]"  # 第一个节点
                    
                    tokens.append(token_pred)
                    tokens.append(token_arg0)
                    tokens.append(token_arg1)
        tokens.append("[GRAPH_END]")
        return tokens
    
    def _linearize_tree(self, tree_data):
        """线性化状态树"""
        if not tree_data:
            return []
        nodes = {n['id']: n for n in tree_data.get('nodes', [])}
        roots = [n for n in tree_data.get('nodes', []) if n.get('depth', -1) == 0]
        if not roots:
            return []
        sequence = []
        stack = [(roots[0], 0)]
        while stack:
            node, depth = stack.pop()
            if not node: 
                continue
            node_type = node.get('type', 'UNKNOWN')
            literal = node.get('literal', {}) or {}
            substitution = node.get('substitution', {}) or {}
            children = node.get('children', [])
            # 统一只输出 [TYPE_x] 而非 [NODE_TYPE_x]
            node_features = [
                f"[TYPE_{node_type}]",
                f"[PRED_{literal.get('predicate', 'UNK')}]"
            ]
            if literal.get('negated', False):
                node_features.append("[NEG]")
            for arg in literal.get('arguments', []):
                if arg.startswith('VAR'):
                    node_features.append("[VAR]")
                elif arg.startswith('CONST'):
                    node_features.append(f"[{arg}]")
                else:
                    node_features.append(f"[ARG_{arg}]")
            for src, tgt in substitution.items():
                node_features += [f"[{src}]", f"[{tgt}]"]
            for child_id in reversed(children):
                child_node = nodes.get(child_id)
                if child_node:
                    stack.append((child_node, depth+1))
            sequence.extend(node_features)
        return sequence
    
    def _process_operations(self, operations):
        """线性化操作信息"""
        op_tokens = []
        for op in operations:
            if not op:
                continue
            action = op.get('action', 'UNKNOWN')
            operand2 = op.get('operand2', {})
            op_tokens.append(f"[ACTION_{action}]")
            op_tokens.append(f"[OP_TYPE_{operand2.get('type', 'UNKNOWN')}]")
        return op_tokens
    
    def _process_samples(self):
        """处理样本，确保每个样本都有自己的图结构"""
        for item in self.data:
            graph = item.get("graph", {})
            sample = item.get("sample", {})
            
            # 线性化图结构
            graph_tokens = self._linearize_graph(graph)
            graph_ids = self.tokenizer.convert_tokens_to_ids(graph_tokens)
            
            # 线性化状态树
            tree_tokens = self._linearize_tree(sample.get('state', {}).get('tree', {}))
            
            # 线性化操作
            op_tokens = self._process_operations(sample.get('available_ops', []))
            
            # 组合完整序列：图 + [SEP] + 状态树 + [TREE_OP_SEP] + 操作
            sep_id = self.tokenizer.vocab["[SEP]"]
            full_sequence = graph_ids + [sep_id] \
                           + self.tokenizer.convert_tokens_to_ids(tree_tokens) \
                           + self.tokenizer.convert_tokens_to_ids(["[TREE_OP_SEP]"]) \
                           + self.tokenizer.convert_tokens_to_ids(op_tokens)
            
            # 截断到最大长度
            full_sequence = full_sequence[:self.max_seq_length]
            
            # 生成标签
            labels = self._generate_action_labels(sample)
            
            # 保存处理后的样本
            self.samples.append({
                'input_ids': full_sequence,
                'raw_data': sample,
                'labels': labels,
            })
    
    def _generate_action_labels(self, sample):
        """生成操作标签"""
        op_seq = ["EXTENSION", "FACTORING", "ANCESTRY"]
        global_reward = sample.get("global_reward", {})
        if isinstance(global_reward, dict) and "expected_by_type" in global_reward:
            exp_reward = global_reward["expected_by_type"]
            return [exp_reward.get(op, 0.0) for op in op_seq]
        else:
            return [0.0 for _ in op_seq]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        pad_id = self.tokenizer.vocab['[PAD]']
        seq_len = len(sample['input_ids'])
        global_input_ids = sample['input_ids'] + [pad_id] * (self.max_seq_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
        
        # 创建图掩码
        graph_mask = self._create_graph_mask(torch.tensor(global_input_ids, dtype=torch.long))
        
        return {
            'input_ids': torch.tensor(global_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'graph_mask': graph_mask,
            'labels': torch.tensor(sample['labels'], dtype=torch.float),
            'raw_data': sample['raw_data']
        }
    
    def _create_graph_mask(self, input_ids):
        """创建图掩码"""
        mask = torch.zeros_like(input_ids)
        token_list = input_ids.tolist()
        try:
            start = token_list.index(self.tokenizer.vocab["[GRAPH_START]"])
            end = token_list.index(self.tokenizer.vocab["[GRAPH_END]"])
            mask[start:end+1] = 1
        except ValueError:
            pass
        return mask

class FactoringModelTester:
    """测试已训练好的模型在factoring样本上的表现"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 设置随机种子
        set_seed(config.get('seed', 42))
        
        # 加载分词器
        self._load_tokenizer()
        
        # 加载或提取factoring样本
        self._prepare_factoring_data()
        
        # 初始化模型
        self._load_model()
    
    def _load_tokenizer(self):
        """加载分词器"""
        tokenizer_path = self.config.get('tokenizer_path')
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"加载分词器: {tokenizer_path}")
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            print("未找到预训练分词器，将使用新的分词器")
            self.tokenizer = EnhancedTokenizer()
    
    def _prepare_factoring_data(self):
        """准备factoring数据集"""
        factoring_file = self.config.get('factoring_file')
        
        # 如果文件存在，加载数据
        if factoring_file and os.path.exists(factoring_file):
            print(f"加载已有的factoring数据文件: {factoring_file}")
            with open(factoring_file, 'r') as f:
                data = json.load(f)
            factoring_samples = data.get('samples', [])
        else:
            # 否则提取数据
            input_dir = self.config.get('input_dir')
            if not input_dir:
                raise ValueError("必须提供输入目录路径")
                
            factoring_file = self.config.get('factoring_file', 'factoring_samples.json')
            print(f"从目录 {input_dir} 提取factoring样本到 {factoring_file}")
            extractor = FactoringDatasetExtractor(input_dir, factoring_file)
            data = extractor.extract()
            if data:
                factoring_samples = data.get('samples', [])
            else:
                factoring_samples = []
        
        print(f"收集到 {len(factoring_samples)} 个factoring样本")
        
        # 创建自定义数据集
        max_seq_length = self.config.get('max_seq_length', 512)
        self.dataset = FactoringDataset(
            data=factoring_samples,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length
        )
        
        print(f"数据集创建完成，包含 {len(self.dataset)} 个样本")
        
        # 创建数据加载器
        batch_size = min(self.config.get('batch_size', 16), len(self.dataset))
        
        # 定义自定义的collate_fn
        def custom_collate_fn(batch):
            max_len = max(item['input_ids'].size(0) for item in batch)
            
            # 收集标签
            labels = torch.stack([item['labels'] for item in batch])
            
            # padding输入序列
            input_ids_list = []
            attention_mask_list = []
            graph_mask_list = []
            
            for item in batch:
                input_ids = item['input_ids']
                attention_mask = item['attention_mask']
                graph_mask = item['graph_mask']
                
                # 对齐到最大长度
                if input_ids.size(0) < max_len:
                    pad_size = max_len - input_ids.size(0)
                    pad_id = self.tokenizer.vocab['[PAD]']
                    
                    input_ids = torch.cat([
                        input_ids, 
                        torch.full((pad_size,), pad_id, dtype=input_ids.dtype)
                    ])
                    
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.zeros(pad_size, dtype=attention_mask.dtype)
                    ])
                    
                    graph_mask = torch.cat([
                        graph_mask,
                        torch.zeros(pad_size, dtype=graph_mask.dtype)
                    ])
                
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                graph_mask_list.append(graph_mask)
            
            return {
                'input_ids': torch.stack(input_ids_list),
                'attention_mask': torch.stack(attention_mask_list),
                'graph_mask': torch.stack(graph_mask_list),
                'labels': labels,
                'raw_data': [item['raw_data'] for item in batch]
            }
        
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )
    
    def _load_model(self):
        """加载预训练模型"""
        pretrained_model_path = self.config.get('pretrained_model_path')
        if not pretrained_model_path or not os.path.exists(pretrained_model_path):
            raise ValueError(f"预训练模型路径无效: {pretrained_model_path}")
        
        print(f"加载预训练模型: {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location=self.device)
        
        # 从checkpoint中获取模型配置
        if isinstance(checkpoint, dict) and 'd_model' in checkpoint:
            d_model = checkpoint.get('d_model', 512)
            nhead = checkpoint.get('nhead', 8)
            num_layers = checkpoint.get('num_layers', 6)
            vocab_size = checkpoint.get('vocab_size', len(self.tokenizer.vocab))
        else:
            # 使用默认配置
            d_model = self.config.get('d_model', 512)
            nhead = self.config.get('nhead', 8)
            num_layers = self.config.get('num_layers', 6)
            vocab_size = len(self.tokenizer.vocab)
        
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
        
        # 加载模型权重
        try:
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"成功加载模型权重 (Epoch {checkpoint.get('epoch', 'unknown')}, Loss {checkpoint.get('loss', 'unknown')})")
            else:
                # 假设是直接的状态字典
                self.model.load_state_dict(checkpoint)
                print("成功加载模型权重")
        except Exception as e:
            print(f"加载模型权重时出错: {e}")
            print("尝试部分加载模型权重...")
            
            # 尝试部分加载
            model_dict = self.model.state_dict()
            
            if 'model_state_dict' in checkpoint:
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
            else:
                pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f"成功部分加载模型权重 ({len(pretrained_dict)}/{len(model_dict)} 参数)")
        
        # 将模型移动到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
    
    def test(self):
        """测试模型在factoring样本上的表现，并输出详细信息"""
        print("开始测试模型在factoring样本上的表现...")
        
        factoring_index = 1  # FACTORING在操作列表中的索引
        total_samples = 0
        correct_factoring = 0
        
        # 用于保存详细结果
        detailed_results = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="测试样本"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                graph_mask = batch['graph_mask'].to(self.device)
                labels = batch.get('labels')
                
                if labels is not None:
                    labels = labels.to(self.device)
                
                # 预测
                outputs = self.model(
                    input_ids.transpose(0, 1),  # 模型需要序列长度在第一维
                    graph_mask=generate_square_subsequent_mask(input_ids.size(1)).to(self.device),
                    src_key_padding_mask=(attention_mask == 0)
                )
                
                # 获取预测结果
                predicted_actions = outputs.argmax(dim=1)
                scores = outputs.cpu().numpy()
                
                # 计算准确度
                is_correct = (predicted_actions == factoring_index)
                correct_factoring += is_correct.sum().item()
                total_samples += input_ids.size(0)
                
                # 收集详细结果
                for i in range(input_ids.size(0)):
                    sample_result = {
                        "sample_id": total_samples - input_ids.size(0) + i + 1,
                        "predicted_action": predicted_actions[i].item(),
                        "action_name": ["EXTENSION", "FACTORING", "ANCESTRY"][predicted_actions[i].item()],
                        "is_correct": is_correct[i].item() == 1,
                        "scores": scores[i].tolist(),
                        "factoring_score": scores[i][factoring_index]
                    }
                    
                    if labels is not None:
                        sample_result["true_labels"] = labels[i].cpu().numpy().tolist()
                    
                    detailed_results.append(sample_result)
        
        # 计算总体准确率
        accuracy = correct_factoring / total_samples if total_samples > 0 else 0
        print(f"\n测试结果:")
        print(f"总样本数: {total_samples}")
        print(f"正确预测为factoring的样本数: {correct_factoring}")
        print(f"准确率: {accuracy:.2%}")
        
        # 打印最后10个样本的详细结果
        # print("\n最后10个样本的详细结果:")
        print("\n所有样本的详细结果:")
        for result in detailed_results:
            print(f"样本ID: {result['sample_id']}")
            print(f"预测操作: {result['action_name']} (索引: {result['predicted_action']})")
            print(f"是否正确预测为factoring: {result['is_correct']}")
            print(f"各操作得分: {[f'{score:.4f}' for score in result['scores']]}")
            print(f"Factoring得分: {result['factoring_score']:.4f}")
            if "true_labels" in result:
                print(f"真实标签: {[f'{score:.4f}' for score in result['true_labels']]}")
            print("-" * 50)
        
        # 统计factoring得分的分布
        factoring_scores = [r["factoring_score"] for r in detailed_results]
        print("\nFactoring得分分布统计:")
        print(f"最小值: {min(factoring_scores):.4f}")
        print(f"最大值: {max(factoring_scores):.4f}")
        print(f"平均值: {sum(factoring_scores)/len(factoring_scores):.4f}")
        print(f"中位数: {sorted(factoring_scores)[len(factoring_scores)//2]:.4f}")
        
        # 统计各操作的预测次数
        action_counts = {}
        for result in detailed_results:
            action = result["action_name"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print("\n各操作的预测次数:")
        for action, count in action_counts.items():
            print(f"操作 {action}: {count} 次 ({count/total_samples:.2%})")
        
        # 分析准确率
        if "true_labels" in detailed_results[0]:
            print("\n模型性能分析:")
            # 对每个操作类型计算性能指标
            for i, action_name in enumerate(["EXTENSION", "FACTORING", "ANCESTRY"]):
                # 获取该操作的真实标签和预测值
                true_positives = sum(1 for r in detailed_results if r["predicted_action"] == i and r["true_labels"][i] > 0)
                false_positives = sum(1 for r in detailed_results if r["predicted_action"] == i and r["true_labels"][i] <= 0)
                true_negatives = sum(1 for r in detailed_results if r["predicted_action"] != i and r["true_labels"][i] <= 0)
                false_negatives = sum(1 for r in detailed_results if r["predicted_action"] != i and r["true_labels"][i] > 0)
                
                # 计算指标
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"\n{action_name} 性能指标:")
                print(f"  精确率(Precision): {precision:.4f}")
                print(f"  召回率(Recall): {recall:.4f}")
                print(f"  F1分数: {f1:.4f}")
                print(f"  混淆矩阵: [TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}]")
        
        return detailed_results

    def examine_model_inputs(self):
        """检查模型输入的组成，确认每个样本的图数据和SLITree正确拼接"""
        print("\n检查模型输入结构...")
        
        # 随机选择几个样本进行检查
        num_samples = min(3, len(self.dataset))
        sample_indices = random.sample(range(len(self.dataset)), num_samples)
        
        for i, idx in enumerate(sample_indices):
            print(f"\n样本 {i+1}/{num_samples} (索引 {idx}):")
            
            # 获取样本数据
            sample = self.dataset[idx]
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']
            graph_mask = sample['graph_mask']
            
            # 将id转换为token以便观察
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
            
            # 查找关键分隔符位置
            graph_start_idx = -1
            graph_end_idx = -1
            sep_idx = -1
            tree_op_sep_idx = -1
            
            for i, token in enumerate(tokens):
                if token == '[GRAPH_START]':
                    graph_start_idx = i
                elif token == '[GRAPH_END]':
                    graph_end_idx = i
                elif token == '[SEP]':
                    sep_idx = i
                elif token == '[TREE_OP_SEP]':
                    tree_op_sep_idx = i
            
            # 打印输入结构概览
            print(f"  输入结构概览:")
            print(f"  总长度: {len(tokens)}")
            print(f"  图开始位置 (GRAPH_START): {graph_start_idx}")
            print(f"  图结束位置 (GRAPH_END): {graph_end_idx}")
            print(f"  分隔符位置 (SEP): {sep_idx}")
            print(f"  树与操作分隔符位置 (TREE_OP_SEP): {tree_op_sep_idx}")
            
            # 打印图部分的token
            if graph_start_idx >= 0 and graph_end_idx >= 0:
                graph_tokens = tokens[graph_start_idx:graph_end_idx+1]
                print(f"\n  图部分 ({len(graph_tokens)} tokens):")
                print("  " + " ".join(graph_tokens[:50]) + ("..." if len(graph_tokens) > 50 else ""))
            
            # 打印SLITree部分的token
            if sep_idx >= 0 and tree_op_sep_idx >= 0:
                tree_tokens = tokens[sep_idx+1:tree_op_sep_idx]
                print(f"\n  SLITree部分 ({len(tree_tokens)} tokens):")
                print("  " + " ".join(tree_tokens[:50]) + ("..." if len(tree_tokens) > 50 else ""))
            
            # 打印操作部分的token
            if tree_op_sep_idx >= 0:
                op_tokens = tokens[tree_op_sep_idx+1:]
                print(f"\n  操作部分 ({len(op_tokens)} tokens):")
                print("  " + " ".join(op_tokens[:50]) + ("..." if len(op_tokens) > 50 else ""))
            
            # 打印有效部分的比例
            valid_count = attention_mask.sum().item()
            print(f"\n  有效token比例: {valid_count}/{len(tokens)} ({valid_count/len(tokens):.2%})")
            
            # 检查图掩码
            graph_mask_count = graph_mask.sum().item()
            print(f"  图掩码标记的token数: {graph_mask_count}")
            
            print("-" * 80)
            
        return True

def main():
    parser = argparse.ArgumentParser(description='测试模型在factoring样本上的表现')
    
    # 数据相关参数
    parser.add_argument('--input_dir', type=str, default='/home/jiangguifei01/aiderun/fol-parser/fol-parser/data/',
                        help='输入JSON文件所在目录')
    parser.add_argument('--factoring_file', type=str, default='/home/jiangguifei01/aiderun/fol-parser/fol-parser/data/factoring_samples_with_graphs.json',
                        help='存储factoring样本的JSON文件路径')
    parser.add_argument('--tokenizer_path', type=str, default='/home/jiangguifei01/aiderun/fol-parser/fol-parser/neural_network/unified_tokenizer.pkl',
                        help='预训练分词器路径')
    
    # 模型相关参数
    parser.add_argument('--pretrained_model_path', type=str, default='/home/jiangguifei01/aiderun/fol-parser/fol-parser/neural_network/first_stage_model.pth',
                        help='预训练模型路径')
    parser.add_argument('--d_model', type=int, default=512,
                        help='模型维度')
    parser.add_argument('--nhead', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Transformer层数')
    
    # 测试相关参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--extract_only', action='store_true',
                        help='仅提取factoring样本，不测试')
    parser.add_argument('--examine_inputs', action='store_true',
                        help='检查模型输入结构')
    
    args = parser.parse_args()
    
    # 检查是否仅提取样本
    if args.extract_only:
        print("仅提取factoring样本...")
        extractor = FactoringDatasetExtractor(args.input_dir, args.factoring_file)
        extractor.extract()
        return
    
    # 配置参数
    config = vars(args)
    
    # 创建测试器
    tester = FactoringModelTester(config)
    
    # 检查模型输入结构（如果需要）
    if args.examine_inputs:
        tester.examine_model_inputs()
    
    # 测试模型
    tester.test()
    
    print("测试完成！")

if __name__ == "__main__":
    main()