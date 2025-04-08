#!/usr/bin/env python3
import os
import json
import pickle
import torch
import numpy as np
import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，避免需要图形界面
import matplotlib.pyplot as plt
import sys
import json
import argparse
import signal 
from models.first_stage_model import GlobalEncoder, FirstStageModel
from models.second_stage_model import SecondStageModel

class NeuralHeuristicServer:
    def __init__(self, model_path, tokenizer_path):
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载分词器
        print(f"加载分词器: {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print(f"词汇表大小: {len(self.tokenizer.vocab)}")
        
        # 加载第一阶段模型
        print(f"加载第一阶段模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从保存的配置中恢复模型参数
        vocab_size = checkpoint.get('vocab_size', len(self.tokenizer.vocab))
        d_model = checkpoint.get('d_model', 512)
        nhead = checkpoint.get('nhead', 8)
        num_layers = checkpoint.get('num_layers', 6)
        
        print(f"模型配置: vocab_size={vocab_size}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        
        # 创建模型实例
        global_encoder = GlobalEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
        self.first_stage_model = FirstStageModel(global_encoder, d_model=d_model, num_actions=3)
        self.first_stage_model.load_state_dict(checkpoint['model_state_dict'])
        self.first_stage_model.to(self.device)
        self.first_stage_model.eval()
        
        print("第一阶段模型已加载并设置为评估模式")
        
        # 加载第二阶段模型（如果提供）
        second_stage_model_path = model_path.replace('first_stage', 'second_stage')
        if os.path.exists(second_stage_model_path):
            print(f"加载第二阶段模型: {second_stage_model_path}")
            second_checkpoint = torch.load(second_stage_model_path, map_location=self.device)
            
            # 从保存的配置中恢复模型参数
            second_stage_config = second_checkpoint.get('config', {})
            branch_hidden_dim = second_stage_config.get('branch_hidden_dim', 256)
            fusion_hidden_dim = second_stage_config.get('fusion_hidden_dim', 128)
            
            # 创建第二阶段模型实例
            self.second_stage_model = SecondStageModel(
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                branch_hidden_dim=branch_hidden_dim,
                fusion_hidden_dim=fusion_hidden_dim,
                tokenizer=self.tokenizer
            )
            
            self.second_stage_model.load_state_dict(second_checkpoint['model_state_dict'])
            self.second_stage_model.to(self.device)
            self.second_stage_model.eval()
            print("第二阶段模型已加载并设置为评估模式")
        else:
            print(f"未找到第二阶段模型: {second_stage_model_path}")
            self.second_stage_model = None
        
        # 初始化统计数据目录
        self.stats_dir = "action_stats"
        self.plots_dir = os.path.join(self.stats_dir, "plots")
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 初始化统计文件路径
        self.action_scores_file = os.path.join(self.stats_dir, "action_scores.npz")
        self.stats_summary_file = os.path.join(self.stats_dir, "stats_summary.json")
        
        # 初始化或加载统计摘要
        self._init_stats_summary()
        
        # 设置保存间隔
        self.stats_save_interval = 100  # 每处理100个请求保存一次统计数据
        
        print("神经启发式服务器初始化完成")

    def _init_stats_summary(self):
        """初始化或加载统计摘要信息"""
        if os.path.exists(self.stats_summary_file):
            try:
                with open(self.stats_summary_file, 'r') as f:
                    self.stats_summary = json.load(f)
                print(f"已加载统计摘要: 总请求数={self.stats_summary['total_requests']}")
            except Exception as e:
                print(f"加载统计摘要失败: {e}，将创建新统计")
                self._create_new_stats_summary()
        else:
            print("未找到统计摘要文件，将创建新统计")
            self._create_new_stats_summary()

    def _create_new_stats_summary(self):
        """创建新的统计摘要"""
        self.stats_summary = {
            "total_requests": 0,
            "selection_counts": {
                "Extension": 0,
                "Factoring": 0,
                "Ancestry": 0
            },
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_stats_summary()

    def _save_stats_summary(self):
        """保存统计摘要到文件"""
        self.stats_summary["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.stats_summary_file, 'w') as f:
            json.dump(self.stats_summary, f, indent=2)
            
    def _append_action_scores(self, action_scores, best_action_idx):
        """将新的操作评分添加到文件中"""
        action_names = ["Extension", "Factoring", "Ancestry"]
        best_action_name = action_names[best_action_idx]
        
        # 更新统计摘要
        self.stats_summary["total_requests"] += 1
        self.stats_summary["selection_counts"][best_action_name] += 1
        
        # 保存评分到npz文件
        if os.path.exists(self.action_scores_file):
            # 加载现有数据
            try:
                data = np.load(self.action_scores_file)
                extension_scores = list(data['Extension'])
                factoring_scores = list(data['Factoring'])
                ancestry_scores = list(data['Ancestry'])
            except Exception as e:
                print(f"加载评分数据失败: {e}，将创建新数据文件")
                extension_scores = []
                factoring_scores = []
                ancestry_scores = []
        else:
            extension_scores = []
            factoring_scores = []
            ancestry_scores = []
        
        # 添加新评分
        extension_scores.append(action_scores[0])
        factoring_scores.append(action_scores[1])
        ancestry_scores.append(action_scores[2])
        
        # 保存回文件
        np.savez(
            self.action_scores_file, 
            Extension=np.array(extension_scores),
            Factoring=np.array(factoring_scores),
            Ancestry=np.array(ancestry_scores)
        )
        
        # 定期保存统计摘要和生成图表
        if self.stats_summary["total_requests"] % self.stats_save_interval == 0:
            self._save_stats_summary()
            self._log_stats()
            self._generate_plots()

    def _log_stats(self):
        """记录当前统计信息到日志文件"""
        log_file = os.path.join(self.stats_dir, "stats_log.txt")
        total_requests = self.stats_summary["total_requests"]
        
        with open(log_file, 'a') as f:
            f.write(f"\n===== 统计更新 [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] =====\n")
            f.write(f"总请求数: {total_requests}\n")
            
            # 操作选择统计
            f.write("\n操作选择分布:\n")
            for action, count in self.stats_summary["selection_counts"].items():
                percentage = (count / total_requests * 100) if total_requests > 0 else 0
                f.write(f"  {action}: {count} 次 ({percentage:.2f}%)\n")
            
            # 如果有足够数据，计算评分统计
            if os.path.exists(self.action_scores_file):
                try:
                    data = np.load(self.action_scores_file)
                    f.write("\n评分统计:\n")
                    for action in ["Extension", "Factoring", "Ancestry"]:
                        scores = data[action]
                        if len(scores) > 0:
                            f.write(f"\n{action}:\n")
                            f.write(f"  样本数: {len(scores)}\n")
                            f.write(f"  评分范围: {scores.min():.4f} ~ {scores.max():.4f}\n")
                            f.write(f"  平均值: {scores.mean():.4f}\n")
                            f.write(f"  中位数: {np.median(scores):.4f}\n")
                            f.write(f"  标准差: {scores.std():.4f}\n")
                except Exception as e:
                    f.write(f"\n无法加载评分数据: {e}\n")
            
            f.write("=================================\n")
        
        print(f"统计信息已记录到 {log_file}")

    def _generate_plots(self):
        """生成评分分布图表"""
        if not os.path.exists(self.action_scores_file):
            print("无评分数据，无法生成图表")
            return
        
        try:
            # 加载评分数据
            data = np.load(self.action_scores_file)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 直方图 - 每个操作的评分分布
            plt.figure(figsize=(15, 5))
            
            actions = ["Extension", "Factoring", "Ancestry"]
            for i, action in enumerate(actions):
                plt.subplot(1, 3, i+1)
                scores = data[action]
                if len(scores) > 0:
                    plt.hist(scores, bins=30, alpha=0.7, density=True)
                    plt.axvline(scores.mean(), color='r', linestyle='dashed', linewidth=1)
                    plt.title(f"{action}评分分布")
                    plt.xlabel("评分")
                    plt.ylabel("密度")
            
            plt.tight_layout()
            hist_file = os.path.join(self.plots_dir, f"score_distributions_{timestamp}.png")
            plt.savefig(hist_file)
            plt.close()
            
            # 箱线图 - 比较三种操作的评分
            plt.figure(figsize=(10, 6))
            box_data = [data[action] for action in actions]
            plt.boxplot(box_data, labels=actions)
            plt.title("三种操作评分的箱线图比较")
            plt.ylabel("评分")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            boxplot_file = os.path.join(self.plots_dir, f"score_boxplot_{timestamp}.png")
            plt.savefig(boxplot_file)
            plt.close()
            
            # 另存一份最新图表(用于方便查看)
            hist_latest = os.path.join(self.plots_dir, "latest_score_distributions.png")
            boxplot_latest = os.path.join(self.plots_dir, "latest_score_boxplot.png")
            
            plt.figure(figsize=(15, 5))
            for i, action in enumerate(actions):
                plt.subplot(1, 3, i+1)
                scores = data[action]
                if len(scores) > 0:
                    plt.hist(scores, bins=30, alpha=0.7, density=True)
                    plt.axvline(scores.mean(), color='r', linestyle='dashed', linewidth=1)
                    plt.title(f"{action}评分分布")
                    plt.xlabel("评分")
                    plt.ylabel("密度")
            plt.tight_layout()
            plt.savefig(hist_latest)
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.boxplot(box_data, labels=actions)
            plt.title("三种操作评分的箱线图比较")
            plt.ylabel("评分")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(boxplot_latest)
            plt.close()
            
            print(f"图表已保存到 {self.plots_dir} 目录")
            
            # 保存评分统计摘要
            stats_data = {}
            for action in actions:
                scores = data[action]
                if len(scores) > 0:
                    stats_data[action] = {
                        "count": int(len(scores)),
                        "min": float(scores.min()),
                        "max": float(scores.max()),
                        "mean": float(scores.mean()),
                        "median": float(np.median(scores)),
                        "std": float(scores.std()),
                        "percentile_25": float(np.percentile(scores, 25)),
                        "percentile_75": float(np.percentile(scores, 75))
                    }
            
            score_stats_file = os.path.join(self.stats_dir, f"score_stats_{timestamp}.json")
            with open(score_stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
                
            # 同时更新最新的统计摘要
            latest_stats_file = os.path.join(self.stats_dir, "latest_score_stats.json")
            with open(latest_stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
            
        except Exception as e:
            print(f"生成图表失败: {e}")
    
    def generate_mask(self, sz):
        """生成Transformer注意力掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def json_to_tokens(self, json_data):
        """将JSON数据转换为tokens序列，使用与训练相同的逻辑"""
        # 提取状态树和图相关信息
        tree_data = json_data.get('state', {}).get('tree', {})
        graph_data = json_data.get('graph', {})
        
        # 构建图tokens
        graph_tokens = ["[GRAPH_START]"]
        for edge in graph_data.get("edges", []):
            for lit in edge.get("literals", []):
                args = lit.get("arguments", [])
                if len(args) >= 2 and lit.get("predicate", "") == "E":
                    token_pred = f"[PRED_E]"
                    token_arg1 = f"[{args[1]}]"  # 第二个节点
                    token_arg0 = f"[{args[0]}]"  # 第一个节点
                    graph_tokens.extend([token_pred, token_arg0, token_arg1])
        graph_tokens.append("[GRAPH_END]")
        
        # 构建树tokens
        tree_tokens = []
        nodes_dict = {n['id']: n for n in tree_data.get('nodes', [])}
        roots = [n for n in tree_data.get('nodes', []) if n.get('depth', -1) == 0]
        
        if roots:
            queue = [roots[0]]
            while queue:
                node = queue.pop(0)
                node_type = node.get('type', 'UNKNOWN')
                literal = node.get('literal', {})
                
                # 添加节点信息
                tree_tokens.append(f"[TYPE_{node_type}]")
                tree_tokens.append(f"[PRED_{literal.get('predicate', 'UNK')}]")
                
                if literal.get('negated', False):
                    tree_tokens.append("[NEG]")
                
                for arg in literal.get('arguments', []):
                    if arg.startswith('VAR'):
                        tree_tokens.append("[VAR]")
                    elif arg.startswith('CONST'):
                        tree_tokens.append(f"[{arg}]")
                    else:
                        tree_tokens.append(f"[ARG_{arg}]")
                
                # 处理子节点
                for child_id in node.get('children', []):
                    child = nodes_dict.get(child_id)
                    if child:
                        queue.append(child)
        
        # 添加操作tokens
        tree_tokens.append("[TREE_OP_SEP]")
        
        # 将所有tokens组合并转换为ID
        all_tokens = graph_tokens + ["[SEP]"] + tree_tokens
        
        return all_tokens
    
    def operation_to_tokens(self, operation):
        """将操作参数转换为tokens序列"""
        tokens = []
        
        # 添加操作类型
        action = operation.get("action", "")
        tokens.append(f"[ACTION_{action}]")
        
        # 处理第一个节点
        node1 = operation.get("node1", {})
        literal1 = node1.get("literal", {})
        
        tokens.append(f"[PRED_{literal1.get('predicate', 'UNK')}]")
        if literal1.get("negated", False):
            tokens.append("[NEG]")
        
        for arg in literal1.get("arguments", []):
            if isinstance(arg, str):
                if arg.startswith("VAR"):
                    tokens.append("[VAR]")
                elif arg.startswith("CONST"):
                    tokens.append(f"[{arg}]")
                else:
                    tokens.append(f"[ARG_{arg}]")
        
        # 处理第二个操作数
        operand2 = operation.get("operand2", {})
        operand2_type = operand2.get("type", "")
        
        if operand2_type == "literal":
            # 第二个操作数是字面量（Extension操作）
            literal2 = operand2.get("literal", {})
            tokens.append(f"[PRED_{literal2.get('predicate', 'UNK')}]")
            if literal2.get("negated", False):
                tokens.append("[NEG]")
            
            for arg in literal2.get("arguments", []):
                if isinstance(arg, str):
                    if arg.startswith("VAR"):
                        tokens.append("[VAR]")
                    elif arg.startswith("CONST"):
                        tokens.append(f"[{arg}]")
                    else:
                        tokens.append(f"[ARG_{arg}]")
        
        elif operand2_type == "node":
            # 第二个操作数是节点（Factoring或Ancestry操作）
            node2 = operand2.get("node", {})
            literal2 = node2.get("literal", {})
            
            if literal2:
                tokens.append(f"[PRED_{literal2.get('predicate', 'UNK')}]")
                if literal2.get("negated", False):
                    tokens.append("[NEG]")
                
                for arg in literal2.get("arguments", []):
                    if isinstance(arg, str):
                        if arg.startswith("VAR"):
                            tokens.append("[VAR]")
                        elif arg.startswith("CONST"):
                            tokens.append(f"[{arg}]")
                        else:
                            tokens.append(f"[ARG_{arg}]")
        
        # 添加KB子句信息（如果有）
        kb_clause = operation.get("kb_clause", [])
        if kb_clause:
            tokens.append("[KB_CLAUSE]")
            for lit in kb_clause:
                tokens.append(f"[PRED_{lit.get('predicate', 'UNK')}]")
                if lit.get("negated", False):
                    tokens.append("[NEG]")
                
                for arg in lit.get("arguments", []):
                    if isinstance(arg, str):
                        if arg.startswith("VAR"):
                            tokens.append("[VAR]")
                        elif arg.startswith("CONST"):
                            tokens.append(f"[{arg}]")
                        else:
                            tokens.append(f"[ARG_{arg}]")
        
        return tokens
    
    def process_first_stage_request(self, request_data):
        """处理第一阶段请求（获取操作类型评分）"""
        try:
            # 获取JSON格式的状态数据
            state_json = request_data.get('state', {})
            graph_json = request_data.get('graph', {})
            
            # 合并为完整的JSON
            combined_json = {
                'state': state_json,
                'graph': graph_json
            }
            
            # 转换为tokens序列
            tokens = self.json_to_tokens(combined_json)
            
            # 通过分词器转换为ID
            input_ids = []
            unknown_token_id = self.tokenizer.vocab.get('[UNK]', 1)
            for token in tokens:
                token_id = self.tokenizer.vocab.get(token, unknown_token_id)
                input_ids.append(token_id)
            
            # 准备模型输入
            max_len = 1024  # 设定一个最大长度
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            
            # 创建注意力掩码
            attention_mask = [1] * len(input_ids)
            
            # 确保树和操作分隔符之后的部分不被关注
            tree_op_sep_id = self.tokenizer.vocab.get("[TREE_OP_SEP]")
            if tree_op_sep_id is not None:
                try:
                    tree_op_sep_idx = input_ids.index(tree_op_sep_id)
                    for i in range(tree_op_sep_idx, len(attention_mask)):
                        attention_mask[i] = 0
                except ValueError:
                    pass  # 如果没有找到分隔符，不做特殊处理
            
            # 将输入数据转换为张量
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            attention_tensor = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
            
            # 生成掩码
            seq_len = input_tensor.size(1)
            graph_mask = self.generate_mask(seq_len).to(self.device)
            
            # 找到图部分进行特殊处理
            token_list = input_ids
            graph_attention_mask = torch.zeros(1, seq_len, dtype=torch.long).to(self.device)
            
            graph_start_id = self.tokenizer.vocab.get("[GRAPH_START]")
            graph_end_id = self.tokenizer.vocab.get("[GRAPH_END]")
            
            if graph_start_id is not None and graph_end_id is not None:
                try:
                    start = token_list.index(graph_start_id)
                    end = token_list.index(graph_end_id)
                    graph_attention_mask[0, start:end+1] = 1
                except ValueError:
                    pass
            
            # 模型推理
            with torch.no_grad():
                outputs = self.first_stage_model(
                    input_tensor.transpose(0, 1),  # 转为 [seq_len, 1]
                    graph_mask=graph_mask,
                    src_key_padding_mask=(attention_tensor == 0)
                )
                
                # 获取操作分数
                action_scores = outputs[0].cpu().numpy().tolist()
                
                # 找出最佳动作
                best_action_idx = np.argmax(action_scores)
                action_names = ["Extension", "Factoring", "Ancestry"]
                best_action_name = action_names[best_action_idx]
            
            # 记录统计数据
            self._append_action_scores(action_scores, best_action_idx)
                
            # 准备响应
            response = {
                'action_scores': action_scores,
                'best_action_idx': int(best_action_idx),
                'best_action_name': best_action_name,
                'status': 'success'
            }
            
            return response
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            # 出错时返回错误信息
            return {
                'status': 'error',
                'error_message': str(e),
                'error_details': error_details
            }
    
    def process_second_stage_request(self, request_data):
        """处理第二阶段请求（获取操作参数评分）"""
        try:
            if self.second_stage_model is None:
                return {
                    'status': 'error',
                    'error_message': '第二阶段模型未加载',
                }
            
            # 获取JSON格式的状态数据
            state_json = request_data.get('state', {})
            graph_json = request_data.get('graph', {})
            operations = request_data.get('operations', [])
            
            if not operations:
                return {
                    'status': 'error',
                    'error_message': '未提供操作参数',
                }
            
            # 合并为完整的JSON
            combined_json = {
                'state': state_json,
                'graph': graph_json
            }
            
            # 转换全局状态为tokens序列
            global_tokens = self.json_to_tokens(combined_json)
            # print('neural_server.py second stage ', global_tokens)
            
            # 通过分词器转换为ID，处理未知token
            unknown_token_id = self.tokenizer.vocab.get('[UNK]', 1)
            global_ids = []
            for token in global_tokens:
                token_id = self.tokenizer.vocab.get(token, unknown_token_id)
                global_ids.append(token_id)
            
            # 限制全局状态长度
            max_len = 1024
            if len(global_ids) > max_len:
                global_ids = global_ids[:max_len]
            
            # 处理每个操作参数
            candidate_tokens_list = []
            for operation in operations:
                op_tokens = self.operation_to_tokens(operation)
                candidate_tokens_list.append(op_tokens)
            
            # 将候选tokens转换为ID
            candidate_ids_list = []
            max_param_len = 0
            for tokens in candidate_tokens_list:
                ids = []
                for token in tokens:
                    token_id = self.tokenizer.vocab.get(token, unknown_token_id)
                    ids.append(token_id)
                candidate_ids_list.append(ids)
                max_param_len = max(max_param_len, len(ids))
            
            # 填充候选参数序列到相同长度
            pad_id = self.tokenizer.vocab.get('[PAD]', 0)
            padded_candidate_ids = []
            for ids in candidate_ids_list:
                padded_ids = ids + [pad_id] * (max_param_len - len(ids))
                padded_candidate_ids.append(padded_ids)
            
            # 创建注意力掩码
            global_attention_mask = [1] * len(global_ids)
            
            # 确保树和操作分隔符之后的部分不被关注
            tree_op_sep_id = self.tokenizer.vocab.get("[TREE_OP_SEP]")
            if tree_op_sep_id is not None:
                try:
                    tree_op_sep_idx = global_ids.index(tree_op_sep_id)
                    for i in range(tree_op_sep_idx, len(global_attention_mask)):
                        global_attention_mask[i] = 0
                except ValueError:
                    pass  # 如果没有找到分隔符，不做特殊处理
            
            # 将输入数据转换为张量
            global_tensor = torch.tensor(global_ids, dtype=torch.long).to(self.device).unsqueeze(1)  # [seq_len, 1]
            candidate_tensor = torch.tensor([padded_candidate_ids], dtype=torch.long).to(self.device)  # [1, num_candidates, max_param_len]
            global_attention_tensor = torch.tensor([global_attention_mask], dtype=torch.long).to(self.device)
            
            # 生成掩码
            seq_len = len(global_ids)
            graph_mask = self.generate_mask(seq_len).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                scores = self.second_stage_model(
                    global_tensor,  # [seq_len, 1]
                    candidate_tensor,  # [1, num_candidates, max_param_len]
                    graph_mask=graph_mask,
                    src_key_padding_mask=(global_attention_tensor == 0)
                )
                
                # 将评分转换为列表
                parameter_scores = scores[0].cpu().numpy().tolist()
                
                # 找出最佳参数
                best_param_idx = np.argmax(parameter_scores)
                
            # 准备响应
            response = {
                'parameter_scores': parameter_scores,
                'best_param_idx': int(best_param_idx),
                'status': 'success'
            }
            # print('use second stage model', response)
            return response
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            # 出错时返回错误信息
            return {
                'status': 'error',
                'error_message': str(e),
                'error_details': error_details
            }
    
    def process_request(self, request_data):
        """处理来自C++的请求数据"""
        try:
            # 确定请求类型
            request_type = request_data.get('request_type', 'action_scores')
            
            if request_type == 'action_scores':
                # 第一阶段请求：获取操作类型评分
                return self.process_first_stage_request(request_data)
            elif request_type == 'parameter_scores':
                # 第二阶段请求：获取操作参数评分
                return self.process_second_stage_request(request_data)
            else:
                return {
                    'status': 'error',
                    'error_message': f'未知的请求类型: {request_type}'
                }
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            # 出错时返回错误信息
            return {
                'status': 'error',
                'error_message': str(e),
                'error_details': error_details
            }
    
    def start_server(self):
        """启动服务器，从标准输入读取请求，向标准输出写入响应"""
        print("神经网络启发式服务器已启动，等待请求...")
        
        # 准备就绪信号
        print("READY", flush=True)
        
        while True:
            try:
                # 从stdin读取请求
                request_line = sys.stdin.readline().strip()
                
                # 检查是否为退出命令
                if request_line.lower() == "exit":
                    print("收到退出命令，服务器关闭", file=sys.stderr)
                    break
                
                # 解析JSON请求
                request_data = json.loads(request_line)
                
                # 处理请求
                response = self.process_request(request_data)
                
                # 发送响应
                print(json.dumps(response), flush=True)
                
            except KeyboardInterrupt:
                print("接收到中断信号，服务器关闭", file=sys.stderr)
                break
            except json.JSONDecodeError:
                print("JSON解析错误", file=sys.stderr)
                print(json.dumps({'status': 'error', 'error_message': '无效的JSON请求'}), flush=True)
            except Exception as e:
                print(f"处理请求时出错: {e}", file=sys.stderr)
                print(json.dumps({'status': 'error', 'error_message': str(e)}), flush=True)

def signal_handler(sig, frame):
    print("接收到信号，服务器关闭", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='神经网络启发式服务器')
    parser.add_argument('--model', default='first_stage_model.pth', help='第一阶段模型文件路径')
    parser.add_argument('--tokenizer', default='unified_tokenizer.pkl', help='分词器文件路径')
    args = parser.parse_args()
    
    server = NeuralHeuristicServer(args.model, args.tokenizer)
    server.start_server()