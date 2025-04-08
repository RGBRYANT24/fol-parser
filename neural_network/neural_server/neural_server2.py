#!/usr/bin/env python3
import os
import sys
import json
import pickle
import torch
import numpy as np
import signal
import argparse
from models.first_stage_model import GlobalEncoder
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
        
        # 直接加载第二阶段模型
        print(f"加载第二阶段模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从保存的配置中恢复模型参数
        vocab_size = checkpoint.get('vocab_size', len(self.tokenizer.vocab))
        d_model = checkpoint.get('d_model', 512)
        nhead = checkpoint.get('nhead', 8)
        num_layers = checkpoint.get('num_layers', 6)
        
        # 第二阶段模型的特定配置
        second_stage_config = checkpoint.get('config', {})
        branch_hidden_dim = second_stage_config.get('branch_hidden_dim', 256)
        fusion_hidden_dim = second_stage_config.get('fusion_hidden_dim', 128)
        
        print(f"模型配置: vocab_size={vocab_size}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        print(f"第二阶段配置: branch_hidden_dim={branch_hidden_dim}, fusion_hidden_dim={fusion_hidden_dim}")
        
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
        
        self.second_stage_model.load_state_dict(checkpoint['model_state_dict'])
        self.second_stage_model.to(self.device)
        self.second_stage_model.eval()
        
        print("第二阶段模型已加载并设置为评估模式")
        
        # 第一阶段模型设为None，因为我们只使用第二阶段
        self.first_stage_model = None
    
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
    
    def process_second_stage_request(self, request_data):
        # print('second stage model in process_second_stage_request')
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
            # print('neural_server2.py second stage ', global_tokens)
            
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
            # print('second stage candidate_tokens_list', candidate_tokens_list)
            
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
    
    def process_first_stage_request(self, request_data):
        """处理第一阶段请求（获取操作类型评分）"""
        # 由于这是第二阶段专用服务器，我们不处理第一阶段请求
        # print('second stage using wrong model')
        return {
            'status': 'error',
            'error_message': '这是第二阶段专用服务器，不支持第一阶段请求'
        }
    
    def process_request(self, request_data):
        """处理来自C++的请求数据"""
        try:
            # 确定请求类型
            request_type = request_data.get('request_type', 'action_scores')
            # print('second stage request_type',request_type)
            
            if request_type == 'action_scores':
                # 第一阶段请求：不支持
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
            # print('second stage process_request error', error_details)
            # 出错时返回错误信息
            return {
                'status': 'error',
                'error_message': str(e),
                'error_details': error_details
            }
    
    def start_server(self):
        """启动服务器，从标准输入读取请求，向标准输出写入响应"""
        print("神经网络启发式服务器（第二阶段）已启动，等待请求...", file=sys.stderr)
        
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
                
                # 调试信息输出到stderr
                # print(f"收到请求: {request_line}", file=sys.stderr)
                
                # 解析JSON请求
                try:
                    request_data = json.loads(request_line)
                    # 处理请求
                    response = self.process_request(request_data)
                    
                    # 确保响应不是None
                    if response is None:
                        print("警告：响应为None，替换为错误消息", file=sys.stderr)
                        response = {
                            'status': 'error',
                            'error_message': '服务器返回了None'
                        }
                    
                    # 发送响应前检查是否为有效的JSON格式
                    response_str = json.dumps(response)
                    
                    # 调试信息输出到stderr
                    # print(f"发送响应: {response_str}", file=sys.stderr)
                    
                    # 实际响应输出到stdout并确保刷新
                    print(response_str, flush=True)
                    
                except json.JSONDecodeError:
                    print("JSON解析错误", file=sys.stderr)
                    print(json.dumps({'status': 'error', 'error_message': '无效的JSON请求'}), flush=True)
                except Exception as e:
                    print(f"处理请求时出错: {e}", file=sys.stderr)
                    print(json.dumps({'status': 'error', 'error_message': str(e)}), flush=True)
            except KeyboardInterrupt:
                print("接收到中断信号，服务器关闭", file=sys.stderr)
                break
            except Exception as e:
                print(f"服务器主循环异常: {e}", file=sys.stderr)
                print(json.dumps({'status': 'error', 'error_message': f'服务器异常: {str(e)}'}), flush=True)

def signal_handler(sig, frame):
    print("接收到信号，服务器关闭", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='神经网络启发式服务器 - 第二阶段专用')
    parser.add_argument('--model', default='second_stage_model.pth', help='第二阶段模型文件路径')
    parser.add_argument('--tokenizer', default='unified_tokenizer.pkl', help='分词器文件路径')
    args = parser.parse_args()
    
    server = NeuralHeuristicServer(args.model, args.tokenizer)
    server.start_server()