#!/usr/bin/env python3
import os
import sys
import json
import pickle
import torch
import numpy as np
import signal
import argparse
from models.first_stage_model import GlobalEncoder, FirstStageModel

class NeuralHeuristicServer:
    def __init__(self, model_path, tokenizer_path):
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载分词器
        print(f"加载分词器: {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # 加载模型
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从保存的配置中恢复模型参数
        vocab_size = checkpoint['vocab_size']
        d_model = checkpoint['d_model']
        nhead = checkpoint['nhead']
        num_layers = checkpoint['num_layers']
        
        print(f"模型配置: vocab_size={vocab_size}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        
        # 创建模型实例
        global_encoder = GlobalEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
        self.model = FirstStageModel(global_encoder, d_model=d_model, num_actions=3)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("模型已加载并设置为评估模式")
    
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
    
    def process_request(self, request_data):
        """处理来自C++的请求数据"""
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
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # 准备模型输入
            max_len = 1024  # 设定一个最大长度
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            
            # 创建注意力掩码
            attention_mask = [1] * len(input_ids)
            
            # 确保树和操作分隔符之后的部分不被关注
            try:
                tree_op_sep_idx = input_ids.index(self.tokenizer.vocab["[TREE_OP_SEP]"])
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
            try:
                start = token_list.index(self.tokenizer.vocab["[GRAPH_START]"])
                end = token_list.index(self.tokenizer.vocab["[GRAPH_END]"])
                graph_attention_mask[0, start:end+1] = 1
            except ValueError:
                pass
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(
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
    parser.add_argument('--model', default='first_stage_model.pth', help='模型文件路径')
    parser.add_argument('--tokenizer', default='unified_tokenizer.pkl', help='分词器文件路径')
    args = parser.parse_args()
    
    server = NeuralHeuristicServer(args.model, args.tokenizer)
    server.start_server()