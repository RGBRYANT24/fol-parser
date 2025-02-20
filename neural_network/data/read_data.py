import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import deque
import torch.nn.functional as F
import re

#########################################
# 自定义分词器及基础数据集实现
#########################################

class CustomTokenizer:
    """完全重构的词汇表管理，解决ID冲突问题"""
    def __init__(self):
        # 固定特殊标记
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3
        }
        # 普通 token 从 4 开始编号
        self.vocab = {k: v for k, v in self.special_tokens.items()}
        self.additional_tokens = {}  # 存储动态添加的 token
        self.next_id = len(self.special_tokens)

    def add_tokens(self, tokens):
        """确保新增 token 的 ID 连续性"""
        for token in tokens:
            if token not in self.vocab and token not in self.additional_tokens:
                self.additional_tokens[token] = self.next_id
                self.next_id += 1
        # 合并词汇表
        self.vocab.update(self.additional_tokens)

    def convert_tokens_to_ids(self, tokens):
        return [
            self.special_tokens.get(token,
                self.additional_tokens.get(token, self.special_tokens['[UNK]'])
            ) for token in tokens
        ]

class SLIDataset(Dataset):
    """支持自定义 tokenizer 的数据集类（第一阶段）"""
    def __init__(self, file_path, max_seq_length=512, tokenizer_class=CustomTokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer_class()  # 动态指定分词器类型
        self.samples = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f).get('samples', [])
        
        self._build_vocab(raw_data)
        self._process_samples(raw_data)
    
    def _safe_get(self, data, key, default=None):
        """安全获取字典值"""
        return data.get(key, default) if data else default
    
    def _build_vocab(self, raw_data):
        all_tokens = set()
        for sample in raw_data:
            tree = self._safe_get(sample, 'state', {}).get('tree', {})
            tree_tokens = self._linearize_tree(tree)
            ops = self._safe_get(sample, 'available_ops', [])
            op_tokens = self._process_operations(ops)
            all_tokens.update(tree_tokens + op_tokens)
        self.tokenizer.add_tokens(all_tokens)
    
    def _linearize_tree(self, tree_data):
        """线性化树结构，增加空值检查"""
        if not tree_data:
            return []
        nodes = {n['id']: n for n in tree_data.get('nodes', [])}
        roots = [n for n in tree_data.get('nodes', []) if n.get('depth', -1)==0]
        if not roots:
            return []
        sequence = []
        stack = [(roots[0], 0)]
        while stack:
            node, depth = stack.pop()
            if not node:
                continue
            node_type = self._safe_get(node, 'type', 'UNKNOWN')
            literal = self._safe_get(node, 'literal', {})
            substitution = self._safe_get(node, 'substitution', {})
            children = self._safe_get(node, 'children', [])
            node_features = [
                f"[DEPTH_{depth}]",
                f"[TYPE_{node_type}]",
                f"[PRED_{literal.get('predicate', 'UNK')}]",
                f"[NEG_{int(literal.get('negated', False))}]"
            ]
            for arg in literal.get('arguments', []):
                if arg.startswith('VAR'):
                    node_features.append("[VAR]")
                elif arg.startswith('CONST'):
                    node_features.append("[CONST]")
                else:
                    node_features.append(f"[ARG_{arg}]")
            if substitution is not None:
                for src, tgt in substitution.items():
                    node_features += [f"[SUB_SRC_{src}]", f"[SUB_TGT_{tgt}]"]
            for child_id in reversed(children):
                child_node = nodes.get(child_id)
                if child_node:
                    stack.append((child_node, depth+1))
            sequence.extend(node_features)
        return sequence

    def _process_operations(self, operations):
        """处理 available_ops，修复操作处理中的空值问题"""
        op_tokens = []
        for op in operations:
            if not op:
                continue
            action = op.get('action', 'UNKNOWN')
            operand2 = op.get('operand2', {})
            op_tokens.append(f"[ACTION_{action}]")
            op_tokens.append(f"[OP_TYPE_{operand2.get('type', 'UNKNOWN')}]")
            lit_info = operand2.get('literal', {})
            if lit_info:
                op_tokens.append(f"[LIT_PRED_{lit_info.get('predicate', 'UNK')}]")
        return op_tokens

    def _process_samples(self, raw_data):
        for raw_sample in raw_data:
            # 对图结构进行线性化（GraphSLIDataset 中会覆盖此方法）
            tree_tokens = self._linearize_tree(raw_sample.get('state', {}).get('tree', {}))
            ops = self._process_operations(raw_sample.get('available_ops', []))
            # 如果有图信息，则在子类中会覆盖
            full_sequence = self.tokenizer.convert_tokens_to_ids(tree_tokens + ops)
            full_sequence = full_sequence[:self.max_seq_length]
            labels = None  # 第一阶段标签在此不作额外处理
            
            # 处理候选操作参数与连续标签（连续回归目标）
            candidate_params = []
            candidate_q_values = []  # 新增：存储每个候选的连续回归标签
            for op in raw_sample.get('available_ops', []):
                cand_tokens = self._linearize_candidate(op)
                if len(cand_tokens) > self.max_seq_length:
                    cand_tokens = cand_tokens[:self.max_seq_length]
                else:
                    cand_tokens += ["[PAD]"] * (self.max_seq_length - len(cand_tokens))
                candidate_ids = self.tokenizer.convert_tokens_to_ids(cand_tokens)
                candidate_params.append(candidate_ids)
                
                # 计算连续标签 q_value: q = reward - λ * op_depth
                reward = raw_sample.get('reward', 0.0)
                op_depth = op.get("depth", 0)
                lamda = 0.1  # 超参数，可根据情况调整
                q_value = reward - lamda * op_depth
                candidate_q_values.append(q_value)
            
            # 转换 candidate_params 和 candidate_q_values 为 tensor
            if candidate_params:
                candidate_params = torch.tensor(candidate_params, dtype=torch.long)
                candidate_q_values = torch.tensor(candidate_q_values, dtype=torch.float)
            else:
                # 如果没有候选参数，则创建空 tensor
                candidate_params = torch.empty((0, self.max_seq_length), dtype=torch.long)
                candidate_q_values = torch.empty((0,), dtype=torch.float)
            
            self.samples.append({
                'input_ids': full_sequence,
                'raw_data': raw_sample,
                'labels': labels,
                'candidate_param_ids': candidate_params,  # 候选参数 token id
                'candidate_q_values': candidate_q_values   # 候选参数对应的连续标签
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        pad_id = self.tokenizer.special_tokens['[PAD]']
        seq_len = len(sample['input_ids'])
        input_ids = sample['input_ids'] + [pad_id] * (self.max_seq_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'reward': torch.tensor(self._safe_get(sample['raw_data'], 'reward', 0.0), dtype=torch.float),
            'candidate_param_ids': sample['candidate_param_ids'],  # [num_candidates, max_seq_length]
            'candidate_q_values': sample['candidate_q_values']       # [num_candidates]
        }

#########################################
# 增强分词器与图结构及候选参数处理
#########################################

class EnhancedTokenizer(CustomTokenizer):
    """修正版增强分词器，增加图结构相关特殊标记"""
    def __init__(self):
        super().__init__()
        self.special_tokens.update({
            '[GRAPH_START]': 4,
            '[GRAPH_END]': 5,
            '[NODE]': 6,
            '[EDGE]': 7
        })
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
        self.next_id = max(self.vocab.values()) + 1

class GraphSLIDataset(SLIDataset):
    """
    扩展自 SLIDataset，整合图结构信息以及候选操作参数信息，
    用于第一阶段和第二阶段神经网络的数据输入。
    """
    ACTION_MAP = {
        "Extension": 0,
        "Factoring": 1,
        "Ancestry": 2,
    }
    def __init__(self, sli_file, graph_file, max_seq_length=768, max_param_seq_length=30):
        with open(graph_file, "r", encoding="utf-8") as f:
            self.graph_data = json.load(f)["graphs"][0]
        self.max_param_seq_length = max_param_seq_length
        super().__init__(sli_file, max_seq_length, tokenizer_class=EnhancedTokenizer)
        self._build_graph_vocab()
        self.max_seq_length = max_seq_length * 2

    def _build_graph_vocab(self):
        """独立构建图结构词汇表"""
        graph_tokens = set()
        for node in self.graph_data["nodes"]:
            graph_tokens.add(f"[NODE_{node['id']}]")
            graph_tokens.add(f"[ALIAS_{node['alias']}]")
        for edge in self.graph_data["edges"]:
            for lit in edge["literals"]:
                graph_tokens.add(f"[PRED_{lit['predicate']}]")
                for arg in lit["arguments"]:
                    graph_tokens.add(f"[ARG_{arg}]")
        self.tokenizer.add_tokens(graph_tokens)

    def _linearize_graph(self):
        """将图结构编码为 token 序列"""
        tokens = ["[GRAPH_START]"]
        for node in self.graph_data["nodes"]:
            tokens.extend([
                "[NODE]",
                f"[NODE_{node['id']}]",
                f"[ALIAS_{node['alias']}]"
            ])
        seen_edges = set()
        for edge in self.graph_data["edges"]:
            lit = edge["literals"][0]
            src, tgt = lit["arguments"]
            if (tgt, src) not in seen_edges:
                tokens.extend([
                    "[EDGE]",
                    f"[PRED_{lit['predicate']}]",
                    f"[ARG_{src}]",
                    f"[ARG_{tgt}]"
                ])
                seen_edges.add((src, tgt))
        tokens.append("[GRAPH_END]")
        return tokens

    def _linearize_candidate(self, op):
        """
        线性化单个候选操作参数，将其转换为 token 序列。
        格式示例：
        [OP_START] [ACTION_Extension] [DEPTH_2] [NODE1] [NODE_ID_1] [NODE_TYPE_B] 
        [NODE_DEPTH_1] [PRED_uncol] ... [OPERAND2_LITERAL] [PRED_uncol] [ARG_arg] ... [OP_END]
        """
        tokens = []
        tokens.append("[OP_START]")
        action = op.get("action", "UNK")
        tokens.append(f"[ACTION_{action}]")
        depth = op.get("depth", 0)
        tokens.append(f"[DEPTH_{depth}]")
        node1 = op.get("node1", {})
        tokens.append("[NODE1]")
        if node1:
            node_id = node1.get("id", -1)
            tokens.append(f"[NODE_ID_{node_id}]")
            node_type = node1.get("type", "UNK")
            tokens.append(f"[NODE_TYPE_{node_type}]")
            node_depth = node1.get("depth", 0)
            tokens.append(f"[NODE_DEPTH_{node_depth}]")
            literal = node1.get("literal", {})
            pred = literal.get("predicate", "UNK")
            tokens.append(f"[PRED_{pred}]")
            for arg in literal.get("arguments", []):
                tokens.append(f"[ARG_{arg}]")
            substitution = node1.get("substitution", {})
            if substitution:
                for k, v in substitution.items():
                    tokens.append(f"[SUB_{k}_{v}]")
        else:
            tokens.append("[NO_NODE1]")
        operand2 = op.get("operand2", {})
        op2_type = operand2.get("type", "").lower()
        if op2_type == "literal":
            tokens.append("[OPERAND2_LITERAL]")
            lit = operand2.get("literal", {})
            pred2 = lit.get("predicate", "UNK")
            tokens.append(f"[PRED_{pred2}]")
            for arg in lit.get("arguments", []):
                tokens.append(f"[ARG_{arg}]")
        elif op2_type == "node":
            tokens.append("[OPERAND2_NODE]")
            node_id2 = operand2.get("id", -1)
            tokens.append(f"[NODE_ID_{node_id2}]")
        else:
            tokens.append("[OPERAND2_UNK]")
        tokens.append("[OP_END]")
        return tokens

    def _process_samples(self, raw_data):
        graph_tokens = self._linearize_graph()
        graph_ids = self.tokenizer.convert_tokens_to_ids(graph_tokens)
        
        for raw_sample in raw_data:
            tree_tokens = self._linearize_tree(raw_sample.get('state', {}).get('tree', {}))
            op_tokens = self._process_operations(raw_sample.get('available_ops', []))
            # 全局输入：图 tokens + 树 tokens + 可用操作 tokens
            full_sequence = graph_ids + self.tokenizer.convert_tokens_to_ids(tree_tokens + op_tokens)
            full_sequence = full_sequence[:self.max_seq_length]
            labels = self._generate_action_labels(raw_sample)
            
            candidate_params = []
            candidate_q_values = []  # 新增：连续回归标签
            for op in raw_sample.get('available_ops', []):
                cand_tokens = self._linearize_candidate(op)
                if len(cand_tokens) > self.max_param_seq_length:
                    cand_tokens = cand_tokens[:self.max_param_seq_length]
                else:
                    cand_tokens += ["[PAD]"] * (self.max_param_seq_length - len(cand_tokens))
                candidate_ids = self.tokenizer.convert_tokens_to_ids(cand_tokens)
                candidate_params.append(candidate_ids)
                
                # 计算 Q 值：reward - λ * depth
                reward = raw_sample.get('reward', 0.0)
                op_depth = op.get("depth", 0)
                lamda = 0.1  # 超参数
                q_value = reward - lamda * op_depth
                candidate_q_values.append(q_value)
            
            candidate_params = torch.tensor(candidate_params, dtype=torch.long)
            candidate_q_values = torch.tensor(candidate_q_values, dtype=torch.float)
            
            self.samples.append({
                'input_ids': full_sequence,
                'raw_data': raw_sample,
                'labels': labels,
                'candidate_param_ids': candidate_params,    # [num_candidates, max_param_seq_length]
                'candidate_q_values': candidate_q_values      # [num_candidates]
            })

    def _generate_action_labels(self, raw_sample):
        """生成动作奖励标签（1x3）"""
        num_actions = len(self.ACTION_MAP)
        labels = [0.0] * num_actions
        selected_op = raw_sample.get('selected_op', {})
        selected_action = selected_op.get('action')
        reward = raw_sample.get('reward', 0.0)
        action_idx = self.ACTION_MAP.get(selected_action, -1)
        if 0 <= action_idx < num_actions:
            labels[action_idx] = reward
        return labels

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pad_id = self.tokenizer.special_tokens['[PAD]']
        seq_len = len(sample['input_ids'])
        global_input_ids = sample['input_ids'] + [pad_id] * (self.max_seq_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
        graph_mask = self._create_graph_mask(torch.tensor(global_input_ids, dtype=torch.long))
    
        return {
            'input_ids': torch.tensor(global_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'graph_mask': graph_mask,
            'labels': torch.tensor(sample['labels'], dtype=torch.float),
            'reward': torch.tensor(self._safe_get(sample['raw_data'], 'reward', 0.0), dtype=torch.float),
            'candidate_param_ids': sample['candidate_param_ids'],   # [num_candidates, max_param_seq_length]
            'candidate_q_values': sample['candidate_q_values']          # [num_candidates]
        }
    
    def _create_graph_mask(self, input_ids):
        """生成图结构注意力掩码，根据 [GRAPH_START] 和 [GRAPH_END] 标记"""
        mask = torch.zeros_like(input_ids)
        token_list = input_ids.tolist()
        try:
            start = token_list.index(self.tokenizer.vocab["[GRAPH_START]"])
            end = token_list.index(self.tokenizer.vocab["[GRAPH_END]"])
            mask[start:end+1] = 1
        except ValueError:
            pass
        return mask

#########################################
# collate_fn：组装一个 batch 时进行统一 padding
#########################################

def collate_fn(batch, tokenizer):
    """
    此 collate_fn 除了对全局 input_ids、attention_mask、graph_mask、labels、reward 进行 padding 外，
    还针对候选操作参数 candidate_param_ids 以及候选的连续标签 candidate_q_values 做额外的 padding，
    确保整个 batch 内每个样本的候选数量一致。
    """
    # 全局部分 padding：input_ids, attention_mask, graph_mask
    max_global_len = max(item['input_ids'].size(0) for item in batch)
    input_ids = torch.stack([
        F.pad(item['input_ids'], (0, max_global_len - item['input_ids'].size(0)), value=tokenizer.special_tokens['[PAD]'])
        for item in batch
    ])
    attention_mask = torch.stack([
        F.pad(item['attention_mask'], (0, max_global_len - item['attention_mask'].size(0)), value=0)
        for item in batch
    ])
    graph_mask = torch.stack([
        F.pad(item['graph_mask'], (0, max_global_len - item['graph_mask'].size(0)), value=0)
        for item in batch
    ])
    labels = torch.stack([item['labels'] for item in batch])
    reward = torch.stack([item['reward'] for item in batch])
    
    # 对候选参数 candidate_param_ids 进行 padding
    max_candidates = max(item['candidate_param_ids'].size(0) for item in batch)
    candidate_param_ids_list = []
    candidate_q_values_list = []  # 针对连续标签
    for item in batch:
        cand_tensor = item['candidate_param_ids']  # [num_candidates, max_param_seq_length]
        num_candidates = cand_tensor.size(0)
        if num_candidates < max_candidates:
            pad_tensor = torch.full((max_candidates - num_candidates, cand_tensor.size(1)),
                                      tokenizer.special_tokens['[PAD]'],
                                      dtype=cand_tensor.dtype)
            cand_tensor = torch.cat([cand_tensor, pad_tensor], dim=0)
        candidate_param_ids_list.append(cand_tensor)
        
        # 对 candidate_q_values 进行 padding
        q_tensor = item['candidate_q_values']  # [num_candidates]
        if q_tensor.size(0) < max_candidates:
            pad_q = torch.zeros(max_candidates - q_tensor.size(0), dtype=q_tensor.dtype)
            q_tensor = torch.cat([q_tensor, pad_q], dim=0)
        candidate_q_values_list.append(q_tensor)
    
    candidate_param_ids = torch.stack(candidate_param_ids_list, dim=0)  # [B, max_candidates, max_param_seq_length]
    candidate_q_values = torch.stack(candidate_q_values_list, dim=0)        # [B, max_candidates]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'graph_mask': graph_mask,
        'labels': labels,
        'reward': reward,
        'candidate_param_ids': candidate_param_ids,
        'candidate_q_values': candidate_q_values
    }

#########################################
# 测试代码
#########################################

if __name__ == "__main__":
    # 假设 training_data.json 中包含提供的样例数据，
    # k3_graph.json 为图数据文件
    sli_file = "../data/training_data.json"  # 要求文件中有 "samples" 字段
    graph_file = "../data/k3_graph.json"
    dataset = GraphSLIDataset(sli_file, graph_file, max_seq_length=512, max_param_seq_length=30)
    
    print("样本总数：", len(dataset))
    
    # 查看单个样本的处理结果
    sample0 = dataset[0]
    print("单个样本全局 input_ids (长度 {}):".format(len(sample0['input_ids'])))
    print(sample0['input_ids'])
    print("单个样本 attention_mask:")
    print(sample0['attention_mask'])
    print("单个样本 graph_mask:")
    print(sample0['graph_mask'])
    print("单个样本 labels:")
    print(sample0['labels'])
    print("单个样本候选参数 candidate_param_ids, 形状:", sample0['candidate_param_ids'].shape)
    print(sample0['candidate_param_ids'])
    print("单个候选参数连续标签 candidate_q_values, 形状:", sample0['candidate_q_values'].shape)
    print(sample0['candidate_q_values'])
    
    # 使用 DataLoader 以及 collate_fn 构造 batch，打印 batch 信息
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False,
                            collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer))
    batch_sample = next(iter(dataloader))
    print("Batch 全局 input_ids shape:", batch_sample['input_ids'].shape)
    print("Batch attention_mask shape:", batch_sample['attention_mask'].shape)
    print("Batch graph_mask shape:", batch_sample['graph_mask'].shape)
    print("Batch labels shape:", batch_sample['labels'].shape)
    print("Batch candidate_param_ids shape:", batch_sample['candidate_param_ids'].shape)
    print("Batch candidate_q_values shape:", batch_sample['candidate_q_values'].shape)