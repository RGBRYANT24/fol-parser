import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os

#########################################
# 自定义分词器及基础数据集实现
#########################################

class CustomTokenizer:
    """
    自定义词汇表管理，所有 token 均存储在 self.vocab 中，
    支持固定 token 与动态扩展 token。
    """
    def __init__(self):
        # 固定特殊标记：编号从 0 开始
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
        }
        # 所有 token 均存储在 vocab 中（初始为特殊标记）
        self.vocab = dict(self.special_tokens)
        self.additional_tokens = {}  # 用于记录后续新增的 token
        self.next_id = len(self.vocab)
    
    def add_tokens(self, tokens):
        """把 token 集合加入词汇表（若已存在则跳过）"""
        for token in tokens:
            if token not in self.vocab:
                self.additional_tokens[token] = self.next_id
                self.vocab[token] = self.next_id
                self.next_id += 1

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab.get("[UNK]")) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        id_to_token = {v: k for k, v in self.vocab.items()}
        return [id_to_token.get(i, "[UNK]") for i in ids]

class SLIDataset(Dataset):
    """
    基础数据集类：用于处理状态树（sliTree）和候选操作（op）的线性化
    """
    def __init__(self, file_path, max_seq_length=512, tokenizer_class=CustomTokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer_class()  # 动态指定分词器类型
        self.samples = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f).get('samples', [])
        
        self._build_vocab(raw_data)
        self._process_samples(raw_data)
    
    def _safe_get(self, data, key, default=None):
        if data is None:
            return default
        value = data.get(key, default)
        return value if value is not None else default
    
    def _build_vocab(self, raw_data):
        all_tokens = set()
        for sample in raw_data:
            tree = self._safe_get(sample, 'state', {}).get('tree', {})
            tree_tokens = self._linearize_tree(tree)
            ops = self._safe_get(sample, 'available_ops', [])
            op_tokens = self._process_operations(ops)
            candidate_tokens = []
            for op in ops:
                candidate_tokens += self._linearize_candidate(op)
            all_tokens.update(tree_tokens + op_tokens + candidate_tokens)
        self.tokenizer.add_tokens(all_tokens)
    
    def _linearize_tree(self, tree_data):
        """
        线性化状态树：
          - 每个节点输出 [TYPE_x] 和 [PRED_y]；
          - 若 literal 中 negated 为 True，则输出固定 token [NEG]；
          - 参数部分：变量统一输出 [VAR]；常量直接输出具体 token（例如 [CONST0]）
        """
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
            node_type = self._safe_get(node, 'type', 'UNKNOWN')
            literal = self._safe_get(node, 'literal', {}) or {}
            substitution = self._safe_get(node, 'substitution', {}) or {}
            children = self._safe_get(node, 'children', [])
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
        """
        线性化 available_ops 的一部分（例如 action、op_type）
        """
        op_tokens = []
        for op in operations:
            if not op:
                continue
            action = op.get('action', 'UNKNOWN')
            operand2 = op.get('operand2', {})
            op_tokens.append(f"[ACTION_{action}]")
            op_tokens.append(f"[OP_TYPE_{operand2.get('type', 'UNKNOWN')}]")
        return op_tokens

    def _process_samples(self, raw_data):
        for raw_sample in raw_data:
            tree_tokens = self._linearize_tree(self._safe_get(raw_sample.get('state', {}), 'tree', {}))
            ops = self._process_operations(raw_sample.get('available_ops', []))
            # 全局序列：图部分 + [SEP] + 状态树部分 + [TREE_OP_SEP]（分隔符）+ 操作参数部分
            full_sequence = self.tokenizer.convert_tokens_to_ids(tree_tokens)
            sep_between = self.tokenizer.convert_tokens_to_ids(["[TREE_OP_SEP]"])
            op_tokens = self._process_operations(raw_sample.get('available_ops', []))
            full_sequence = full_sequence + sep_between + self.tokenizer.convert_tokens_to_ids(op_tokens)
            full_sequence = full_sequence[:self.max_seq_length]
            labels = None  # 此处标签暂不处理
            
            candidate_params = []
            candidate_q_values = []
            for op in raw_sample.get('available_ops', []):
                cand_tokens = self._linearize_candidate(op)
                if len(cand_tokens) > self.max_seq_length:
                    cand_tokens = cand_tokens[:self.max_seq_length]
                else:
                    cand_tokens += ["[PAD]"] * (self.max_seq_length - len(cand_tokens))
                candidate_ids = self.tokenizer.convert_tokens_to_ids(cand_tokens)
                candidate_params.append(candidate_ids)
                
                if "reward" in op:
                    q_value = op.get("reward", 0.0)
                else:
                    reward = raw_sample.get('reward', 0.0)
                    lamda = 0.1
                    q_value = reward - lamda * op.get("depth", 0)
                candidate_q_values.append(q_value)
            
            if candidate_params:
                candidate_params = torch.tensor(candidate_params, dtype=torch.long)
                candidate_q_values = torch.tensor(candidate_q_values, dtype=torch.float)
            else:
                candidate_params = torch.empty((0, self.max_seq_length), dtype=torch.long)
                candidate_q_values = torch.empty((0,), dtype=torch.float)
            
            self.samples.append({
                'input_ids': full_sequence,
                'raw_data': raw_sample,
                'labels': labels,
                'candidate_param_ids': candidate_params,
                'candidate_q_values': candidate_q_values
            })
    
    def _linearize_candidate(self, op):
        """
        线性化候选操作参数，格式为：
          [OP_START] [ACTION_xxx] [NODE1] ... [OP_END]
        对于 node1 部分，统一使用 [TYPE_x] 表示节点类型；
        literal 中若 negated 为 True则统一输出 [NEG]；
        kb_clause 部分也仅输出一次 [NEG]。
        """
        tokens = []
        tokens.append("[OP_START]")
        action = op.get("action", "UNK")
        tokens.append(f"[ACTION_{action}]")
        node1 = op.get("node1", {})
        tokens.append("[NODE1]")
        if node1:
            node_type = node1.get("type", "UNK")
            tokens.append(f"[TYPE_{node_type}]")
            literal = node1.get("literal", {}) or {}
            pred = literal.get("predicate", "UNK")
            tokens.append(f"[PRED_{pred}]")
            if literal.get('negated', False):
                tokens.append("[NEG]")
            for arg in literal.get("arguments", []):
                if arg.startswith("VAR"):
                    tokens.append("[VAR]")
                elif arg.startswith("CONST"):
                    tokens.append(f"[{arg}]")
                else:
                    tokens.append(f"[ARG_{arg}]")
        else:
            tokens.append("[NO_NODE1]")
        operand2 = op.get("operand2", {})
        op2_type = operand2.get("type", "").lower()
        if op2_type == "literal":
            tokens.append("[OPERAND2_LITERAL]")
            lit = operand2.get("literal", {}) or {}
            pred2 = lit.get("predicate", "UNK")
            tokens.append(f"[PRED_{pred2}]")
            if lit.get('negated', False):
                tokens.append("[NEG]")
            for arg in lit.get("arguments", []):
                if arg.startswith("VAR"):
                    tokens.append("[VAR]")
                elif arg.startswith("CONST"):
                    tokens.append(f"[{arg}]")
                else:
                    tokens.append(f"[ARG_{arg}]")
        elif op2_type == "node":
            tokens.append("[OPERAND2_NODE]")
        else:
            tokens.append("[OPERAND2_UNK]")
        tokens.append("[OP_END]")
    
        if "kb_clause" in op:
            for clause in op["kb_clause"]:
                for arg in clause.get("arguments", []):
                    if arg.startswith("VAR"):
                        tokens.append("[VAR]")
                    elif arg.startswith("CONST"):
                        tokens.append(f"[{arg}]")
                    else:
                        tokens.append(f"[KB_ARG_{arg}]")
                tokens.append("[NEG]")
        return tokens

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        pad_id = self.tokenizer.vocab['[PAD]']
        seq_len = len(sample['input_ids'])
        global_input_ids = sample['input_ids'] + [pad_id] * (self.max_seq_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
        graph_mask = self._create_graph_mask(torch.tensor(global_input_ids, dtype=torch.long))
    
        raw_reward = self._safe_get(sample['raw_data'], 'reward', 0.0)
        if isinstance(raw_reward, dict):
            raw_reward = raw_reward.get("expected_by_type", {}).get("EXTENSION", 0.0)
    
        global_reward = self._safe_get(sample['raw_data'], 'global_reward', None)
    
        return {
            'input_ids': torch.tensor(global_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'graph_mask': graph_mask,
            'labels': torch.tensor(sample['labels'], dtype=torch.float) if sample['labels'] is not None else None,
            'reward': torch.tensor(raw_reward, dtype=torch.float),
            'candidate_param_ids': sample['candidate_param_ids'],
            'candidate_q_values': sample['candidate_q_values'],
            'global_reward': global_reward
        }
    
    def _create_graph_mask(self, input_ids):
        """根据 [GRAPH_START] 与 [GRAPH_END] 的位置生成图结构注意力 mask"""
        mask = torch.zeros_like(input_ids)
        token_list = input_ids.tolist()
        try:
            start = token_list.index(self.tokenizer.vocab["[GRAPH_START]"])
            end = token_list.index(self.tokenizer.vocab["[GRAPH_END]"])
            mask[start:end+1] = 1
        except ValueError:
            pass
        return mask

class EnhancedTokenizer(CustomTokenizer):
    """
    增强版分词器：在特殊标记基础上增加图结构相关 token，
    同时预置固定常量（[CONST0]~[CONST9]）、固定变量（[VAR0]~[VAR9]）、
    固定谓词及其它固定 token，并按照要求赋予具体 index。
    """
    def __init__(self):
        super().__init__()
        # 预置特殊标记（保留已有 [PAD],[UNK],[CLS],[SEP]）
        fixed_specials = {
            '[GRAPH_START]': 4,
            '[GRAPH_END]': 5,
            '[NODE]': 6,
            '[EDGE]': 7,
        }
        for token, idx in fixed_specials.items():
            self.special_tokens[token] = idx
            self.vocab[token] = idx
        
        # 固定常量 token，CONST0-9 固定为 8-17
        for i in range(10):
            token = f"[CONST{i}]"
            self.vocab[token] = 8 + i
        
        # 固定变量 token：[VAR0] ~ [VAR9]，固定为 18-27
        for i in range(10):
            token = f"[VAR{i}]"
            self.vocab[token] = 18 + i
        
        # 固定谓词 token
        fixed_predicates = {
            "[PRED_E]": 28,
            "[PRED_uncol]": 29,
        }
        for token, idx in fixed_predicates.items():
            self.vocab[token] = idx
        
        # 固定其它 token（按照类别顺序排列）
        fixed_tokens = {
            # 否定相关
            "[NEG]": 30,
            
            # 节点相关
            "[NODE1]": 31,
            
            # 谓词相关
            "[PRED_]": 32,
            
            # 类型相关
            "[TYPE_A]": 33,
            "[TYPE_B]": 34,
            
            # 操作类型相关
            "[OP_START]": 35,
            "[OP_END]": 36,
            "[OP_TYPE_literal]": 37,
            "[OP_TYPE_node]": 38,
            "[TREE_OP_SEP]": 39,
            
            # 操作数相关
            "[OPERAND2_LITERAL]": 40,
            "[OPERAND2_NODE]": 41,
            
            # 动作相关
            "[ACTION_ANCESTRY]": 42,
            "[ACTION_FACTORING]": 43,
            "[ACTION_EXTENSION]": 44,
            "[ACTION_TRUNCATE]": 45
        }
        for token, idx in fixed_tokens.items():
            self.vocab[token] = idx
        
        # 设置下一个可用的ID
        self.next_id = 46
        
        # 对于搜索路径中可能出现的动态谓词 "R" 和 "G"，不固定其 index
        optional_predicates = ["G", "R"]
        for pred in optional_predicates:
            token = f"[PRED_{pred}]"
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.next_id += 1

class GraphSLIDataset(SLIDataset):
    """
    在 SLIDataset 基础上整合图结构与搜索路径信息，
    数据文件要求统一格式：顶层包含 "graph" 与 "search_path" 两个字段。
    
    图的线性化：每条边生成 3 个 token，顺序为：
         [PRED_{predicate}] [second node] [first node]
    最终图序列格式：
         [GRAPH_START] ...边 token... [GRAPH_END]
    """
    ACTION_MAP = {
        "Extension": 0,
        "Factoring": 1,
        "Ancestry": 2,
    }
    def __init__(self, unified_file, max_seq_length=768, max_param_seq_length=30):
        with open(unified_file, "r", encoding="utf-8") as f:
            unified_data = json.load(f)
        self.graph_data = unified_data.get("graph", {})
        self.max_param_seq_length = max_param_seq_length
        
        tokenizer = EnhancedTokenizer()
        self.tokenizer = tokenizer
        self._build_graph_vocab()
        
        raw_data = unified_data.get("search_path", [])
        # 为防止图与状态树过长，设定全局序列长度为 max_seq_length 的两倍
        self.max_seq_length = max_seq_length * 2
        self.samples = []
        self._build_vocab(raw_data)
        self._process_samples(raw_data)

    def _build_graph_vocab(self):
        """
        构建图结构词汇表：
          遍历每条边的 literal，生成三个 token：
            [PRED_{predicate}], [second node], [first node]
        """
        graph_tokens = set()
        for edge in self.graph_data.get("edges", []):
            for lit in edge.get("literals", []):
                args = lit.get("arguments", [])
                if len(args) >= 2:
                    predicate = lit.get("predicate", "UNK")
                    token_pred = f"[PRED_{predicate}]"
                    token_arg0 = f"[{args[0]}]"  # 第二个节点
                    token_arg1 = f"[{args[1]}]"  # 第一个节点
                    graph_tokens.update([token_pred, token_arg0, token_arg1])
        self.tokenizer.add_tokens(graph_tokens)

    def _linearize_graph(self):
        """
        将图结构线性化为 token 序列：
        每条边生成 3 个 token：
            [PRED_{predicate}] [second node] [first node]
        最终格式： [GRAPH_START] ... [GRAPH_END]
        """
        tokens = ["[GRAPH_START]"]
        for edge in self.graph_data.get("edges", []):
            for lit in edge.get("literals", []):
                args = lit.get("arguments", [])
                if len(args) >= 2:
                    predicate = lit.get("predicate", "UNK")
                    # Skip non-E predicates
                    if predicate != "E":
                        continue
                    
                    token_pred = f"[PRED_{predicate}]"
                    # Create node tokens with proper formatting
                    token_arg1 = f"[{args[1]}]"  # Second node
                    token_arg0 = f"[{args[0]}]"  # First node
                    # print('_linearize_graph Get Token', token_pred, token_arg0, token_arg1)
                    
                    # Ensure we're adding tokens in the right order
                    tokens.append(token_pred)
                    tokens.append(token_arg0)
                    tokens.append(token_arg1)
        tokens.append("[GRAPH_END]")
        # print('_linearize_graph Final Tokens')
        # print(tokens)
        return tokens

    def _process_samples(self, raw_data):
        graph_tokens = self._linearize_graph()
        # print('_process_samples Get graph_tokens')
        # print(graph_tokens)
        graph_ids = self.tokenizer.convert_tokens_to_ids(graph_tokens) # problem 
        # print('_process_samples Get graph_ids')
        # print(graph_ids)
        # print('_process_samples From graph_ids to Tokens')
        # print(self.tokenizer.convert_ids_to_tokens(graph_ids))
        sep_id = self.tokenizer.vocab["[SEP]"]
        
        for raw_sample in raw_data:
            tree_tokens = self._linearize_tree(raw_sample.get('state', {}).get('tree', {}))
            # 在状态树和操作参数之间增加新分隔符 [TREE_OP_SEP]
            op_tokens = self._process_operations(raw_sample.get('available_ops', []))
            full_sequence = graph_ids + [sep_id] \
                           + self.tokenizer.convert_tokens_to_ids(tree_tokens) \
                           + self.tokenizer.convert_tokens_to_ids(["[TREE_OP_SEP]"]) \
                           + self.tokenizer.convert_tokens_to_ids(op_tokens)
            full_sequence = full_sequence[:self.max_seq_length]
            labels = self._generate_action_labels(raw_sample)
            
            candidate_params = []
            candidate_q_values = []
            for op in raw_sample.get('available_ops', []):
                cand_tokens = self._linearize_candidate(op)
                if len(cand_tokens) > self.max_param_seq_length:
                    cand_tokens = cand_tokens[:self.max_param_seq_length]
                else:
                    cand_tokens += ["[PAD]"] * (self.max_param_seq_length - len(cand_tokens))
                candidate_ids = self.tokenizer.convert_tokens_to_ids(cand_tokens)
                candidate_params.append(candidate_ids)
                
                if "reward" in op:
                    q_value = op.get("reward", 0.0)
                else:
                    reward = raw_sample.get('reward', 0.0)
                    lamda = 0.1
                    q_value = reward - lamda * op.get("depth", 0)
                candidate_q_values.append(q_value)
            
            candidate_params = torch.tensor(candidate_params, dtype=torch.long)
            candidate_q_values = torch.tensor(candidate_q_values, dtype=torch.float)
            
            self.samples.append({
                'input_ids': full_sequence,
                'raw_data': raw_sample,
                'labels': labels,
                'candidate_param_ids': candidate_params,
                'candidate_q_values': candidate_q_values
            })

    def _generate_action_labels(self, raw_sample):
        op_seq = ["EXTENSION", "FACTORING", "ANCESTRY"]
        global_reward = raw_sample.get("global_reward", {})
        if isinstance(global_reward, dict) and "expected_by_type" in global_reward:
            exp_reward = global_reward["expected_by_type"]
            return [exp_reward.get(op, 0.0) for op in op_seq]
        else:
            return [0.0 for _ in op_seq]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pad_id = self.tokenizer.vocab['[PAD]']
        seq_len = len(sample['input_ids'])
        global_input_ids = sample['input_ids'] + [pad_id] * (self.max_seq_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_seq_length - seq_len)

            # 修改：找到[TREE_OP_SEP]位置并将其之后的注意力掩码设为0
        token_list = sample['input_ids']
        try:
            tree_op_sep_idx = token_list.index(self.tokenizer.vocab["[TREE_OP_SEP]"])
            for i in range(tree_op_sep_idx, len(attention_mask)):
                attention_mask[i] = 0
        except ValueError:
            # 如果没有找到[TREE_OP_SEP]，则不需修改
            pass

        graph_mask = self._create_graph_mask(torch.tensor(global_input_ids, dtype=torch.long))
    
        raw_reward = self._safe_get(sample['raw_data'], 'reward', 0.0)
        if isinstance(raw_reward, dict):
            raw_reward = raw_reward.get("expected_by_type", {}).get("EXTENSION", 0.0)
    
        global_reward = self._safe_get(sample['raw_data'], 'global_reward', None)
    
        return {
            'input_ids': torch.tensor(global_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'graph_mask': graph_mask,
            'labels': torch.tensor(sample['labels'], dtype=torch.float) if sample['labels'] is not None else None,
            'reward': torch.tensor(raw_reward, dtype=torch.float),
            'candidate_param_ids': sample['candidate_param_ids'],
            'candidate_q_values': sample['candidate_q_values'],
            'global_reward': global_reward
        }
    
    def _create_graph_mask(self, input_ids):
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
    labels_list = []
    for item in batch:
        if item.get("labels") is not None:
            labels_list.append(item["labels"])
        elif isinstance(item.get("global_reward"), dict) and "expected_by_type" in item["global_reward"]:
            op_seq = ["EXTENSION", "FACTORING", "ANCESTRY"]
            exp_reward = item["global_reward"]["expected_by_type"]
            label = torch.tensor([exp_reward.get(op, 0.0) for op in op_seq], dtype=torch.float)
            labels_list.append(label)
        else:
            labels_list.append(torch.zeros(3, dtype=torch.float))
    labels = torch.stack(labels_list)
    
    max_global_len = max(item["input_ids"].size(0) for item in batch)
    
    input_ids = torch.stack([
        F.pad(item["input_ids"],
              (0, max_global_len - item["input_ids"].size(0)),
              value=tokenizer.vocab["[PAD]"])
        for item in batch
    ])
    
    attention_mask = torch.stack([
        F.pad(item["attention_mask"],
              (0, max_global_len - item["attention_mask"].size(0)),
              value=0)
        for item in batch
    ])
    
    graph_mask = torch.stack([
        F.pad(item["graph_mask"],
              (0, max_global_len - item["graph_mask"].size(0)),
              value=0)
        for item in batch
    ])

    max_candidates = max(item["candidate_param_ids"].size(0) for item in batch)
    candidate_param_ids_list = []
    candidate_q_values_list = []
    
    for item in batch:
        cand_tensor = item["candidate_param_ids"]
        num_candidates = cand_tensor.size(0)
        if num_candidates < max_candidates:
            pad_tensor = torch.full(
                (max_candidates - num_candidates, cand_tensor.size(1)),
                tokenizer.vocab["[PAD]"],
                dtype=cand_tensor.dtype
            )
            cand_tensor = torch.cat([cand_tensor, pad_tensor], dim=0)
        candidate_param_ids_list.append(cand_tensor)
        
        q_tensor = item["candidate_q_values"]
        if q_tensor.size(0) < max_candidates:
            pad_q = torch.zeros(max_candidates - q_tensor.size(0), dtype=q_tensor.dtype)
            q_tensor = torch.cat([q_tensor, pad_q], dim=0)
        candidate_q_values_list.append(q_tensor)
    
    candidate_param_ids = torch.stack(candidate_param_ids_list, dim=0)
    candidate_q_values = torch.stack(candidate_q_values_list, dim=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "graph_mask": graph_mask,
        "labels": labels,
        "global_reward": None,
        "candidate_param_ids": candidate_param_ids,
        "candidate_q_values": candidate_q_values
    }

#########################################
# 测试代码
#########################################

if __name__ == "__main__":
    # 假设文件 "training_data_0.json" 存在，
    # 文件内容需包含 "graph" 与 "search_path" 两个字段，
    # 示例格式：
    # {
    #   "graph": { ... },
    #   "search_path": [ { ... }, ... ]
    # }
    unified_file = "/home/jiangguifei01/aiderun/fol-parser/fol-parser/data/training_data_0_success.json"  

    dataset = GraphSLIDataset(unified_file, max_seq_length=512, max_param_seq_length=30)
    
    print("样本总数：", len(dataset))
    
    # 查看第 3 个样本的处理结果
    sample0 = dataset[5]
    print("单个样本全局 input_ids (长度 {}):".format(len(sample0['input_ids'])))
    print(sample0['input_ids'])
    tokens = dataset.tokenizer.convert_ids_to_tokens(sample0['input_ids'].tolist())
    print("对应的 token 序列：")
    print(tokens)
    print("单个样本 attention_mask:")
    print(sample0['attention_mask'])
    print("单个样本 graph_mask:")
    print(sample0['graph_mask'])
    print("单个样本 labels:")
    print(sample0['labels'])
    print("单个样本候选参数 candidate_param_ids, 形状:", sample0['candidate_param_ids'].shape)
    print(sample0['candidate_param_ids'])
    num_candidates = sample0['candidate_param_ids'].shape[0]
    for i in range(num_candidates):
        candidate_ids = sample0['candidate_param_ids'][i].tolist()
        candidate_tokens = dataset.tokenizer.convert_ids_to_tokens(candidate_ids)
        print(f"候选参数 {i} 对应的 token 序列:")
        print(candidate_tokens)
    print("单个候选参数连续标签 candidate_q_values, 形状:", sample0['candidate_q_values'].shape)
    print(sample0['candidate_q_values'])
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False,
                              collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer))
    batch_sample = next(iter(dataloader))
    print("Batch 全局 input_ids shape:", batch_sample['input_ids'].shape)
    print("Batch attention_mask shape:", batch_sample['attention_mask'].shape)
    print("Batch graph_mask shape:", batch_sample['graph_mask'].shape)
    print("Batch labels shape:", batch_sample['labels'].shape)
    print("Batch candidate_param_ids shape:", batch_sample['candidate_param_ids'].shape)
    print("Batch candidate_q_values shape:", batch_sample['candidate_q_values'].shape)

    print("完整词汇表映射：")
    for token, idx in sorted(dataset.tokenizer.vocab.items(), key=lambda x: x[1]):
        print(f"{token}: {idx}")