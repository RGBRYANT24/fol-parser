import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from collections import deque

class CustomTokenizer:
    """修复substitution为空的版本"""
    def __init__(self):
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3
        }
        self.vocab = defaultdict(lambda: len(self.vocab) + len(self.special_tokens))
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.special_tokens['[UNK]']) for token in tokens]

class SLIDataset(Dataset):
    """修复空值问题的数据集类"""
    def __init__(self, file_path, max_seq_length=512):
        self.max_seq_length = max_seq_length
        self.tokenizer = CustomTokenizer()
        self.samples = []
        
        with open(file_path) as f:
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
        """添加空值检查的树线性化方法"""
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
                
            # 安全获取各字段
            node_type = self._safe_get(node, 'type', 'UNKNOWN')
            literal = self._safe_get(node, 'literal', {})
            substitution = self._safe_get(node, 'substitution', {})
            children = self._safe_get(node, 'children', [])
            
            # 节点特征
            node_features = [
                f"[DEPTH_{depth}]",
                f"[TYPE_{node_type}]",
                f"[PRED_{literal.get('predicate', 'UNK')}]",
                f"[NEG_{int(literal.get('negated', False))}]"
            ]
            
            # 参数处理
            for arg in literal.get('arguments', []):
                if arg.startswith('VAR'):
                    node_features.append("[VAR]")
                elif arg.startswith('CONST'):
                    node_features.append("[CONST]")
                else:
                    node_features.append(f"[ARG_{arg}]")
            
            # 替换关系（修复空值问题）
            if substitution is not None:
                for src, tgt in substitution.items():
                    node_features += [f"[SUB_SRC_{src}]", f"[SUB_TGT_{tgt}]"]
            
            # 子节点处理
            for child_id in reversed(children):
                child_node = nodes.get(child_id)
                if child_node:
                    stack.append((child_node, depth+1))
            
            sequence.extend(node_features)
        
        return sequence

    def _process_operations(self, operations):
        """修复操作处理中的空值"""
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
        for sample in raw_data:
            tree = self._safe_get(sample, 'state', {}).get('tree', {})
            tree_tokens = self._linearize_tree(tree)
            
            ops = self._safe_get(sample, 'available_ops', [])
            op_tokens = self._process_operations(ops)
            
            full_sequence = tree_tokens + op_tokens
            token_ids = self.tokenizer.convert_tokens_to_ids(full_sequence)
            token_ids = token_ids[:self.max_seq_length]
            
            self.samples.append({
                'input_ids': token_ids,
                'raw_data': sample
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pad_id = self.tokenizer.special_tokens['[PAD]']
        seq_len = len(sample['input_ids'])
        
        input_ids = sample['input_ids'] + [pad_id] * (self.max_seq_length - seq_len)
        attention_mask = [1]*seq_len + [0]*(self.max_seq_length - seq_len)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'reward': torch.tensor(self._safe_get(sample['raw_data'], 'reward', 0.0), dtype=torch.float)
        }

# 使用示例
if __name__ == "__main__":
    dataset = SLIDataset("../data/training_data.json")
    print(f"数据集大小: {len(dataset)}")
    print(f"第一个样本的输入ID: {dataset[0]['input_ids']}")