import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from collections import deque

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
        # 普通token从4开始编号
        self.vocab = {k: v for k, v in self.special_tokens.items()}
        self.additional_tokens = {}  # 存储动态添加的token
        self.next_id = len(self.special_tokens)

    def add_tokens(self, tokens):
        """确保新增token的ID连续性"""
        for token in tokens:
            if token not in self.vocab and token not in self.additional_tokens:
                self.additional_tokens[token] = self.next_id
                self.next_id += 1

        # 合并词汇表（注意：实际使用时需通过get方法访问）
        self.vocab.update(self.additional_tokens)

    def convert_tokens_to_ids(self, tokens):
        return [
            self.special_tokens.get(token, 
                self.additional_tokens.get(token, self.special_tokens['[UNK]'])
            ) for token in tokens
        ]

class SLIDataset(Dataset):
    """需要修改构造方法支持自定义tokenizer"""
    def __init__(self, file_path, max_seq_length=512, tokenizer_class=CustomTokenizer):  # 新增参数
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer_class()  # 支持动态指定分词器类型
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
        # 在GraphSLIDataset的_process_samples方法中添加：
        print("Graph Token IDs:", graph_ids)
        print("Special Token Mappings:", 
            {k: v for k, v in self.tokenizer.vocab.items() 
            if k in ['[GRAPH_START]', '[GRAPH_END]']})

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

class EnhancedTokenizer(CustomTokenizer):
    """修正版增强分词器"""
    def __init__(self):
        # 先初始化父类
        super().__init__()
        
        # 强制指定图结构标记ID（覆盖父类可能存在的冲突）
        self.special_tokens.update({
            '[GRAPH_START]': 4,
            '[GRAPH_END]': 5,
            '[NODE]': 6,
            '[EDGE]': 7
        })
        
        # 直接更新词汇表
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx  # 确保覆盖可能存在的旧值
        
        # 重置普通token起始ID
        self.next_id = max(self.vocab.values()) + 1

class GraphSLIDataset(SLIDataset):
    ACTION_MAP = {
        "Extension": 0,
        "Factoring": 1,
        "Ancestry": 2,
    }
    """整合图结构和SLI树的数据集"""
    def __init__(self, sli_file, graph_file, max_seq_length=768):
        # 初始化图数据
        with open(graph_file) as f:
            self.graph_data = json.load(f)["graphs"][0]
        
        # 显式指定使用EnhancedTokenizer
        super().__init__(sli_file, max_seq_length, tokenizer_class=EnhancedTokenizer)
        
        # 必须在父类初始化完成后构建图词汇表
        self._build_graph_vocab()
        self.max_seq_length = max_seq_length * 2

    def _build_graph_vocab(self):
        """独立构建图结构词汇表"""
        graph_tokens = set()
        
        # 处理节点
        for node in self.graph_data["nodes"]:
            graph_tokens.add(f"[NODE_{node['id']}]")
            graph_tokens.add(f"[ALIAS_{node['alias']}]")
        
        # 处理边
        for edge in self.graph_data["edges"]:
            for lit in edge["literals"]:
                graph_tokens.add(f"[PRED_{lit['predicate']}]")
                graph_tokens.update([f"[ARG_{arg}]" for arg in lit["arguments"]])
        
        # 添加图token到分词器
        self.tokenizer.add_tokens(graph_tokens)

    def _linearize_graph(self):
        """将K3图结构编码为token序列"""
        tokens = ["[GRAPH_START]"]
        
        # 编码节点
        for node in self.graph_data["nodes"]:
            tokens.extend([
                "[NODE]",
                f"[NODE_{node['id']}]",
                f"[ALIAS_{node['alias']}]"
            ])
        
        # 编码边（优化无向边表示）
        seen_edges = set()
        for edge in self.graph_data["edges"]:
            lit = edge["literals"][0]
            src, tgt = lit["arguments"]
            # 去重处理
            if (tgt, src) not in seen_edges:
                tokens.extend([
                    "[EDGE]",
                    f"[PRED_{lit['predicate']}]",
                    f"[ARG_{src}]",
                    f"[ARG_{tgt}]"
                ])
                seen_edges.add((src, tgt))
        
        tokens.append("[GRAPH_END]")
        print(tokens)
        return tokens

    def _process_samples(self, raw_data):
        """处理样本时整合图结构和生成标签"""
        graph_tokens = self._linearize_graph()
        graph_ids = self.tokenizer.convert_tokens_to_ids(graph_tokens)
        
        for raw_sample in raw_data:
            # 原数据处理流程
            tree_tokens = self._linearize_tree(raw_sample.get('state', {}).get('tree', {}))
            op_tokens = self._process_operations(raw_sample.get('available_ops', []))
            
            # 生成输入序列
            full_sequence = graph_ids + self.tokenizer.convert_tokens_to_ids(tree_tokens + op_tokens)
            
            # 生成动作标签
            labels = self._generate_action_labels(raw_sample)
            
            self.samples.append({
                'input_ids': full_sequence[:self.max_seq_length],
                'raw_data': raw_sample,
                'labels': labels  # 确保添加labels字段
            })

    def _generate_action_labels(self, raw_sample):
        """生成三维动作奖励标签（根据ACTION_MAP长度）"""
        num_actions = len(self.ACTION_MAP)
        labels = [0.0] * num_actions
        
        selected_action = raw_sample.get('selected_op').get('action')
        reward = raw_sample.get('reward', 0.0)
        
        action_idx = self.ACTION_MAP.get(selected_action, -1)
        # print(action_idx, selected_action)
        if 0 <= action_idx < num_actions:
            labels[action_idx] = reward
            
        return labels

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 处理填充
        pad_id = self.tokenizer.special_tokens['[PAD]']
        seq_len = len(sample['input_ids'])
        
        # 转换为张量后再处理
        input_ids = torch.tensor(
            sample['input_ids'] + [pad_id] * (self.max_seq_length - seq_len),
            dtype=torch.long
        )
        
        attention_mask = torch.tensor(
            [1]*seq_len + [0]*(self.max_seq_length - seq_len),
            dtype=torch.long
        )
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'graph_mask': self._create_graph_mask(input_ids),  # 传入张量
            'labels': torch.tensor(sample['labels'], dtype=torch.float),
            'reward': torch.tensor(self._safe_get(sample['raw_data'], 'reward', 0.0), dtype=torch.float)
        }

    def _create_graph_mask(self, input_ids):
        """创建图结构注意力掩码（接收张量输入）"""
        mask = torch.zeros_like(input_ids)
        token_list = input_ids.tolist()
        
        try:
            start = token_list.index(self.tokenizer.vocab["[GRAPH_START]"])
            end = token_list.index(self.tokenizer.vocab["[GRAPH_END]"])
            mask[start:end+1] = 1
        except ValueError:
            pass
        
        return mask

# 使用示例
if __name__ == "__main__":
    dataset = GraphSLIDataset(
        sli_file="../data/training_data.json",
        graph_file="../data/k3_graph.json",
        max_seq_length=512
    )
    
    sample = dataset[0]
    print("输入序列长度:", len(sample['input_ids']))
    print("图掩码示例:", sample['graph_mask'][:10])
    print(sample)
    print(sample['reward'].shape)
    print(sample['labels'])
    # print(sample['input_ids'])
    # print(sample['reward'])
    # print(sample['attention_mask'].shape)
    # print(sample['raw_data'])
    