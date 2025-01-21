import json
from collections import defaultdict

# ======================
# 1. JSON数据加载与解析
# ======================
def load_data(file_path):
    """加载并解析JSON数据"""
    with open(file_path) as f:
        data = json.load(f)
    return data["samples"]

# ======================
# 2. 词汇表构建器（关键修改）
# ======================
class VocabularyBuilder:
    def __init__(self):
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.special_tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[NODE]",
            "(", ")", ",", "¬", "∨"
        ]
        
        # 初始化特殊标记（包括 [NODE]）
        for token in self.special_tokens:
            _ = self.vocab[token]
    
    def build_from_data(self, data):
        """从数据中自动构建词汇表"""
        action_types = set()  # 收集所有动作类型
        
        for sample in data:
            # 处理树节点
            for node in sample["current_tree_state"]["nodes"]:
                self._process_literal(node["node_lit"])
                # 添加结构标记
                _ = self.vocab[f"[DEPTH]{node['depth']}"]
                _ = self.vocab[f"[ACTIVE]{int(node['is_active'])}"]
                _ = self.vocab[f"[A_LIT]{int(node['is_A_literal'])}"]
            
            # 处理操作
            for op in sample["available_operations"] + [sample["selected_operation"]]:
                action_types.add(op["action_type"])  # 收集动作类型
                self._process_literal(op["node1_lit"])
                if "second_operand" in op:
                    self._process_operand(op["second_operand"])
                if "kb_clause" in op:
                    self._process_clause(op["kb_clause"])
        
        # 添加动作标记（与OperationEncoder中的action_map一致）
        action_map = {
            "EXTENSION": "[ACTION_EXT]",
            "FACTORING": "[ACTION_FACT]",
            "ANCESTRY": "[ACTION_ANC]",
            "TRUNCATE": "[ACTION_TRUN]"
        }
        for action in action_types:
            token = action_map.get(action, f"[ACTION_{action}]")
            _ = self.vocab[token]
    
    def _process_literal(self, literal):
        """处理文字字符串"""
        parts = literal.replace("(", " ( ").replace(")", " ) ").split()
        for part in parts:
            if part not in ["(", ")", "¬"]:
                self.vocab[part]
    
    def _process_clause(self, clause):
        """处理子句字符串"""
        parts = clause.replace("∨", " ∨ ").split()
        for part in parts:
            if part not in ["∨", "(", ")"]:
                self.vocab[part]
    
    def _process_operand(self, operand):
        """处理操作数字符串"""
        if isinstance(operand, str):  # 如果是literal
            self._process_literal(operand)
    
    def get_vocab(self):
        """获取最终的词汇表字典"""
        return dict(self.vocab)

# ======================
# 3. 树结构编码器
# ======================
class TreeEncoder:
    def __init__(self, vocab, max_length=256):
        self.vocab = vocab
        self.max_length = max_length
        self.unk_token = "[UNK]"
    
    def encode(self, tree_state):
        """编码整个树结构"""
        encoded_sequence = [self.vocab["[CLS]"]]
        
        # 按深度遍历节点
        for depth_level in tree_state["depth_map"]:
            for node_id in depth_level:
                node = next(n for n in tree_state["nodes"] if n["node_id"] == node_id)
                encoded_sequence += self._encode_node(node)
        
        # 截断并填充序列
        return self._pad_sequence(encoded_sequence)
    
    def _encode_node(self, node):
        """编码单个节点"""
        tokens = []
        tokens += self._tokenize_literal(node["node_lit"])
        tokens.append(self.vocab[f"[DEPTH]{node['depth']}"])
        tokens.append(self.vocab[f"[ACTIVE]{int(node['is_active'])}"])
        tokens.append(self.vocab[f"[A_LIT]{int(node['is_A_literal'])}"])
        return tokens
    
    def _tokenize_literal(self, literal):
        """将文字转换为标记序列"""
        tokens = []
        parts = literal.replace("(", " ( ").replace(")", " ) ").split()
        for part in parts:
            tokens.append(self.vocab.get(part, self.vocab[self.unk_token]))
        return tokens
    
    def _pad_sequence(self, sequence):
        """处理序列长度"""
        if len(sequence) >= self.max_length:
            return sequence[:self.max_length-1] + [self.vocab["[SEP]"]]
        else:
            return sequence + [self.vocab["[SEP]"]] + [self.vocab["[PAD]"]] * (self.max_length - len(sequence) - 1)

# ======================
# 4. 操作编码器（关键修改）
# ======================
class OperationEncoder:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_token = "[UNK]"
        self.action_map = {
            "EXTENSION": "[ACTION_EXT]",
            "FACTORING": "[ACTION_FACT]",
            "ANCESTRY": "[ACTION_ANC]",
            "TRUNCATE": "[ACTION_TRUN]"
        }
    
    def encode(self, operation):
        """编码单个操作"""
        tokens = []
        # 动作类型
        action_token = self.action_map[operation["action_type"]]
        tokens.append(self.vocab[action_token])
        # 节点文字
        tokens += self._tokenize_literal(operation["node1_lit"])
        # 操作数
        if operation["second_operand_type"] == "literal":
            tokens += self._tokenize_literal(operation["second_operand"])
        elif operation["second_operand_type"] == "node":
            tokens.append(self.vocab["[NODE]"])
            tokens.append(int(operation["second_operand_id"]))
        # 知识库子句
        if "kb_clause" in operation:
            tokens += self._tokenize_clause(operation["kb_clause"])
        return tokens
    
    def _tokenize_literal(self, literal):
        """独立实现文字处理"""
        tokens = []
        parts = literal.replace("(", " ( ").replace(")", " ) ").split()
        for part in parts:
            tokens.append(self.vocab.get(part, self.vocab[self.unk_token]))
        return tokens
    
    def _tokenize_clause(self, clause):
        """处理子句"""
        tokens = []
        parts = clause.replace("∨", " ∨ ").split()
        for part in parts:
            tokens.append(self.vocab.get(part, self.vocab[self.unk_token]))
        return tokens

# ======================
# 5. 主处理流程
# ======================
def process_data(input_file, output_file):
    data = load_data(input_file)
    vocab_builder = VocabularyBuilder()
    vocab_builder.build_from_data(data)
    vocab = vocab_builder.get_vocab()
    
    tree_encoder = TreeEncoder(vocab)
    op_encoder = OperationEncoder(vocab)
    
    processed_data = []
    for sample in data:
        tree_tokens = tree_encoder.encode(sample["current_tree_state"])
        available_ops_tokens = [op_encoder.encode(op) for op in sample["available_operations"]]
        selected_op_tokens = op_encoder.encode(sample["selected_operation"])
        processed_data.append({
            "state_id": sample["state_id"],
            "tree_tokens": tree_tokens,
            "available_ops_tokens": available_ops_tokens,
            "selected_op_tokens": selected_op_tokens,
            "reward": sample["reward"]
        })
    
    with open(output_file, "w") as f:
        json.dump({"vocab": vocab, "data": processed_data}, f, indent=2)

if __name__ == "__main__":
    input_json = "../data/training_data.json"
    output_json = "processed_sequences.json"
    process_data(input_json, output_json)
    print(f"数据处理完成，已保存到 {output_json}")
    print("词汇表大小:", len(json.load(open(output_json))["vocab"]))