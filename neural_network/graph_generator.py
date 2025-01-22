# graph_generator.py
import json
from itertools import permutations  # 改为导入permutations

class GraphGenerator:
    @staticmethod
    def generate_k3(name="K3_default"):
        """生成标准K3图的JSON结构，包含六条有向边"""
        nodes = [{"id": f"VAR{i}", "type": "variable", "alias": chr(97+i)} for i in range(3)]
        
        edges = []
        # 遍历所有节点排列生成有向边
        for (i, j) in permutations(range(3), 2):
            a, b = chr(97+i), chr(97+j)
            edges.append({
                "type": "directed",  # 类型改为有向
                "literals": [
                    {"predicate": "E", "arguments": [a, b]}  # 每个边只包含一个方向的文字
                ]
            })
        
        return {
            "graph_id": f"K3_{name}",
            "type": "complete",
            "nodes": nodes,
            "edges": edges
        }

    @staticmethod
    def save_graphs(graphs, filename):
        """保存图集合到JSON文件"""
        data = {"graphs": graphs}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    # 生成示例K3图
    k3_graph = GraphGenerator.generate_k3("demo")
    GraphGenerator.save_graphs([k3_graph], "k3_graph.json")