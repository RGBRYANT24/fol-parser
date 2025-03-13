#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <map>
#include <filesystem>
#include <nlohmann/json.hpp>

class BipartiteChecker {
private:
    // 图的邻接表表示
    std::vector<std::vector<int>> graph;
    // 节点映射：从JSON中的CONST标识符到内部节点ID
    std::map<std::string, int> nodeMap;
    // 颜色数组：-1表示未着色，0和1表示两种不同的颜色
    std::vector<int> colors;
    int numNodes;
    int numEdges;
    double EdgeDensity;

public:
    // 构造函数
    BipartiteChecker() : numNodes(0), numEdges(0) {}
    
    // 从JSON文件加载图
    bool loadGraphFromJson(const std::string& filePath) {
        try {
            std::ifstream file(filePath);
            if (!file.is_open()) {
                std::cerr << "无法打开文件: " << filePath << std::endl;
                return false;
            }

            nlohmann::json data;
            file >> data;
            file.close();

            // 清除旧图
            nodeMap.clear();
            graph.clear();
            numEdges = 0;
            
            // 处理节点
            int nodeId = 0;
            for (const auto& node : data["graph"]["nodes"]) {
                std::string constId = node["id"];
                nodeMap[constId] = nodeId++;
            }
            
            numNodes = nodeId;
            graph.resize(numNodes);
            
            // 处理边
            for (const auto& edge : data["graph"]["edges"]) {
                for (const auto& literal : edge["literals"]) {
                    if (literal["predicate"] == "E") {
                        std::string arg1 = literal["arguments"][0];
                        std::string arg2 = literal["arguments"][1];
                        
                        int node1 = nodeMap[arg1];
                        int node2 = nodeMap[arg2];
                        
                        // 添加无向边
                        graph[node1].push_back(node2);
                        graph[node2].push_back(node1);
                        numEdges++; // 计数边的数量（注意这里计算的是无向边）
                    }
                }
            }
            this->EdgeDensity = this->getEdgeDensity();
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "加载图时出错: " << e.what() << std::endl;
            return false;
        }
    }
    
    // 获取图的节点数
    int getNumNodes() const {
        return numNodes;
    }
    
    // 获取图的边数（无向边）
    int getNumEdges() const {
        return numEdges / 2; // 因为每条边在邻接表中被计算了两次
    }
    
    // 获取图的边密度：边数/(n(n-1)/2)
    double getEdgeDensity() const {
        if (numNodes <= 1) return 0.0; // 避免除以零
        
        int actualEdges = getNumEdges();
        double maxPossibleEdges = (numNodes * (numNodes - 1)) / 2.0;
        return actualEdges / maxPossibleEdges;
    }
    
    // 检查图是否可以二染色
    bool isBipartite() {
        if (numNodes == 0) return true;
        
        // 初始化所有节点为未着色
        colors.assign(numNodes, -1);
        
        // 对于每个连通分量进行BFS着色
        for (int start = 0; start < numNodes; start++) {
            if (colors[start] != -1) continue; // 跳过已着色的节点
            
            // BFS着色
            std::queue<int> q;
            q.push(start);
            colors[start] = 0; // 初始节点染色为0
            
            while (!q.empty()) {
                int node = q.front();
                q.pop();
                
                // 检查所有邻居
                for (int neighbor : graph[node]) {
                    if (colors[neighbor] == -1) {
                        // 邻居未着色，染上相反的颜色
                        colors[neighbor] = 1 - colors[node];
                        q.push(neighbor);
                    } 
                    else if (colors[neighbor] == colors[node]) {
                        // 邻居已着色且颜色相同，说明存在奇环，不是二分图
                        return false;
                    }
                }
            }
        }
        
        // 所有连通分量都可以正确二染色
        return true;
    }
    
    // 打印图的结构（用于调试）
    void printGraph() {
        std::cout << "Graph structure:" << std::endl;
        for (int i = 0; i < numNodes; i++) {
            std::cout << "Node " << i << " connects to: ";
            for (int neighbor : graph[i]) {
                std::cout << neighbor << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // 打印图的统计信息
    void printGraphStats() {
        std::cout << "图统计信息:" << std::endl;
        std::cout << "节点数: " << getNumNodes() << std::endl;
        std::cout << "边数: " << getNumEdges() << std::endl;
        std::cout << "边密度: " << getEdgeDensity() << std::endl;
    }
};