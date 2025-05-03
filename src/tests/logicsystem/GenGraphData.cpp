#include <gtest/gtest.h>
#include "SLIResolution.h"
#include "BFSStrategy.h"
#include "KnowledgeBase.h"
#include "SLIMCTSProver.h"
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>
#include <set>
#include <unordered_set>
#include <memory>
#include <queue>
#include <ctime>

namespace LogicSystem
{
    class GenRandomGraphColoringTest : public ::testing::Test
    {
    protected:
        std::vector<SymbolId> allConstants;

        GenRandomGraphColoringTest() {}

        void SetUp() override
        {
            allConstants.clear();
            // 初始化随机数种子
            std::srand(static_cast<unsigned int>(std::time(nullptr)));
        }

        // 创建搜索策略
        BFSStrategy createStrategy(int maxDepth = 10)
        {
            return BFSStrategy(maxDepth, 60.0, 1024 * 1024 * 100);
        }

        // 将图编码成字符串用于去重
        std::string hashGraph(const std::vector<std::pair<SymbolId, SymbolId>> &edges, int numNodes)
        {
            std::stringstream ss;
            ss << numNodes << ":";
            std::set<std::pair<int, int>> normalized;
            for (auto &e : edges)
            {
                int v1 = std::distance(allConstants.begin(),
                                       std::find(allConstants.begin(), allConstants.end(), e.first));
                int v2 = std::distance(allConstants.begin(),
                                       std::find(allConstants.begin(), allConstants.end(), e.second));
                if (v1 > v2)
                    std::swap(v1, v2);
                normalized.insert({v1, v2});
            }
            for (auto &p : normalized)
            {
                ss << "(" << p.first << "," << p.second << ")";
            }
            return ss.str();
        }

        // 使用简单的C标准库随机数函数
        int simpleRandom(int min, int max) {
            return min + std::rand() % (max - min + 1);
        }

        double simpleRandomDouble() {
            return static_cast<double>(std::rand()) / RAND_MAX;
        }

        // 生成一个新的、未出现过的随机连通图
        std::vector<std::pair<SymbolId, SymbolId>> generateUniqueRandomGraph(
            int numNodes,
            std::unordered_set<std::string> &generatedGraphs,
            KnowledgeBase &kb)
        {
            const int MAX_ATTEMPTS = 100;
            int attempts = 0;
            
            while (attempts < MAX_ATTEMPTS)
            {
                ++attempts;
                std::vector<std::pair<SymbolId, SymbolId>> edges;
                
                // 方法1: 简单的随机生成模型
                if (attempts % 3 == 0) {
                    // 使用小概率随机图模型 (Erdős–Rényi G(n,p) 模型)
                    double edgeProb = 0.3 + (0.4 * simpleRandomDouble()); // 在[0.3, 0.7]之间
                    
                    for (int i = 0; i < numNodes; ++i) {
                        for (int j = i + 1; j < numNodes; ++j) {
                            if (simpleRandomDouble() < edgeProb) {
                                edges.emplace_back(allConstants[i], allConstants[j]);
                            }
                        }
                    }
                }
                // 方法2: 确保连通性的树加随机边
                else if (attempts % 3 == 1) {
                    // 首先建立一个生成树确保连通性
                    std::vector<int> tree;
                    std::vector<bool> inTree(numNodes, false);
                    
                    // 从0号节点开始
                    tree.push_back(0);
                    inTree[0] = true;
                    
                    // 添加其他所有节点
                    while (tree.size() < static_cast<size_t>(numNodes)) {
                        int newNode = simpleRandom(0, numNodes - 1);
                        if (!inTree[newNode]) {
                            // 随机选择一个树中的节点连接
                            int treeNode = tree[simpleRandom(0, tree.size() - 1)];
                            edges.emplace_back(allConstants[treeNode], allConstants[newNode]);
                            tree.push_back(newNode);
                            inTree[newNode] = true;
                        }
                    }
                    
                    // 再随机添加一些边
                    int extraEdges = simpleRandom(0, numNodes); // 额外添加0到numNodes条边
                    for (int i = 0; i < extraEdges; ++i) {
                        int u = simpleRandom(0, numNodes - 1);
                        int v = simpleRandom(0, numNodes - 1);
                        if (u != v) {
                            if (u > v) std::swap(u, v);
                            // 检查边是否已存在
                            bool exists = false;
                            for (const auto& edge : edges) {
                                int e1 = std::distance(allConstants.begin(), 
                                        std::find(allConstants.begin(), allConstants.end(), edge.first));
                                int e2 = std::distance(allConstants.begin(), 
                                        std::find(allConstants.begin(), allConstants.end(), edge.second));
                                if (e1 > e2) std::swap(e1, e2);
                                if (e1 == u && e2 == v) {
                                    exists = true;
                                    break;
                                }
                            }
                            if (!exists) {
                                edges.emplace_back(allConstants[u], allConstants[v]);
                            }
                        }
                    }
                }
                // 方法3: 环加随机边
                else {
                    // 创建一个环
                    for (int i = 0; i < numNodes; ++i) {
                        edges.emplace_back(allConstants[i], allConstants[(i + 1) % numNodes]);
                    }
                    
                    // 添加少量随机边
                    int extraEdges = simpleRandom(0, numNodes / 2);
                    for (int i = 0; i < extraEdges; ++i) {
                        int u = simpleRandom(0, numNodes - 1);
                        int v = simpleRandom(0, numNodes - 1);
                        if (u != v && abs(u - v) > 1 && !(u == 0 && v == numNodes - 1) && !(v == 0 && u == numNodes - 1)) {
                            if (u > v) std::swap(u, v);
                            edges.emplace_back(allConstants[u], allConstants[v]);
                        }
                    }
                }
                
                // 确保图是连通的
                if (!isConnected(edges, numNodes)) {
                    makeConnected(edges, numNodes);
                }
                
                // 检查是否生成了新图
                auto h = hashGraph(edges, numNodes);
                if (!generatedGraphs.count(h)) {
                    generatedGraphs.insert(h);
                    return edges;
                }
            }
            
            // 如果MAX_ATTEMPTS内没生成新图，使用简单线性连接模式
            std::vector<std::pair<SymbolId, SymbolId>> edges;
            
            // 创建一个路径图
            for (int i = 0; i < numNodes - 1; ++i) {
                edges.emplace_back(allConstants[i], allConstants[i + 1]);
            }
            
            // 根据尝试次数添加一些额外的边
            int extraCount = attempts % numNodes;
            for (int i = 0; i < extraCount; ++i) {
                int u = i % numNodes;
                int v = (u + 2) % numNodes;
                if (u > v) std::swap(u, v);
                edges.emplace_back(allConstants[u], allConstants[v]);
            }
            
            auto h = hashGraph(edges, numNodes);
            generatedGraphs.insert(h);
            return edges;
        }

        bool isConnected(const std::vector<std::pair<SymbolId, SymbolId>> &edges, int numNodes)
        {
            if (numNodes <= 1)
                return true;
            std::vector<std::vector<int>> adj(numNodes);
            for (auto &e : edges)
            {
                int v1 = std::distance(allConstants.begin(),
                                       std::find(allConstants.begin(), allConstants.end(), e.first));
                int v2 = std::distance(allConstants.begin(),
                                       std::find(allConstants.begin(), allConstants.end(), e.second));
                adj[v1].push_back(v2);
                adj[v2].push_back(v1);
            }
            std::vector<bool> vis(numNodes, false);
            std::queue<int> q;
            q.push(0);
            vis[0] = true;
            while (!q.empty())
            {
                int u = q.front();
                q.pop();
                for (int w : adj[u])
                {
                    if (!vis[w])
                    {
                        vis[w] = true;
                        q.push(w);
                    }
                }
            }
            for (bool b : vis)
                if (!b)
                    return false;
            return true;
        }

        void makeConnected(std::vector<std::pair<SymbolId, SymbolId>> &edges, int numNodes)
        {
            if (numNodes <= 1)
                return;
                
            // 找出所有连通分量
            std::vector<std::vector<int>> adj(numNodes);
            for (auto &e : edges)
            {
                int v1 = std::distance(allConstants.begin(),
                                       std::find(allConstants.begin(), allConstants.end(), e.first));
                int v2 = std::distance(allConstants.begin(),
                                       std::find(allConstants.begin(), allConstants.end(), e.second));
                adj[v1].push_back(v2);
                adj[v2].push_back(v1);
            }
            
            std::vector<bool> vis(numNodes, false);
            std::vector<std::vector<int>> components;
            
            for (int i = 0; i < numNodes; ++i) {
                if (!vis[i]) {
                    std::vector<int> comp;
                    std::queue<int> q;
                    q.push(i);
                    vis[i] = true;
                    comp.push_back(i);
                    
                    while (!q.empty()) {
                        int u = q.front();
                        q.pop();
                        
                        for (int v : adj[u]) {
                            if (!vis[v]) {
                                vis[v] = true;
                                q.push(v);
                                comp.push_back(v);
                            }
                        }
                    }
                    
                    components.push_back(comp);
                }
            }
            
            // 连接所有连通分量
            for (size_t i = 1; i < components.size(); ++i) {
                // 选择每个分量中的第一个节点进行连接
                int u = components[0][0];
                int v = components[i][0];
                edges.emplace_back(allConstants[u], allConstants[v]);
            }
        }

        void saveGraphToFile(const std::vector<std::pair<SymbolId, SymbolId>> &edges,
                             int numNodes,
                             const std::string &filename,
                             KnowledgeBase &kb)
        {
            std::ofstream out(filename);
            if (!out)
                return;
            out << "Graph with " << numNodes << " nodes:\nNodes: ";
            for (int i = 0; i < numNodes; ++i)
            {
                out << kb.getSymbolName(allConstants[i])
                    << (i + 1 < numNodes ? ", " : "\n");
            }
            out << "Edges:\n";
            for (auto &e : edges)
            {
                out << kb.getSymbolName(e.first)
                    << " -- "
                    << kb.getSymbolName(e.second)
                    << "\n";
            }
        }

        void setupTwoColoringKB(KnowledgeBase &kb,
                                int &pred_E, int &pred_R, int &pred_G, int &pred_uncol,
                                SymbolId &var_x, SymbolId &var_y)
        {
            pred_E = kb.addPredicate("E");
            pred_R = kb.addPredicate("R");
            pred_G = kb.addPredicate("G");
            pred_uncol = kb.addPredicate("uncol");
            var_x = kb.addVariable("x");
            var_y = kb.addVariable("y");

            // 相邻不能同色 R
            Clause c1;
            c1.addLiteral(Literal(pred_E, {var_x, var_y}, true));
            c1.addLiteral(Literal(pred_R, {var_x}, true));
            c1.addLiteral(Literal(pred_R, {var_y}, true));
            c1.addLiteral(Literal(pred_uncol, {}, false));
            kb.addClause(c1);

            // 相邻不能同色 G
            Clause c2;
            c2.addLiteral(Literal(pred_E, {var_x, var_y}, true));
            c2.addLiteral(Literal(pred_G, {var_x}, true));
            c2.addLiteral(Literal(pred_G, {var_y}, true));
            c2.addLiteral(Literal(pred_uncol, {}, false));
            kb.addClause(c2);

            // 每点至少着色
            Clause c3;
            c3.addLiteral(Literal(pred_R, {var_x}, false));
            c3.addLiteral(Literal(pred_G, {var_x}, false));
            kb.addClause(c3);
        }

        void addGraphEdgesToKB(const std::vector<std::pair<SymbolId, SymbolId>> &edges,
                               KnowledgeBase &kb, int pred_E)
        {
            for (auto &e : edges)
            {
                Clause c;
                c.addLiteral(Literal(pred_E, {e.first, e.second}, false));
                kb.addClause(c);
                Clause d;
                d.addLiteral(Literal(pred_E, {e.second, e.first}, false));
                kb.addClause(d);
            }
        }
    };

    TEST_F(GenRandomGraphColoringTest, GenerateGraphData)
    {
        std::string outputDir = "/home/jiangguifei01/aiderun/fol-parser/fol-parser/data/random_graphs";
        std::unordered_set<std::string> generatedGraphs;
        
        for (int test = 0; test < 35; ++test)
        {
            KnowledgeBase kb;
            int pred_E, pred_R, pred_G, pred_uncol;
            SymbolId var_x, var_y;

            // 添加 10 个常量
            SymbolId consts[] = {
                kb.addConstant("a"), kb.addConstant("b"), kb.addConstant("c"),
                kb.addConstant("d"), kb.addConstant("e"), kb.addConstant("f"),
                kb.addConstant("g"), kb.addConstant("h"), kb.addConstant("i"),
                kb.addConstant("j")};
            allConstants.assign(consts, consts + 10);

            // 生成6-10个节点的随机图
            int numNodes = 6 + std::rand() % 5; // 生成[6,10]范围内的节点数
            auto edges = generateUniqueRandomGraph(numNodes, generatedGraphs, kb);

            std::stringstream ss;
            ss << outputDir << "/graph_" << numNodes << "_nodes_test_" << test << ".txt";
            std::string filename = ss.str();
            saveGraphToFile(edges, numNodes, filename, kb);

            setupTwoColoringKB(kb, pred_E, pred_R, pred_G, pred_uncol, var_x, var_y);
            addGraphEdgesToKB(edges, kb, pred_E);

            std::cout << "Test " << test << ", nodes=" << numNodes << "\n";
            kb.print();

            Clause goal;
            goal.addLiteral(Literal(pred_uncol, {}, true));

            bool proved = false;
            std::ofstream rf(filename, std::ios::app);
            rf << "\nMCTS Attempts:\n";
            for (int attempt = 1; attempt <= 5 && !proved; ++attempt)
            {
                SLIMCTSProver prover(kb, goal);
                proved = prover.prove("/home/jiangguifei01/aiderun/fol-parser/fol-parser/data/newreward");
                std::string res = proved ? "Not two-colorable" : "Two-colorable/inconclusive";
                std::cout << "  Attempt " << attempt << ": " << res << "\n";
                rf << "Attempt " << attempt << ": " << res << "\n";
            }
            rf << "\nFinal: " << (proved ? "Not two-colorable" : "Likely two-colorable")
               << "\n";
            rf.close();
        }
    }

} // namespace LogicSystem