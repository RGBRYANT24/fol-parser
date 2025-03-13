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

namespace LogicSystem
{
    class RandomGraphColoringTest : public ::testing::Test
    {
    protected:
        int requestedDensityType = -1; // -1: 随机, 0: 稀疏, 1: 中等, 2: 稠密, 3: 自定义
        double customMinDensity = 0.0;
        double customMaxDensity = 1.0;
        void SetUp() override
        {
            // 设置随机数生成器
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            rng = std::mt19937(seed);

            // 初始化常量向量
            allConstants.clear();
        }

        // 设置图密度类型的方法
        void setDensityType(int type)
        {
            requestedDensityType = type;
        }

        // 设置自定义密度范围的方法
        void setCustomDensityRange(double min, double max)
        {
            customMinDensity = min;
            customMaxDensity = max;
            // 自动设置为自定义密度类型
            requestedDensityType = 0;
        }

        // 辅助函数：创建搜索策略
        BFSStrategy createStrategy(int maxDepth = 10)
        {
            return BFSStrategy(maxDepth, 60.0, 1024 * 1024 * 100);
        }

        // 哈希图结构以便比较不同的图
        std::string hashGraph(const std::vector<std::pair<SymbolId, SymbolId>> &edges, int numNodes)
        {
            std::stringstream ss;
            ss << numNodes << ":";

            // 标准化边表示（确保第一个顶点编号小于第二个）
            std::set<std::pair<int, int>> normalizedEdges;
            for (const auto &edge : edges)
            {
                int v1 = std::distance(allConstants.begin(), std::find(allConstants.begin(), allConstants.end(), edge.first));
                int v2 = std::distance(allConstants.begin(), std::find(allConstants.begin(), allConstants.end(), edge.second));
                if (v1 > v2)
                    std::swap(v1, v2);
                normalizedEdges.insert({v1, v2});
            }

            // 将规范化的边集转为字符串
            for (const auto &edge : normalizedEdges)
            {
                ss << "(" << edge.first << "," << edge.second << ")";
            }

            return ss.str();
        }

        // 检查图是否包含K3完全子图（三角形）
        bool containsK3(const std::vector<std::pair<SymbolId, SymbolId>> &edges, int numNodes)
        {
            // 构建邻接表
            std::vector<std::vector<int>> adj(numNodes);
            for (const auto &edge : edges)
            {
                int v1 = std::distance(allConstants.begin(), std::find(allConstants.begin(), allConstants.end(), edge.first));
                int v2 = std::distance(allConstants.begin(), std::find(allConstants.begin(), allConstants.end(), edge.second));
                adj[v1].push_back(v2);
                adj[v2].push_back(v1);
            }

            // 检查是否存在三角形
            for (int i = 0; i < numNodes; i++)
            {
                for (size_t j = 0; j < adj[i].size(); j++)
                {
                    int neighbor1 = adj[i][j];
                    if (neighbor1 <= i)
                        continue; // 避免重复检查

                    for (size_t k = j + 1; k < adj[i].size(); k++)
                    {
                        int neighbor2 = adj[i][k];
                        if (neighbor2 <= i)
                            continue; // 避免重复检查

                        // 检查neighbor1和neighbor2之间是否存在边
                        for (int node : adj[neighbor1])
                        {
                            if (node == neighbor2)
                            {
                                return true; // 找到三角形 i-neighbor1-neighbor2
                            }
                        }
                    }
                }
            }
            return false;
        }

        // 在图中添加一个K3完全子图（三角形）
        void addK3(std::vector<std::pair<SymbolId, SymbolId>> &edges)
        {
            // 随机选择三个节点
            std::uniform_int_distribution<int> dist(0, allConstants.size() - 1);

            std::set<int> triangleNodes;
            while (triangleNodes.size() < 3)
            {
                triangleNodes.insert(dist(rng));
            }

            std::vector<int> nodes(triangleNodes.begin(), triangleNodes.end());

            // 添加三条边形成三角形
            edges.push_back({allConstants[nodes[0]], allConstants[nodes[1]]});
            edges.push_back({allConstants[nodes[1]], allConstants[nodes[2]]});
            edges.push_back({allConstants[nodes[0]], allConstants[nodes[2]]});
        }

        // 随机生成无向图，确保与之前生成的图不同且包含K3完全子图
        // 随机生成无向图，确保与之前生成的图不同且包含K3完全子图
        std::vector<std::pair<SymbolId, SymbolId>> generateUniqueRandomGraph(
            int numNodes,
            std::unordered_set<std::string> &generatedGraphs,
            KnowledgeBase &kb,
            std::string &densityCategory,
            int densityTypeRequest = 0, // -1: 随机, 0: 稀疏, 1: 中等, 2: 稠密, 3: 自定义
            double minDensity = 0.0,     // 自定义密度下限
            double maxDensity = 0.5)     // 自定义密度上限
        {
            // 最大尝试次数，防止无限循环
            const int MAX_ATTEMPTS = 100;
            int attempts = 0;

            while (attempts < MAX_ATTEMPTS)
            {
                attempts++;

                std::vector<std::pair<SymbolId, SymbolId>> edges;

                // 密度类型（如果没有指定，则随机选择）
                int densityType = densityTypeRequest;
                if (densityType == -1)
                {
                    std::uniform_int_distribution<int> densityTypeDist(0, 2);
                    densityType = densityTypeDist(rng);
                }

                // 根据密度类别设置边概率
                double edgeProb;
                double maxEdges = numNodes * (numNodes - 1) / 2.0; // 完全图的边数

                // 稀疏图 (密度 < 0.15)
                if (densityType == 0)
                {
                    std::uniform_real_distribution<double> edgeProbDist(0.05, 0.14);
                    edgeProb = edgeProbDist(rng);
                    densityCategory = "Sparse";
                }
                // 中等密度图 (0.15 <= 密度 < 0.35)
                else if (densityType == 1)
                {
                    std::uniform_real_distribution<double> edgeProbDist(0.15, 0.34);
                    edgeProb = edgeProbDist(rng);
                    densityCategory = "Medium";
                }
                // 稠密图 (密度 >= 0.35)
                else if (densityType == 2)
                {
                    std::uniform_real_distribution<double> edgeProbDist(0.35, 0.7);
                    edgeProb = edgeProbDist(rng);
                    densityCategory = "Dense";
                }
                // 自定义密度范围
                else if (densityType == 3)
                {
                    std::uniform_real_distribution<double> edgeProbDist(minDensity, maxDensity);
                    edgeProb = edgeProbDist(rng);
                    densityCategory = "Custom";
                }

                std::uniform_real_distribution<double> dist(0.0, 1.0);

                // 尝试在每对节点之间添加边
                for (int i = 0; i < numNodes; i++)
                {
                    for (int j = i + 1; j < numNodes; j++)
                    {
                        if (dist(rng) < edgeProb)
                        {
                            edges.push_back({allConstants[i], allConstants[j]});
                        }
                    }
                }

                // 确保图包含K3完全子图
                if (!containsK3(edges, numNodes))
                {
                    addK3(edges);
                }

                // 确保图是连通的
                if (!isConnected(edges, numNodes))
                {
                    makeConnected(edges, numNodes);
                }

                // 重新计算实际密度，因为添加连通性和K3可能改变了边的数量
                double actualDensity = edges.size() / maxEdges;

                // 更新密度类别(根据实际密度)
                if (actualDensity < 0.15)
                {
                    densityCategory = "Sparse";
                }
                else if (actualDensity < 0.35)
                {
                    densityCategory = "Medium";
                }
                else
                {
                    densityCategory = "Dense";
                }

                // 对于自定义密度类型，加上实际的密度值
                if (densityType == 3)
                {
                    densityCategory = "Custom(" + std::to_string(actualDensity).substr(0, 5) + ")";
                }

                // 检查这个图是否已经生成过
                std::string graphHash = hashGraph(edges, numNodes);
                if (generatedGraphs.find(graphHash) == generatedGraphs.end())
                {
                    generatedGraphs.insert(graphHash);
                    return edges;
                }
            }

            // 如果无法生成唯一的图，添加边数来创建新图
            std::vector<std::pair<SymbolId, SymbolId>> edges;

            // 默认密度类别
            densityCategory = "Medium";

            // 用一个变化的模式来添加边
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = i + 1; j < numNodes; j++)
                {
                    if ((i + j + attempts) % 3 == 0)
                    {
                        edges.push_back({allConstants[i], allConstants[j]});
                    }
                }
            }

            // 确保图是连通的
            makeConnected(edges, numNodes);

            // 确保图包含K3完全子图
            if (!containsK3(edges, numNodes))
            {
                addK3(edges);
            }

            // 重新计算实际密度
            double maxEdges = numNodes * (numNodes - 1) / 2.0;
            double actualDensity = edges.size() / maxEdges;

            // 更新密度类别
            if (actualDensity < 0.15)
            {
                densityCategory = "Sparse";
            }
            else if (actualDensity < 0.35)
            {
                densityCategory = "Medium";
            }
            else
            {
                densityCategory = "Dense";
            }

            // 对于自定义密度类型，加上实际的密度值
            if (densityTypeRequest == 3)
            {
                densityCategory = "Custom(" + std::to_string(actualDensity).substr(0, 5) + ")";
            }

            std::string graphHash = hashGraph(edges, numNodes);
            generatedGraphs.insert(graphHash);

            return edges;
        }

        // 检查图是否连通
        bool isConnected(const std::vector<std::pair<SymbolId, SymbolId>> &edges, int numNodes)
        {
            if (numNodes <= 1)
                return true;

            // 构建邻接表
            std::vector<std::vector<int>> adj(numNodes);
            for (const auto &edge : edges)
            {
                auto it1 = std::find(allConstants.begin(), allConstants.end(), edge.first);
                auto it2 = std::find(allConstants.begin(), allConstants.end(), edge.second);

                // 确保找到了两个顶点
                if (it1 == allConstants.end() || it2 == allConstants.end())
                    continue; // 跳过找不到的边

                int v1 = std::distance(allConstants.begin(), it1);
                int v2 = std::distance(allConstants.begin(), it2);

                // 确保索引在有效范围内
                if (v1 >= 0 && v1 < numNodes && v2 >= 0 && v2 < numNodes)
                {
                    adj[v1].push_back(v2);
                    adj[v2].push_back(v1);
                }
            }

            // BFS检查连通性
            std::vector<bool> visited(numNodes, false);
            std::queue<int> q;
            q.push(0);
            visited[0] = true;

            while (!q.empty())
            {
                int node = q.front();
                q.pop();

                for (int neighbor : adj[node])
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }

            // 检查是否所有节点都被访问到
            for (int i = 0; i < numNodes; i++)
            {
                if (!visited[i])
                    return false;
            }

            return true;
        }

        // 使图连通
        void makeConnected(std::vector<std::pair<SymbolId, SymbolId>> &edges, int numNodes)
        {
            if (numNodes <= 1)
                return;

            // 确保 allConstants 的大小足够
            if (allConstants.size() < static_cast<size_t>(numNodes))
            {
                std::cerr << "Error: allConstants size (" << allConstants.size()
                          << ") is less than numNodes (" << numNodes << ")" << std::endl;
                return; // 或抛出异常
            }

            // 构建邻接表
            std::vector<std::vector<int>> adj(numNodes);
            for (const auto &edge : edges)
            {
                auto it1 = std::find(allConstants.begin(), allConstants.end(), edge.first);
                auto it2 = std::find(allConstants.begin(), allConstants.end(), edge.second);

                // 检查边的节点是否存在于 allConstants 中
                if (it1 == allConstants.end() || it2 == allConstants.end())
                {
                    std::cerr << "Error: Edge contains node not in allConstants" << std::endl;
                    continue; // 跳过这条边
                }

                int v1 = std::distance(allConstants.begin(), it1);
                int v2 = std::distance(allConstants.begin(), it2);

                // 验证索引是否在合理范围内
                if (v1 >= numNodes || v2 >= numNodes || v1 < 0 || v2 < 0)
                {
                    std::cerr << "Error: Edge index out of range: " << v1 << ", " << v2 << std::endl;
                    continue; // 跳过这条边
                }

                adj[v1].push_back(v2);
                adj[v2].push_back(v1);
            }

            // 寻找连通分量
            std::vector<bool> visited(numNodes, false);
            std::vector<std::vector<int>> components;

            for (int i = 0; i < numNodes; i++)
            {
                if (!visited[i])
                {
                    std::vector<int> component;
                    std::queue<int> q;
                    q.push(i);
                    visited[i] = true;
                    component.push_back(i);

                    while (!q.empty())
                    {
                        int node = q.front();
                        q.pop();

                        for (int neighbor : adj[node])
                        {
                            if (!visited[neighbor])
                            {
                                visited[neighbor] = true;
                                q.push(neighbor);
                                component.push_back(neighbor);
                            }
                        }
                    }

                    components.push_back(component);
                }
            }

            // 如果有多个连通分量，添加边连接它们
            for (size_t i = 1; i < components.size(); i++)
            {
                if (components[0].empty() || components[i].empty())
                {
                    std::cerr << "Error: Empty component detected" << std::endl;
                    continue;
                }

                int v1 = components[0][0]; // 第一个连通分量的一个节点
                int v2 = components[i][0]; // 当前连通分量的一个节点

                // 再次验证索引是否在合理范围内
                if (v1 >= numNodes || v2 >= numNodes || v1 < 0 || v2 < 0)
                {
                    std::cerr << "Error: Component node index out of range: " << v1 << ", " << v2 << std::endl;
                    continue;
                }

                edges.push_back({allConstants[v1], allConstants[v2]});
            }
        }

        // 将图结构保存到文件
        void saveGraphToFile(const std::vector<std::pair<SymbolId, SymbolId>> &edges, int numNodes, const std::string &filename, KnowledgeBase &kb, std::string density)
        {
            std::ofstream outFile(filename);
            if (!outFile.is_open())
            {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }

            outFile << "Graph with " << numNodes << " nodes:" << std::endl;
            outFile << "Nodes: ";
            for (int i = 0; i < numNodes; i++)
            {
                outFile << kb.getSymbolName(allConstants[i]);
                if (i < numNodes - 1)
                    outFile << ", ";
            }
            outFile << std::endl;

            outFile << "Edges:" << std::endl;
            for (const auto &edge : edges)
            {
                outFile << kb.getSymbolName(edge.first) << " -- " << kb.getSymbolName(edge.second) << std::endl;
            }

            // 标记是否包含K3完全子图
            outFile << "\nContains K3 complete subgraph (triangle): "
                    << (containsK3(edges, numNodes) ? "Yes" : "No") << std::endl;

            outFile.close();
        }

        // 设置两染色问题的知识库
        void setupTwoColoringKB(KnowledgeBase &kb, int &pred_E, int &pred_R, int &pred_G, int &pred_uncol,
                                SymbolId &var_x, SymbolId &var_y)
        {
            // 添加谓词和变量
            pred_E = kb.addPredicate("E");
            pred_R = kb.addPredicate("R");
            pred_G = kb.addPredicate("G");
            pred_uncol = kb.addPredicate("uncol");
            var_x = kb.addVariable("x");
            var_y = kb.addVariable("y");

            // 1. 相邻顶点不能同色（红色）
            // ¬E(x,y)∨ ¬R(x)∨ ¬R(y)∨ uncol
            Clause adjacentNotSameColorR;
            adjacentNotSameColorR.addLiteral(Literal(pred_E, {var_x, var_y}, true));
            adjacentNotSameColorR.addLiteral(Literal(pred_R, {var_x}, true));
            adjacentNotSameColorR.addLiteral(Literal(pred_R, {var_y}, true));
            adjacentNotSameColorR.addLiteral(Literal(pred_uncol, {}, false));
            kb.addClause(adjacentNotSameColorR);

            // 2. 相邻顶点不能同色（绿色）
            // ¬E(x,y)∨ ¬G(x)∨ ¬G(y)∨ uncol
            Clause adjacentNotSameColorG;
            adjacentNotSameColorG.addLiteral(Literal(pred_E, {var_x, var_y}, true));
            adjacentNotSameColorG.addLiteral(Literal(pred_G, {var_x}, true));
            adjacentNotSameColorG.addLiteral(Literal(pred_G, {var_y}, true));
            adjacentNotSameColorG.addLiteral(Literal(pred_uncol, {}, false));
            kb.addClause(adjacentNotSameColorG);

            // 3. 每个顶点必须着色（红色或绿色）
            // R(x) ∨ G(x)
            Clause vertexMustBeColored;
            vertexMustBeColored.addLiteral(Literal(pred_R, {var_x}, false));
            vertexMustBeColored.addLiteral(Literal(pred_G, {var_x}, false));
            kb.addClause(vertexMustBeColored);
        }

        // 添加图的边到知识库
        void addGraphEdgesToKB(const std::vector<std::pair<SymbolId, SymbolId>> &edges,
                               KnowledgeBase &kb, int pred_E)
        {
            for (const auto &edge : edges)
            {
                Clause edgeClause;
                edgeClause.addLiteral(Literal(pred_E, {edge.first, edge.second}, false));
                kb.addClause(edgeClause);

                Clause edgeClauseReverse;
                edgeClauseReverse.addLiteral(Literal(pred_E, {edge.second, edge.first}, false));
                kb.addClause(edgeClauseReverse);
            }
        }

        std::mt19937 rng;
        std::vector<SymbolId> allConstants;
    };

    TEST_F(RandomGraphColoringTest, RandomGraphTwoColoringMCTS)
    {
        // 设置输出目录
        std::string outputDir = "/home/adrin/Projects/fol-parser/data/random_graphs";

        // 设置密度类型（可根据需要修改）
        // -1: 随机选择, 0: 稀疏, 1: 中等, 2: 稠密, 3: 自定义
        setDensityType(-1);

        // 如果需要自定义密度范围，可以使用以下代码（会自动设置类型为3）
        // setCustomDensityRange(0.25, 0.45);

        // 追踪已生成的图的哈希值，确保图的唯一性
        std::unordered_set<std::string> generatedGraphs;

        // 随机生成3-10个节点的图进行测试
        std::uniform_int_distribution<int> nodeDist(3, 10);

        // 跟踪不同密度类别的图数量
        int sparseCount = 0;
        int mediumCount = 0;
        int denseCount = 0;
        int customCount = 0;

        // 执行多次测试
        int numTests = 50;
        for (int test = 0; test < numTests; test++)
        {
            // 对每个测试用例创建新的知识库
            KnowledgeBase testKb;

            // 谓词ID
            int pred_E;
            int pred_R;
            int pred_G;
            int pred_uncol;

            // 变量符号
            SymbolId var_x;
            SymbolId var_y;

            // 常量符号
            SymbolId const_a = testKb.addConstant("a");
            SymbolId const_b = testKb.addConstant("b");
            SymbolId const_c = testKb.addConstant("c");
            SymbolId const_d = testKb.addConstant("d");
            SymbolId const_e = testKb.addConstant("e");
            SymbolId const_f = testKb.addConstant("f");
            SymbolId const_g = testKb.addConstant("g");
            SymbolId const_h = testKb.addConstant("h");
            SymbolId const_i = testKb.addConstant("i");
            SymbolId const_j = testKb.addConstant("j");

            // 更新常量列表
            allConstants = {const_a, const_b, const_c, const_d, const_e,
                            const_f, const_g, const_h, const_i, const_j};

            // 随机选择节点数
            int numNodes = nodeDist(rng);

            // 密度类别
            std::string densityCategory;

            // 生成唯一的随机图，确保包含K3完全子图，使用指定的密度类型
            std::vector<std::pair<SymbolId, SymbolId>> edges = generateUniqueRandomGraph(
                numNodes, generatedGraphs, testKb, densityCategory,
                requestedDensityType, customMinDensity, customMaxDensity);

            // 更新计数
            if (densityCategory.find("Sparse") != std::string::npos)
                sparseCount++;
            else if (densityCategory.find("Medium") != std::string::npos)
                mediumCount++;
            else if (densityCategory.find("Dense") != std::string::npos)
                denseCount++;
            else if (densityCategory.find("Custom") != std::string::npos)
                customCount++;

            // 为此图创建文件名
            std::stringstream ss;
            ss << outputDir << "/graph_" << numNodes << "_nodes_" << densityCategory << "_test_" << test << ".txt";
            std::string filename = ss.str();

            // 保存图结构到文件
            saveGraphToFile(edges, numNodes, filename, testKb, densityCategory);

            // 设置两染色问题的知识库并添加图的边
            setupTwoColoringKB(testKb, pred_E, pred_R, pred_G, pred_uncol, var_x, var_y);
            addGraphEdgesToKB(edges, testKb, pred_E);

            // 打印知识库内容
            std::cout << "Random Graph Two-Coloring Knowledge Base (Test " << test
                      << ", " << numNodes << " nodes, " << densityCategory << "):" << std::endl;
            testKb.print();
            std::cout << "Contains K3 complete subgraph: " << (containsK3(edges, numNodes) ? "Yes" : "No") << std::endl;

            // 设置目标：证明图不可着色
            Clause goal;
            goal.addLiteral(Literal(pred_uncol, {}, true));

            // 多次尝试MCTS，直到成功或达到最大尝试次数
            const int MAX_MCTS_ATTEMPTS = 5;
            bool proved = false;
            int mctsAttempt = 0;

            std::ofstream resultsFile(filename, std::ios_base::app);
            if (resultsFile.is_open())
            {
                resultsFile << "\nMCTS Resolution Attempts:" << std::endl;
            }

            while (!proved && mctsAttempt < MAX_MCTS_ATTEMPTS)
            {
                mctsAttempt++;

                // 执行SLI消解，使用统一的数据保存目录
                SLIMCTSProver prover(testKb, goal);
                proved = prover.prove("/home/adrin/Projects/fol-parser/data");

                // 记录此次尝试结果
                std::cout << "Graph with " << numNodes << " nodes, Test " << test
                          << ", MCTS Attempt " << mctsAttempt
                          << ": " << (proved ? "Not two-colorable" : "Two-colorable (or inconclusive)") << std::endl;

                if (resultsFile.is_open())
                {
                    resultsFile << "Attempt " << mctsAttempt << ": "
                                << (proved ? "Not two-colorable" : "Two-colorable (or inconclusive)") << std::endl;
                }

                // 如果证明成功，跳出循环
                if (proved)
                    break;
            }

            // 记录最终结论
            std::string finalResult;
            if (proved)
            {
                finalResult = "Not two-colorable (proved on attempt " + std::to_string(mctsAttempt) + ")";
            }
            else
            {
                finalResult = "Likely two-colorable (not disproven after " + std::to_string(MAX_MCTS_ATTEMPTS) + " attempts)";
            }

            std::cout << "Final conclusion for graph " << test << ": " << finalResult << std::endl;

            if (resultsFile.is_open())
            {
                resultsFile << "\nFinal Coloring Conclusion: " << finalResult << std::endl;
                resultsFile.close();
            }
        }
        // 打印不同密度类别的图数量统计
        std::cout << "\nDensity Distribution Summary:" << std::endl;
        std::cout << "Sparse graphs (density < 0.15): " << sparseCount << " (" << (sparseCount * 100.0 / numTests) << "%)" << std::endl;
        std::cout << "Medium graphs (0.15 <= density < 0.35): " << mediumCount << " (" << (mediumCount * 100.0 / numTests) << "%)" << std::endl;
        std::cout << "Dense graphs (density >= 0.35): " << denseCount << " (" << (denseCount * 100.0 / numTests) << "%)" << std::endl;
        if (customCount > 0)
        {
            std::cout << "Custom density graphs: " << customCount << " (" << (customCount * 100.0 / numTests) << "%)" << std::endl;
        }
    }
}