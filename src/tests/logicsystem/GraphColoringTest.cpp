#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "SLIResolution.h"
#include "BFSStrategy.h"
#include "KnowledgeBase.h"
#include "SLIMCTSProver.h"
#include "SLIHeuristicProver.h"
#include "BiapartiteChecker.h"

namespace LogicSystem
{

    // 辅助函数：提取图信息
    struct GraphInfo
    {
        int nodeCount;
        int edgeCount;
    };

    // GraphInfo extractGraphInfo(const std::string &jsonPath)
    // {
    //     GraphInfo info = {0, 0};
    //     std::ifstream file(jsonPath);
    //     if (file.is_open())
    //     {
    //         json j;
    //         file >> j;

    //         if (j.contains("nodes"))
    //         {
    //             info.nodeCount = j["nodes"].size();
    //         }

    //         if (j.contains("edges"))
    //         {
    //             info.edgeCount = j["edges"].size();
    //         }
    //     }
    //     return info;
    // }
    class GraphColoringTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // 设置知识库的基本谓词
            pred_E = kb.addPredicate("E");         // 边关系
            pred_R = kb.addPredicate("R");         // 红色
            pred_G = kb.addPredicate("G");         // 绿色
            pred_B = kb.addPredicate("B");         // 蓝色
            pred_uncol = kb.addPredicate("uncol"); // 不可着色标记

            // 添加变量
            var_x = kb.addVariable("x");
            var_y = kb.addVariable("y");
        }

        // 从JSON文件加载图并构建知识库
        void loadGraphFromJson(const std::string &filePath)
        {
            std::ifstream file(filePath);
            if (!file.is_open())
            {
                throw std::runtime_error("无法打开文件: " + filePath);
            }

            nlohmann::json data;
            file >> data;
            file.close();

            // 映射关系: CONST0-CONST9 -> a-j
            std::map<std::string, std::string> constMapping = {
                {"CONST0", "a"}, {"CONST1", "b"}, {"CONST2", "c"}, {"CONST3", "d"}, {"CONST4", "e"}, {"CONST5", "f"}, {"CONST6", "g"}, {"CONST7", "h"}, {"CONST8", "i"}, {"CONST9", "j"}};

            // 添加常量（图的顶点）
            for (const auto &node : data["graph"]["nodes"])
            {
                std::string constId = node["id"];
                std::string mappedName = constMapping[constId];
                constantMap[constId] = kb.addConstant(mappedName);
            }

            // 添加边关系
            for (const auto &edge : data["graph"]["edges"])
            {
                for (const auto &literal : edge["literals"])
                {
                    if (literal["predicate"] == "E")
                    {
                        std::string arg1 = literal["arguments"][0];
                        std::string arg2 = literal["arguments"][1];

                        Clause edgeClause;
                        edgeClause.addLiteral(Literal(pred_E, {constantMap[arg1], constantMap[arg2]}, false));
                        kb.addClause(edgeClause);
                    }
                }
            }
        }

        // 添加图着色规则
        void addColoringRules(int numColors)
        {
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

            if (numColors >= 3)
            {
                // 相邻顶点不能同色（蓝色）
                // ¬E(x,y)∨ ¬B(x)∨ ¬B(y)∨ uncol
                Clause adjacentNotSameColorB;
                adjacentNotSameColorB.addLiteral(Literal(pred_E, {var_x, var_y}, true));
                adjacentNotSameColorB.addLiteral(Literal(pred_B, {var_x}, true));
                adjacentNotSameColorB.addLiteral(Literal(pred_B, {var_y}, true));
                adjacentNotSameColorB.addLiteral(Literal(pred_uncol, {}, false));
                kb.addClause(adjacentNotSameColorB);
            }

            // 每个顶点必须着色
            Clause vertexMustBeColored;
            vertexMustBeColored.addLiteral(Literal(pred_R, {var_x}, false));
            vertexMustBeColored.addLiteral(Literal(pred_G, {var_x}, false));

            if (numColors >= 3)
            {
                vertexMustBeColored.addLiteral(Literal(pred_B, {var_x}, false));
            }

            kb.addClause(vertexMustBeColored);
        }

        // 辅助函数：创建搜索策略
        BFSStrategy createStrategy(int maxDepth = 10)
        {
            return BFSStrategy(maxDepth, 60.0, 1024 * 1024 * 100);
        }

        // 重置知识库
        void resetKnowledgeBase()
        {
            kb = KnowledgeBase();
            constantMap.clear();

            // 重新设置谓词
            pred_E = kb.addPredicate("E");
            pred_R = kb.addPredicate("R");
            pred_G = kb.addPredicate("G");
            pred_B = kb.addPredicate("B");
            pred_uncol = kb.addPredicate("uncol");

            // 重新设置变量
            var_x = kb.addVariable("x");
            var_y = kb.addVariable("y");
        }

        KnowledgeBase kb;
        // 谓词ID
        int pred_E;
        int pred_R;
        int pred_G;
        int pred_B;
        int pred_uncol;
        // 变量符号
        SymbolId var_x;
        SymbolId var_y;
        // 常量映射表
        std::map<std::string, SymbolId> constantMap;
    };

    // 测试从目录读取所有图并根据选择的算法进行测试
    TEST_F(GraphColoringTest, BatchTestTwoColoring)
    {
        // 可选参数配置
        //std::vector<std::string> selectedMethods = {"NeuralHeuristic_ALL", "DFS", "MCTS", "NeuralHeuristic_1", "NeuralHeuristic_2"};
        // std::vector<std::string> selectedMethods = {"MCTS"};
        // std::vector<std::string> selectedMethods = {"NeuralHeuristic_ALL"};
        // std::vector<std::string> selectedMethods = {"DFS"};
        //std::vector<std::string> selectedMethods = {"NeuralHeuristic_1", "NeuralHeuristic_2"};
        std::vector<std::string> selectedMethods = {"NeuralHeuristic_1"};
        //std::vector<std::string> selectedMethods = {"NeuralHeuristic_2"};
        int maxTestFiles = 10;
        bool skipBipartite = true;

        std::string dataDir = "/home/jiangguifei01/aiderun/fol-parser/fol-parser/data/newreward";
        std::string resultsDir = "/home/jiangguifei01/aiderun/fol-parser/fol-parser/results";

        // 创建结果目录（如果不存在）
        std::filesystem::create_directories(resultsDir);

        // 获取当前时间，用于文件名
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now = *std::localtime(&time_t_now);

        // 格式化时间戳: YYYY-MM-DD_HH-MM-SS
        std::ostringstream timestamp;
        timestamp << std::put_time(&tm_now, "%Y-%m-%d_%H-%M-%S");

        // 生成包含选择方法和时间戳的文件名
        std::string methodsString;
        for (const auto &method : selectedMethods)
        {
            if (!methodsString.empty())
                methodsString += "_";
            methodsString += method;
        }

        std::string resultFileName = "search_comparison_" + methodsString + "_" + timestamp.str() + ".csv";
        std::string fullResultPath = resultsDir + "/" + resultFileName;

        // 创建CSV文件并写入表头
        std::ofstream resultsFile(fullResultPath);
        resultsFile << "FileName,GraphSize,EdgeDensity,CanTwoColored,Method,Success,VisitedStates,Duration(ms)\n";
        resultsFile.close(); // 先关闭文件，之后每次都追加写入

        int count = 0;
        // 遍历数据目录下的所有JSON文件
        for (const auto &entry : std::filesystem::directory_iterator(dataDir))
        {
            if (entry.path().extension() == ".json")
            {
                std::string filename = entry.path().filename().string();
                std::cout << "Testing file: " << filename << std::endl;

                // 通过BFS判断这个图是否可二染色
                BipartiteChecker checker;
                checker.loadGraphFromJson(entry.path().string());
                bool canTwoColored = checker.isBipartite();
                std::cout << "Result by BFS can be two colored: " << canTwoColored << std::endl;

                // 如果设置了跳过二分图且当前图是二分图，则跳过此图
                if (skipBipartite && canTwoColored)
                    continue;

                // 获取图的大小（节点数）
                int graphSize = checker.getNumNodes();
                int edgeSize = checker.getNumEdges();
                double edgeDensity = checker.getEdgeDensity();
                std::cout << "Graph Node Size " << graphSize << " Graph Edge SIze " << edgeSize << " Graph Density " << edgeDensity << std::endl;

                // 只测试选定的搜索方法
                for (const auto &method : selectedMethods)
                {
                    std::cout << "Testing with method: " << method << std::endl;

                    // 重置知识库以准备新的测试
                    resetKnowledgeBase();

                    // 从JSON文件加载图
                    loadGraphFromJson(entry.path().string());

                    // 添加图着色规则（2着色）
                    addColoringRules(2);

                    // 设置目标：证明图不可2着色
                    Clause goal;
                    goal.addLiteral(Literal(pred_uncol, {}, true));

                    // 根据方法选择对应的证明器并执行证明
                    SearchResult searchResult;

                    if (method == "NeuralHeuristic_ALL")
                    {
                        SLIHeuristicProver prover(kb, goal, 80000);
                        // 设置第一阶段配置
                        prover.setPhase1Config(
                            "python3",
                            "../../../neural_network/neural_server/neural_server.py",
                            "../../../neural_network/first_stage_model.pth",
                            "../../../neural_network/unified_tokenizer.pkl");

                        // 设置第二阶段配置
                        prover.setPhase2Config(
                            "../../../neural_network/neural_server/neural_server2.py",
                            "../../../neural_network/second_stage_model.pth",
                            "../../../neural_network/unified_tokenizer.pkl");
                        prover.setExperimentMode("both_phases"); // 使用两个阶段
                        searchResult = prover.prove("");
                    }
                    else if (method == "NeuralHeuristic_1")
                    {
                        SLIHeuristicProver prover(kb, goal, 80000);
                        // 设置第一阶段配置
                        prover.setPhase1Config(
                            "python3",
                            "../../../neural_network/neural_server/neural_server.py",
                            "../../../neural_network/first_stage_model.pth",
                            "../../../neural_network/unified_tokenizer.pkl");
                        prover.setExperimentMode("phase1_only"); // 只使用第一阶段
                        searchResult = prover.prove("");
                    }
                    else if (method == "NeuralHeuristic_2")
                    {
                        SLIHeuristicProver prover(kb, goal, 80000);
                        std::cout << "NeuralHeuristic_2 " <<std::endl;
                        // 设置第二阶段配置
                        prover.setPhase2Config(
                            "../../../neural_network/neural_server/neural_server2.py",
                            "../../../neural_network/second_stage_model.pth",
                            "../../../neural_network/unified_tokenizer.pkl");
                        prover.setExperimentMode("phase2_only"); // 只使用第二阶段
                        searchResult = prover.prove("");
                    }
                    else if (method == "DFS")
                    {
                        searchResult = SLIResolution::prove(kb, goal);
                    }
                    else if (method == "MCTS")
                    {
                        SLIMCTSProver prover(kb, goal);
                        searchResult = prover.prove_search_result("");
                    }
                    else
                    {
                        std::cerr << "Unknown method: " << method << ", skipping..." << std::endl;
                        continue;
                    }

                    // 每次测试完一个方法后立即保存结果到CSV（追加模式）
                    {
                        // 使用追加模式打开文件
                        std::ofstream appendResultsFile(fullResultPath, std::ios::app);
                        if (!appendResultsFile.is_open())
                        {
                            std::cerr << "Failed to open results file for appending!" << std::endl;
                            // 确保结果目录存在
                            std::filesystem::create_directories(resultsDir);
                            appendResultsFile.open(fullResultPath, std::ios::app);
                            // 如果文件是新创建的，需要写入表头
                            if (appendResultsFile.tellp() == 0)
                            {
                                appendResultsFile << "FileName,GraphSize,EdgeDensity,CanTwoColored,Method,Success,VisitedStates,Duration(ms)\n";
                            }
                        }

                        appendResultsFile << filename << ","
                                          << graphSize << ","
                                          << edgeDensity << ","
                                          << (canTwoColored ? "Yes" : "No") << ","
                                          << method << ","
                                          << (searchResult.success ? "Yes" : "No") << ","
                                          << searchResult.visitedStates << ","
                                          << searchResult.duration << "\n";

                        appendResultsFile.close(); // 关闭文件以确保数据被保存
                    }

                    // 输出结果
                    std::cout << "Graph " << filename << " using " << method
                              << (searchResult.success ? " proved cannot" : " failed to prove cannot")
                              << " be two-colored." << std::endl;
                    std::cout << "Visited States: " << searchResult.visitedStates
                              << ", Duration: " << searchResult.duration << "ms" << std::endl;
                }

                // if (++count >= maxTestFiles)
                //     break; // 限制测试文件数量
            }
        }

        std::cout << "Test results saved to: " << fullResultPath << std::endl;
    }

    // 测试特定图文件的3着色
    TEST_F(GraphColoringTest, TestSpecificGraphThreeColoring)
    {
        std::string filePath = "/home/jiangguifei01/aiderun/fol-parser/fol-parser/data/graphs/petersen.json"; // 你可以指定特定的图文件

        // 从JSON文件加载图
        loadGraphFromJson(filePath);

        // 添加图着色规则（3着色）
        addColoringRules(3);

        // 打印知识库内容
        std::cout << "Testing 3-coloring for specific graph:" << std::endl;
        kb.print();

        // 设置目标：证明图不可3着色
        Clause goal;
        goal.addLiteral(Literal(pred_uncol, {}, true));

        // 执行SLI消解
        SLIHeuristicProver prover(kb, goal);
        bool proved = prover.prove("results/specific_graph_3coloring").success;

        // 验证结果 - 这里我们断言特定的结果
        std::cout << "Specific graph " << (proved ? "cannot" : "can")
                  << " be three-colored." << std::endl;

        // 如果你知道这个图是否可以3着色，可以添加相应的断言
        // EXPECT_TRUE(proved); // 如果图不可3着色
        // EXPECT_FALSE(proved); // 如果图可以3着色
    }

    // 示例：构建并测试完整的彼得森图
    TEST_F(GraphColoringTest, PetersenGraphThreeColoring)
    {
        // 彼得森图是一个特殊的图，需要3着色
        // 手动构建彼得森图以便进行比较

        // 添加常量（图的顶点）
        SymbolId const_a = kb.addConstant("a");
        SymbolId const_b = kb.addConstant("b");
        SymbolId const_c = kb.addConstant("c");
        SymbolId const_d = kb.addConstant("d");
        SymbolId const_e = kb.addConstant("e");
        SymbolId const_f = kb.addConstant("f");
        SymbolId const_g = kb.addConstant("g");
        SymbolId const_h = kb.addConstant("h");
        SymbolId const_i = kb.addConstant("i");
        SymbolId const_j = kb.addConstant("j");

        // 彼得森图的边
        std::vector<std::pair<SymbolId, SymbolId>> edges = {
            {const_a, const_b}, {const_b, const_c}, {const_c, const_d}, {const_d, const_e}, {const_e, const_a}, {const_f, const_h}, {const_g, const_i}, {const_h, const_j}, {const_i, const_f}, {const_j, const_g}, {const_a, const_f}, {const_b, const_g}, {const_c, const_h}, {const_d, const_i}, {const_e, const_j}};

        for (const auto &edge : edges)
        {
            Clause edgeClause;
            edgeClause.addLiteral(Literal(pred_E, {edge.first, edge.second}, false));
            kb.addClause(edgeClause);
            Clause edgeClauseReverse;
            edgeClauseReverse.addLiteral(Literal(pred_E, {edge.second, edge.first}, false));
            kb.addClause(edgeClauseReverse);
        }

        // 添加图着色规则（3着色）
        addColoringRules(3);

        // 打印知识库内容
        std::cout << "Petersen Graph Three-Coloring Knowledge Base:" << std::endl;
        kb.print();

        // 设置目标：证明图不可3着色
        Clause goal;
        goal.addLiteral(Literal(pred_uncol, {}, true));

        // 执行SLI消解
        SLIHeuristicProver prover(kb, goal);
        bool proved = prover.prove("results/petersen_3coloring").success;

        // 验证结果 - 彼得森图可以3着色
        EXPECT_FALSE(proved);
        std::cout << "SLI Resolution confirms that Petersen graph can be three-colored." << std::endl;
    }
} // namespace LogicSystem