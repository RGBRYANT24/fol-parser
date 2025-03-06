TEST_F(RandomGraphColoringTest, RandomGraphTwoColoringMCTS)
{
    // 设置输出目录
    std::string outputDir = "/home/adrin/Projects/fol-parser/data/random_graphs";

    // 追踪已生成的图的哈希值，确保图的唯一性
    std::unordered_set<std::string> generatedGraphs;

    // 随机生成3-10个节点的图进行测试
    std::uniform_int_distribution<int> nodeDist(3, 10);

    // 执行多次测试
    int numTests = 5;
    for (int test = 0; test < numTests; test++)
    {
        // 对每个测试用例创建新的知识库
        KnowledgeBase testKb;
        
        // 重新添加所有谓词和变量
        pred_E = testKb.addPredicate("E");
        pred_R = testKb.addPredicate("R");
        pred_G = testKb.addPredicate("G");
        pred_B = testKb.addPredicate("B");
        pred_uncol = testKb.addPredicate("uncol");
        
        // 重新添加常量
        const_a = testKb.addConstant("a");
        const_b = testKb.addConstant("b");
        const_c = testKb.addConstant("c");
        const_d = testKb.addConstant("d");
        const_e = testKb.addConstant("e");
        const_f = testKb.addConstant("f");
        const_g = testKb.addConstant("g");
        const_h = testKb.addConstant("h");
        const_i = testKb.addConstant("i");
        const_j = testKb.addConstant("j");
        
        // 更新常量列表
        allConstants = {const_a, const_b, const_c, const_d, const_e,
                      const_f, const_g, const_h, const_i, const_j};
        
        // 重新添加变量
        var_x = testKb.addVariable("x");
        var_y = testKb.addVariable("y");

        // 随机选择节点数
        int numNodes = nodeDist(rng);

        // 生成唯一的随机图
        std::vector<std::pair<SymbolId, SymbolId>> edges = generateUniqueRandomGraph(numNodes, generatedGraphs);

        // 为此图创建文件名
        std::stringstream ss;
        ss << outputDir << "/graph_" << numNodes << "_nodes_test_" << test << ".txt";
        std::string filename = ss.str();

        // 设置两染色问题的规则并添加图的边
        
        // 1. 相邻顶点不能同色（红色）
        Clause adjacentNotSameColorR;
        adjacentNotSameColorR.addLiteral(Literal(pred_E, {var_x, var_y}, true));
        adjacentNotSameColorR.addLiteral(Literal(pred_R, {var_x}, true));
        adjacentNotSameColorR.addLiteral(Literal(pred_R, {var_y}, true));
        adjacentNotSameColorR.addLiteral(Literal(pred_uncol, {}, false));
        testKb.addClause(adjacentNotSameColorR);

        // 2. 相邻顶点不能同色（绿色）
        Clause adjacentNotSameColorG;
        adjacentNotSameColorG.addLiteral(Literal(pred_E, {var_x, var_y}, true));
        adjacentNotSameColorG.addLiteral(Literal(pred_G, {var_x}, true));
        adjacentNotSameColorG.addLiteral(Literal(pred_G, {var_y}, true));
        adjacentNotSameColorG.addLiteral(Literal(pred_uncol, {}, false));
        testKb.addClause(adjacentNotSameColorG);

        // 3. 每个顶点必须着色（红色或绿色）
        Clause vertexMustBeColored;
        vertexMustBeColored.addLiteral(Literal(pred_R, {var_x}, false));
        vertexMustBeColored.addLiteral(Literal(pred_G, {var_x}, false));
        testKb.addClause(vertexMustBeColored);
        
        // 添加图的边到知识库
        for (const auto &edge : edges)
        {
            Clause edgeClause;
            edgeClause.addLiteral(Literal(pred_E, {edge.first, edge.second}, false));
            testKb.addClause(edgeClause);

            Clause edgeClauseReverse;
            edgeClauseReverse.addLiteral(Literal(pred_E, {edge.second, edge.first}, false));
            testKb.addClause(edgeClauseReverse);
        }

        // 打印知识库内容
        std::cout << "Random Graph Two-Coloring Knowledge Base (Test " << test
                  << ", " << numNodes << " nodes):" << std::endl;
        testKb.print();

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
}