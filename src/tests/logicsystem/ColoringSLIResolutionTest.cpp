#include <gtest/gtest.h>
#include "SLIResolution.h"
#include "BFSStrategy.h"
#include "KnowledgeBase.h"

namespace LogicSystem {

class ColoringSLIResolutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置知识库的基本谓词
        pred_E = kb.addPredicate("E");    // 边关系
        pred_R = kb.addPredicate("R");    // 红色
        pred_G = kb.addPredicate("G");    // 绿色
        pred_B = kb.addPredicate("B");    // 蓝色
        pred_uncol = kb.addPredicate("uncol"); // 不可着色标记

        // 添加常量（图的顶点）
        const_a = kb.addConstant("a");
        const_b = kb.addConstant("b");
        const_c = kb.addConstant("c");
        const_d = kb.addConstant("d");

        // 添加变量
        var_x = kb.addVariable("x");
        var_y = kb.addVariable("y");
    }

    // 辅助函数：创建搜索策略
    BFSStrategy createStrategy(int maxDepth = 10) {
        return BFSStrategy(maxDepth, 60.0, 1024 * 1024 * 100);
    }

    KnowledgeBase kb;
    // 谓词ID
    int pred_E;
    int pred_R;
    int pred_G;
    int pred_B;
    int pred_uncol;
    // 常量符号
    SymbolId const_a;
    SymbolId const_b;
    SymbolId const_c;
    SymbolId const_d;
    // 变量符号
    SymbolId var_x;
    SymbolId var_y;
};

TEST_F(ColoringSLIResolutionTest, K3GraphCannotBeTwoColored) {
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

    // 4. K3图的边
    std::vector<std::pair<SymbolId, SymbolId>> edges = {
        {const_a, const_b},
        {const_b, const_c},
        {const_a, const_c}
    };

    for (const auto& edge : edges) {
        Clause edgeClause;
        edgeClause.addLiteral(Literal(pred_E, {edge.first, edge.second}, false));
        kb.addClause(edgeClause);
        /*Clause edgeClauseReverse;
        edgeClauseReverse.addLiteral(Literal(pred_E, {edge.second, edge.first}, false));
        kb.addClause(edgeClauseReverse);*/
    }

    // 打印知识库内容
    std::cout << "K3 Two-Coloring Knowledge Base:" << std::endl;
    kb.print();

    // 设置目标：证明图不可着色
    Clause goal;
    goal.addLiteral(Literal(pred_uncol, {}, true));

    // 执行SLI消解
    auto strategy = createStrategy();
    bool proved = SLIResolution::prove(kb, goal, strategy);

    // 验证结果
    EXPECT_TRUE(proved);
    std::cout << "SLI Resolution proves that K3 graph cannot be two-colored." << std::endl;
    std::cout << "States searched: " << strategy.getSearchedStates() << std::endl;
}

TEST_F(ColoringSLIResolutionTest, K4GraphCannotBeThreeColored) {
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

    // 3. 相邻顶点不能同色（蓝色）
    // ¬E(x,y)∨ ¬B(x)∨ ¬B(y)∨ uncol
    Clause adjacentNotSameColorB;
    adjacentNotSameColorB.addLiteral(Literal(pred_E, {var_x, var_y}, true));
    adjacentNotSameColorB.addLiteral(Literal(pred_B, {var_x}, true));
    adjacentNotSameColorB.addLiteral(Literal(pred_B, {var_y}, true));
    adjacentNotSameColorB.addLiteral(Literal(pred_uncol, {}, false));
    kb.addClause(adjacentNotSameColorB);

    // 4. 每个顶点必须着色（红色或绿色）
    // R(x) ∨ G(x) ∨ B(x)
    Clause vertexMustBeColored;
    vertexMustBeColored.addLiteral(Literal(pred_R, {var_x}, false));
    vertexMustBeColored.addLiteral(Literal(pred_G, {var_x}, false));
    vertexMustBeColored.addLiteral(Literal(pred_B, {var_x}, false));
    kb.addClause(vertexMustBeColored);

    // 5. K4图的边
    std::vector<std::pair<SymbolId, SymbolId>> edges = {
        {const_a, const_b},
        {const_b, const_c},
        {const_a, const_c},
        {const_a, const_d},
        {const_b, const_d},
        {const_c, const_d}
    };

    for (const auto& edge : edges) {
        Clause edgeClause;
        edgeClause.addLiteral(Literal(pred_E, {edge.first, edge.second}, false));
        kb.addClause(edgeClause);
        /*Clause edgeClauseReverse;
        edgeClauseReverse.addLiteral(Literal(pred_E, {edge.second, edge.first}, false));
        kb.addClause(edgeClauseReverse);*/
    }

    // 打印知识库内容
    std::cout << "K4 Three-Coloring Knowledge Base:" << std::endl;
    kb.print();

    // 设置目标：证明图不可着色
    Clause goal;
    goal.addLiteral(Literal(pred_uncol, {}, true));

    // 执行SLI消解
    auto strategy = createStrategy();
    bool proved = SLIResolution::prove(kb, goal, strategy);

    // 验证结果
    EXPECT_TRUE(proved);
    std::cout << "SLI Resolution proves that K4 graph cannot be three-colored." << std::endl;
    std::cout << "States searched: " << strategy.getSearchedStates() << std::endl;
}

// 可以添加更多测试用例，比如K4的三染色问题等

} // namespace LogicSystem