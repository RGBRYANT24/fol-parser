#include <gtest/gtest.h>
#include "SLIResolution.h"
#include "BFSStrategy.h"
#include "KnowledgeBase.h"

namespace LogicSystem {

class SLIResolutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置知识库的基本谓词和符号
        pred_P = kb.addPredicate("P");
        pred_Q = kb.addPredicate("Q");
        pred_R = kb.addPredicate("R");
        const_a = kb.addConstant("a");
        const_b = kb.addConstant("b");
        const_c = kb.addConstant("c");
        var_x = kb.addVariable("x");
        var_y = kb.addVariable("y");
        var_z = kb.addVariable("z");
    }

    // 辅助函数：创建一个基本的BFS策略
    BFSStrategy createStrategy(int maxDepth = 5) {
        return BFSStrategy(maxDepth, 60.0, 1024 * 1024 * 100);
    }

    KnowledgeBase kb;
    int pred_P;
    int pred_Q;
    int pred_R;
    SymbolId const_a;
    SymbolId const_b;
    SymbolId const_c;
    SymbolId var_x;
    SymbolId var_y;
    SymbolId var_z;
};

TEST_F(SLIResolutionTest, SimpleOneStepResolution) {
    // P(x,a)
    Clause kb_clause;
    kb_clause.addLiteral(Literal(pred_P, {var_x, const_a}, false));
    kb.addClause(kb_clause);

    // ¬P(b,a)
    Clause goal;
    goal.addLiteral(Literal(pred_P, {const_b, const_a}, true));

    auto strategy = createStrategy();
    bool result = SLIResolution::prove(kb, goal, strategy);
    
    EXPECT_TRUE(result);
    EXPECT_LT(strategy.getSearchedStates(), 10);
}

// ... 其他测试用例类似修改 ...

TEST_F(SLIResolutionTest, BacktrackingTest) {
    Clause c1, c2, c3;
    c1.addLiteral(Literal(pred_P, {var_x, var_y}, false));
    c1.addLiteral(Literal(pred_Q, {var_x}, false));
    c2.addLiteral(Literal(pred_Q, {var_x}, true));
    c2.addLiteral(Literal(pred_R, {var_x}, false));
    c3.addLiteral(Literal(pred_R, {var_x}, true));
    kb.addClause(c1);
    kb.addClause(c2);
    kb.addClause(c3);

    Clause goal;
    goal.addLiteral(Literal(pred_P, {const_a, const_b}, true));

    auto strategy = createStrategy();
    bool result = SLIResolution::prove(kb, goal, strategy);
    
    EXPECT_TRUE(result);
    EXPECT_GT(strategy.getSearchedStates(), 1);
}

} // namespace LogicSystem