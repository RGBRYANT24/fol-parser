#include <gtest/gtest.h>
#include "Clause.h"
#include "Literal.h"
#include "KnowledgeBase.h"
namespace LogicSystem
{
    class ClauseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up predicates
        pred_R = kb.addPredicate("R"); // unary predicate
        pred_P = kb.addPredicate("P"); // binary predicate

        // Set up constants
        const_a = kb.addConstant("a");
        const_b = kb.addConstant("b");

        // Set up variables
        var_x = kb.addVariable("x");
        var_y = kb.addVariable("y");
        var_z = kb.addVariable("z");
    }

    KnowledgeBase kb;
    int pred_R;
    int pred_P;
    SymbolId const_a;
    SymbolId const_b;
    SymbolId var_x;
    SymbolId var_y;
    SymbolId var_z;
};

TEST_F(ClauseTest, FindLiteralIndex) {
    Clause clause;
    
    // 添加 R(x)
    Literal lit1(pred_R, {var_x}, false);
    clause.addLiteral(lit1);
    
    // 添加 ¬R(y)
    Literal lit2(pred_R, {var_y}, true);
    clause.addLiteral(lit2);
    
    // 添加 P(x,y)
    Literal lit3(pred_P, {var_x, var_y}, false);
    clause.addLiteral(lit3);

    // 测试找到第一个文字
    EXPECT_EQ(clause.findLiteralIndex(lit1), 0);
    
    // 测试找到第二个文字
    EXPECT_EQ(clause.findLiteralIndex(lit2), 1);
    
    // 测试找到第三个文字
    EXPECT_EQ(clause.findLiteralIndex(lit3), 2);
    
    // 测试找不到的情况
    Literal lit_not_exist(pred_R, {var_z}, false); // R(z)
    EXPECT_EQ(clause.findLiteralIndex(lit_not_exist), -1);
    
    // 测试相同谓词不同极性
    Literal lit_same_pred_diff_pol(pred_R, {var_x}, true); // ¬R(x)
    EXPECT_EQ(clause.findLiteralIndex(lit_same_pred_diff_pol), -1);
    
    // 测试相同谓词相同极性不同变量
    Literal lit_same_pred_diff_var(pred_R, {var_y}, false); // R(y)
    EXPECT_EQ(clause.findLiteralIndex(lit_same_pred_diff_var), -1);
    
    // 测试常量
    Literal lit_with_const(pred_R, {const_a}, false); // R(a)
    clause.addLiteral(lit_with_const);
    EXPECT_EQ(clause.findLiteralIndex(lit_with_const), 3);
}
}

