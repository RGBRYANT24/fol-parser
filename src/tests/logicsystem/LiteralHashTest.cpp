#include <gtest/gtest.h>
#include "Literal.h"
#include "KnowledgeBase.h"
#include <iostream>

namespace LogicSystem {
namespace testing {

class LiteralHashTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 添加谓词
        pred1 = kb.addPredicate("likes");      // likes(x, y)
        pred2 = kb.addPredicate("follows");    // follows(x, y)
        
        // 添加变量
        var1 = kb.addVariable("x");
        var2 = kb.addVariable("y");
        var3 = kb.addVariable("z");
        
        // 添加常量
        const1 = kb.addConstant("alice");
        const2 = kb.addConstant("bob");
    }

    void PrintLiteralInfo(const Literal& lit, const std::string& label) {
        std::cout << label << ": " << lit.toString(kb) << " (hash: " << lit.hash() << ")\n";
    }

    KnowledgeBase kb;
    int pred1, pred2;
    SymbolId var1, var2, var3, const1, const2;
};

TEST_F(LiteralHashTest, SamePredicateDifferentVariables) {
    Literal lit1(pred1, {var1, var2}, false);  // likes(x, y)
    Literal lit2(pred1, {var2, var3}, false);  // likes(y, z)
    
    PrintLiteralInfo(lit1, "lit1");
    PrintLiteralInfo(lit2, "lit2");
    
    EXPECT_NE(lit1.hash(), lit2.hash());
}

TEST_F(LiteralHashTest, DifferentPredicateSameVariables) {
    Literal lit1(pred1, {var1, var2}, false);  // likes(x, y)
    Literal lit2(pred2, {var1, var2}, false);  // follows(x, y)
    
    PrintLiteralInfo(lit1, "lit1");
    PrintLiteralInfo(lit2, "lit2");
    
    EXPECT_NE(lit1.hash(), lit2.hash());
}

TEST_F(LiteralHashTest, DifferentPredicateVariableConstant) {
    Literal lit1(pred1, {var1, const1}, false);    // likes(x, alice)
    Literal lit2(pred2, {const2, var1}, false);    // follows(bob, x)
    
    PrintLiteralInfo(lit1, "lit1");
    PrintLiteralInfo(lit2, "lit2");
    
    EXPECT_NE(lit1.hash(), lit2.hash());
}

TEST_F(LiteralHashTest, SamePredicateSameVariables) {
    Literal lit1(pred1, {var1, var2}, false);  // likes(x, y)
    Literal lit2(pred1, {var1, var2}, false);  // likes(x, y)
    
    PrintLiteralInfo(lit1, "lit1");
    PrintLiteralInfo(lit2, "lit2");
    
    EXPECT_EQ(lit1.hash(), lit2.hash());
}

TEST_F(LiteralHashTest, NegationAffectsHash) {
    Literal lit1(pred1, {var1, var2}, false);  // likes(x, y)
    Literal lit2(pred1, {var1, var2}, true);   // !likes(x, y)
    
    PrintLiteralInfo(lit1, "lit1");
    PrintLiteralInfo(lit2, "lit2");
    
    EXPECT_NE(lit1.hash(), lit2.hash());
}

TEST_F(LiteralHashTest, ArgumentOrderAffectsHash) {
    Literal lit1(pred1, {var1, var2}, false);  // likes(x, y)
    Literal lit2(pred1, {var2, var1}, false);  // likes(y, x)
    
    PrintLiteralInfo(lit1, "lit1");
    PrintLiteralInfo(lit2, "lit2");
    
    EXPECT_NE(lit1.hash(), lit2.hash());
}

TEST_F(LiteralHashTest, EmptyLiteralHash) {
    Literal empty1;  // 空文字
    Literal empty2;  // 另一个空文字
    Literal nonEmpty(pred1, {var1, var2}, false);
    
    std::cout << "empty1 hash: " << empty1.hash() << "\n";
    std::cout << "empty2 hash: " << empty2.hash() << "\n";
    PrintLiteralInfo(nonEmpty, "nonEmpty");
    
    EXPECT_EQ(empty1.hash(), empty2.hash());
    EXPECT_NE(empty1.hash(), nonEmpty.hash());
}

}  // namespace testing
}  // namespace LogicSystem