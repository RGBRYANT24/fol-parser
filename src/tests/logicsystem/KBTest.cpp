#include <gtest/gtest.h>
#include "KnowledgeBase.h"

namespace LogicSystem {

class KnowledgeBaseTest : public ::testing::Test {
protected:
    KnowledgeBase kb;
};

TEST_F(KnowledgeBaseTest, AddAndRetrieveVariable) {
    SymbolId varX = kb.addVariable("X");
    EXPECT_EQ(kb.getSymbolName(varX), "X");
    EXPECT_TRUE(kb.isVariable(varX));
}

TEST_F(KnowledgeBaseTest, AddAndRetrieveConstant) {
    SymbolId constA = kb.addConstant("a");
    EXPECT_EQ(kb.getSymbolName(constA), "a");
    EXPECT_FALSE(kb.isVariable(constA));
}

TEST_F(KnowledgeBaseTest, AddAndRetrievePredicate) {
    int predId = kb.addPredicate("likes");
    EXPECT_EQ(kb.getPredicateName(predId), "likes");
}

TEST_F(KnowledgeBaseTest, MultipleSymbols) {
    SymbolId varX = kb.addVariable("X");
    SymbolId varY = kb.addVariable("Y");
    SymbolId constA = kb.addConstant("apple");
    SymbolId constB = kb.addConstant("banana");
    
    EXPECT_EQ(kb.getSymbolName(varX), "X");
    EXPECT_EQ(kb.getSymbolName(varY), "Y");
    EXPECT_EQ(kb.getSymbolName(constA), "apple");
    EXPECT_EQ(kb.getSymbolName(constB), "banana");
    
    EXPECT_TRUE(kb.isVariable(varX));
    EXPECT_TRUE(kb.isVariable(varY));
    EXPECT_FALSE(kb.isVariable(constA));
    EXPECT_FALSE(kb.isVariable(constB));
}

TEST_F(KnowledgeBaseTest, MultiplePredVars) {
    int pred1 = kb.addPredicate("likes");
    int pred2 = kb.addPredicate("eats");
    
    EXPECT_EQ(kb.getPredicateName(pred1), "likes");
    EXPECT_EQ(kb.getPredicateName(pred2), "eats");
}

}  // namespace LogicSystem

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}