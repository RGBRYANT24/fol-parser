#include <gtest/gtest.h>
#include "SLITree.h"
#include "KnowledgeBase.h"

using namespace LogicSystem;

class SLITreeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置知识库
        pred_P = kb.addPredicate("P");       // 二元谓词 P
        pred_Q = kb.addPredicate("Q");       // 一元谓词 Q
        const_a = kb.addConstant("a");
        const_b = kb.addConstant("b");
        var_x = kb.addVariable("x");
        var_y = kb.addVariable("y");
    }

    KnowledgeBase kb;
    int pred_P;
    int pred_Q;
    SymbolId const_a;
    SymbolId const_b;
    SymbolId var_x;
    SymbolId var_y;
};

TEST_F(SLITreeTest, AddNode)
{
    SLITree tree(kb);
    //打印空树信息
    std::cout << "Print Empty Tree with a single root" << std::endl;
    tree.print_tree(kb);

    // 创建测试文字
    // P(x,a)
    Literal l1(pred_P, {var_x, const_a}, false);
    // Q(b)
    Literal l2(pred_Q, {const_b}, false);
    // ~P(b,y)
    Literal l3(pred_P, {const_b, var_y}, true);

    //创建测试子句、
    //P(x, a) ∨ Q(b)
    Clause c1;
    c1.addLiteral(l1);
    c1.addLiteral(l2);

    Clause c2;
    c2.addLiteral(l3);

    // 测试添加根节点
    auto node1 = tree.add_node(c1, Literal(), false, tree.getRoot());
    tree.print_tree(kb);
}