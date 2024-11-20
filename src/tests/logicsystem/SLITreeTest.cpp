#include <gtest/gtest.h>
#include "SLITree.h"
#include "KnowledgeBase.h"

using namespace LogicSystem;

class SLITreeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 设置知识库
        pred_P = kb.addPredicate("P"); // 二元谓词 P
        pred_Q = kb.addPredicate("Q"); // 一元谓词 Q
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
    std::cout << "Print Empty Tree with a single root" << std::endl;
    tree.print_tree(kb);

    // 创建测试文字
    // P(x,a)
    Literal l1(pred_P, {var_x, const_a}, false);
    // Q(b)
    Literal l2(pred_Q, {const_b}, false);
    // ~P(b,y)
    Literal l3(pred_P, {const_b, var_y}, true);
    // P(b,a)
    Literal l4(pred_P, {const_b, const_a}, false);
    // ~Q(b)
    Literal l5(pred_Q, {const_b}, true);

    // 测试场景1：基本添加
    std::cout << "\nTest Scenario 1: Basic Addition" << std::endl;
    Clause c1;
    c1.addLiteral(l1);
    c1.addLiteral(l2);
    std::vector<std::shared_ptr<SLINode>> active_nodes = tree.add_node(c1, Literal(), false, tree.getRoot());
    ASSERT_EQ(active_nodes.size(), 2);
    tree.print_tree(kb);

    // 测试场景2：与P(x,a)消解
    std::cout << "\nTest Scenario 2: Resolution with P(x,a)" << std::endl;
    Clause c2;
    c2.addLiteral(l3);
    c2.addLiteral(l2); // 添加Q(b)使子句不为空
    auto nodes2 = tree.add_node(c2, l3, true, active_nodes[0]);
    ASSERT_FALSE(nodes2.empty());
    tree.print_tree(kb);

    // 测试场景3：MGU失败的情况
    std::cout << "\nTest Scenario 3: MGU Failure" << std::endl;
    Clause c3;
    c3.addLiteral(l4);  // P(b,a)
    auto nodes3 = tree.add_node(c3, l4, false, nodes2[0]);  // 尝试与Q(b)消解
    ASSERT_TRUE(nodes3.empty());  // 应该失败因为不能统一
    tree.print_tree(kb);

    // // 测试场景4：多重消解
    // std::cout << "\nTest Scenario 4: Multiple Resolution" << std::endl;
    // Clause c4;
    // c4.addLiteral(l5);  // ~Q(b)
    // auto nodes4 = tree.add_node(c4, l2, true, active_nodes[1]);  // 与Q(b)消解
    // ASSERT_FALSE(nodes4.empty());
    // tree.print_tree(kb);

    // // 测试场景5：空子句的情况
    // std::cout << "\nTest Scenario 5: Empty Clause" << std::endl;
    // Clause empty_clause;
    // auto nodes5 = tree.add_node(empty_clause, Literal(), false, tree.getRoot());
    // ASSERT_TRUE(nodes5.empty());

    // // 测试场景6：单个文字的消解
    // std::cout << "\nTest Scenario 6: Single Literal Resolution" << std::endl;
    // Clause c6;
    // c6.addLiteral(l5);  // ~Q(b)
    // auto nodes6 = tree.add_node(c6, l2, true, active_nodes[1]);
    // ASSERT_FALSE(nodes6.empty());
    // tree.print_tree(kb);
}

// // 测试撤销操作
// TEST_F(SLITreeTest, UndoOperation)
// {
//     SLITree tree(kb);

//     // P(x,a)
//     Literal l1(pred_P, {var_x, const_a}, false);
//     // Q(b)
//     Literal l2(pred_Q, {const_b}, false);

//     // 添加初始节点
//     Clause c1;
//     c1.addLiteral(l1);
//     c1.addLiteral(l2);
//     auto nodes1 = tree.add_node(c1, Literal(), false, tree.getRoot());

//     std::cout << "Before Undo:" << std::endl;
//     tree.print_tree(kb);

//     // 执行撤销
//     bool undo_result = tree.undo();
//     ASSERT_TRUE(undo_result);

//     std::cout << "After Undo:" << std::endl;
//     tree.print_tree(kb);

//     // 验证树恢复到之前状态
//     ASSERT_EQ(tree.getDepthMap().size(), 1);  // 只有根节点层
// }

// // 测试深度限制
// TEST_F(SLITreeTest, DepthLimit)
// {
//     SLITree tree(kb, 2);  // 设置最大深度为2

//     // P(x,a)
//     Literal l1(pred_P, {var_x, const_a}, false);
//     // ~P(b,y)
//     Literal l3(pred_P, {const_b, var_y}, true);

//     // 添加第一层节点
//     Clause c1;
//     c1.addLiteral(l1);
//     auto nodes1 = tree.add_node(c1, Literal(), false, tree.getRoot());
//     ASSERT_FALSE(nodes1.empty());

//     // 添加第二层节点
//     Clause c2;
//     c2.addLiteral(l3);
//     auto nodes2 = tree.add_node(c2, l1, true, nodes1[0]);
//     ASSERT_FALSE(nodes2.empty());

//     // 尝试添加第三层节点（应该失败）
//     auto nodes3 = tree.add_node(c1, l3, false, nodes2[0]);
//     ASSERT_TRUE(nodes3.empty());
// }