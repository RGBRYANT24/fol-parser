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
        pred_R = kb.addPredicate("R");
        // pred_E = kb.addPredicate("E"); // 二元谓词 E
        pred_G = kb.addPredicate("G"); // 一元谓词 G
        pred_A = kb.addPredicate("A"); // 一元谓词 A
        pred_B = kb.addPredicate("B"); // 一元谓词 B
        pred_C = kb.addPredicate("C"); // 一元谓词 C
        pred_D = kb.addPredicate("D"); // 一元谓词 D
        pred_E = kb.addPredicate("E"); // 一元谓词 E
        pred_F = kb.addPredicate("F"); // 一元谓词 F
        const_a = kb.addConstant("a");
        const_b = kb.addConstant("b");
        const_c = kb.addConstant("c");
        var_x = kb.addVariable("x");
        var_y = kb.addVariable("y");
    }

    KnowledgeBase kb;
    int pred_P;
    int pred_Q;
    int pred_R;
    // int pred_E;
    int pred_G;
    int pred_A;
    int pred_B;
    int pred_C;
    int pred_D;
    int pred_E;
    int pred_F;
    SymbolId const_a;
    SymbolId const_b;
    SymbolId const_c;
    SymbolId var_x;
    SymbolId var_y;
};

TEST_F(SLITreeTest, AddNodeWith2SamePredicate)
{
    SLITree tree(kb);

    // 创建文字
    // ~E(x,y)
    Literal l1(pred_E, {var_x, var_y}, true);
    // ~G(x)
    Literal l2(pred_G, {var_x}, true);
    // ~G(y)
    Literal l3(pred_G, {var_y}, true);

    // 创建子句并添加文字
    std::cout << "\nTest Scenario: Adding clause with same predicate" << std::endl;
    Clause c1;
    c1.addLiteral(l1);
    c1.addLiteral(l2);
    c1.addLiteral(l3);

    Literal l4(pred_G, {var_x}, false);
    Clause c2;
    c2.addLiteral(l4);
    auto new_nodes = tree.add_node(c2, Literal(), false, tree.getRoot());
    std::cout << "Print Empty Tree with a single node" << std::endl;
    tree.print_tree(kb);

    // 添加到树中并验证
    std::vector<std::shared_ptr<SLINode>> active_nodes = tree.add_node(c1, l2, false, new_nodes[0]);
    std::cout << "After Resolvent with ~E(x,y) ~G(x) ~G(y)" << std::endl;
    tree.print_tree(kb);
    ASSERT_EQ(active_nodes.size(), 2);
}

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
    std::cout << "After Add Node, befor t-factoring" << std::endl;
    tree.print_tree(kb);
    std::cout << "upper_nodes " << std::endl;
    active_nodes[1]->print(kb);
    std::cout << "lower_nodes " << std::endl;
    nodes2[0]->print(kb);
    bool result = tree.t_factoring(active_nodes[1], nodes2[0]);
    EXPECT_TRUE(result);
    std::cout << "\nAfter t-Factoring" << std::endl;
    tree.print_tree(kb);

    // 测试场景3：MGU失败的情况
    std::cout << "\nTest Scenario 3: MGU Failure" << std::endl;
    Clause c3;
    c3.addLiteral(l4);                                     // P(b,a)
    auto nodes3 = tree.add_node(c3, l4, false, nodes2[0]); // 尝试与Q(b)消解
    ASSERT_TRUE(nodes3.empty());                           // 应该失败因为不能统一
    tree.print_tree(kb);

    // 测试场景4：多重消解
    std::cout << "\nTest Scenario 4: Multiple Resolution" << std::endl;
    Clause c4;
    c4.addLiteral(l5);                                          // ~Q(b)
    auto nodes4 = tree.add_node(c4, l5, true, active_nodes[1]); // 与Q(b)消解
    ASSERT_TRUE(nodes4.empty());
    tree.print_tree(kb);

    // 测试场景5：空子句的情况
    std::cout << "\nTest Scenario 5: Empty Clause" << std::endl;
    Clause empty_clause;
    auto nodes5 = tree.add_node(empty_clause, Literal(), false, tree.getRoot());
    ASSERT_TRUE(nodes5.empty());

    // 测试场景6：单个文字的消解
    std::cout << "\nTest Scenario 6: Single Literal Resolution" << std::endl;
    Clause c6;
    c6.addLiteral(l5); // ~Q(b)
    auto nodes6 = tree.add_node(c6, l2, true, active_nodes[1]);
    ASSERT_FALSE(nodes6.empty());
    tree.print_tree(kb);
}

// 测试撤销操作
TEST_F(SLITreeTest, UndoOperation)
{
    SLITree tree(kb);

    // P(x,a)
    Literal l1(pred_P, {var_x, const_a}, false);
    // Q(b)
    Literal l2(pred_Q, {const_b}, false);

    // 添加初始节点
    Clause c1;
    c1.addLiteral(l1);
    c1.addLiteral(l2);
    auto nodes1 = tree.add_node(c1, Literal(), false, tree.getRoot());

    std::cout << "Before Undo:" << std::endl;
    tree.print_tree(kb);

    // 执行撤销
    tree.rollback();

    std::cout << "After Undo:" << std::endl;
    tree.print_tree(kb);

    // 验证树恢复到之前状态
    ASSERT_EQ(tree.getDepthMap().size(), 1); // 只有根节点层
}

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

TEST_F(SLITreeTest, TFactoring)
{
    SLITree tree(kb);

    // 测试场景1：基本的t-factoring操作
    std::cout << "\nTest Scenario 1: Basic T-Factoring" << std::endl;
    // P(x,a) ∨ P(b,y)
    Clause c1;
    Literal l1(pred_P, {var_x, const_a}, false);
    Literal l2(pred_P, {const_b, var_y}, false);
    c1.addLiteral(l1);
    c1.addLiteral(l2);

    auto nodes1 = tree.add_node(c1, Literal(), false, tree.getRoot());
    ASSERT_EQ(nodes1.size(), 2);

    bool result = tree.t_factoring(nodes1[0], nodes1[1]);
    EXPECT_TRUE(result);
    tree.print_tree(kb);

    // 测试场景2：不同深度的节点进行t-factoring
    std::cout << "\nTest Scenario 2: T-Factoring at Different Depths" << std::endl;
    // 创建第一个子句：P(x,a) ∨ Q(b)
    Clause c2;
    Literal l3(pred_P, {var_x, const_a}, false);
    Literal l4(pred_Q, {const_b}, false);
    c2.addLiteral(l3);
    c2.addLiteral(l4);

    auto active_nodes = tree.add_node(c2, Literal(), false, tree.getRoot());
    ASSERT_EQ(active_nodes.size(), 2);
    tree.print_tree(kb);

    // 创建第二个子句：~P(b,y) ∨ Q(x)
    Clause c3;
    Literal l5(pred_P, {const_b, var_y}, true);
    Literal l6(pred_Q, {var_x}, false);
    c3.addLiteral(l5);
    c3.addLiteral(l6); // Q(x)

    auto nodes2 = tree.add_node(c3, l5, true, active_nodes[0]);
    ASSERT_FALSE(nodes2.empty());
    std::cout << "After Add Node, before t-factoring" << std::endl;
    tree.print_tree(kb);

    std::cout << "upper_nodes " << std::endl;
    active_nodes[1]->print(kb);
    std::cout << "lower_nodes " << std::endl;
    nodes2[0]->print(kb);

    result = tree.t_factoring(active_nodes[1], nodes2[0]);
    EXPECT_TRUE(result);
    std::cout << "\nAfter t-Factoring" << std::endl;
    tree.print_tree(kb);

    // 测试场景3：回滚操作
    std::cout << "\nTest Scenario 3: Rollback Operation" << std::endl;
    // P(a,x) ∨ P(y, b)
    Clause c4;
    Literal l7(pred_P, {const_a, var_x}, false);
    Literal l8(pred_P, {var_y, const_b}, false);
    c4.addLiteral(l7);
    c4.addLiteral(l8);

    auto nodes3 = tree.add_node(c4, Literal(), false, tree.getRoot());
    ASSERT_EQ(nodes3.size(), 2);
    tree.print_tree(kb);

    // 保存原始状态
    auto original_lit1 = nodes3[0]->literal;
    auto original_lit2 = nodes3[1]->literal;
    auto original_active1 = nodes3[0]->is_active;
    auto original_active2 = nodes3[1]->is_active;

    result = tree.t_factoring(nodes3[0], nodes3[1]);
    EXPECT_TRUE(result);
    std::cout << "After t-factoring" << std::endl;
    tree.print_tree(kb);

    tree.rollback();

    // 验证状态恢复
    EXPECT_EQ(nodes3[0]->literal, original_lit1);
    EXPECT_EQ(nodes3[1]->literal, original_lit2);
    EXPECT_EQ(nodes3[0]->is_active, original_active1);
    EXPECT_EQ(nodes3[1]->is_active, original_active2);
    tree.print_tree(kb);

    // 测试场景4：带否定的文字
    std::cout << "\nTest Scenario 4: Literals with Negation" << std::endl;
    // ~P(x,a) ∨ ~P(b,y)
    Clause c5;
    Literal l9(pred_P, {var_x, const_a}, true);
    Literal l10(pred_P, {const_b, var_y}, true);
    c5.addLiteral(l9);
    c5.addLiteral(l10);

    auto nodes4 = tree.add_node(c5, Literal(), false, tree.getRoot());
    ASSERT_EQ(nodes4.size(), 2);
    tree.print_tree(kb);

    result = tree.t_factoring(nodes4[0], nodes4[1]);
    EXPECT_TRUE(result);
    tree.print_tree(kb);

    // 测试场景5：直接祖先关系的情况
    std::cout << "\nTest Scenario 5: Ancestor Relationship Cases" << std::endl;

    // 创建一个基础子句：P(x,a) ∨ Q(b)
    Clause c6;
    Literal l11(pred_P, {var_x, const_a}, false);
    Literal l12(pred_Q, {const_b}, false);
    c6.addLiteral(l11);
    c6.addLiteral(l12);

    auto nodes5 = tree.add_node(c6, Literal(), false, tree.getRoot());
    ASSERT_EQ(nodes5.size(), 2);

    // 在第一个节点下添加子节点：P(y,b) ∨ R(c)
    Clause c7;
    Literal l13(pred_P, {var_y, const_b}, false);
    Literal l14(pred_R, {const_c}, false);
    c7.addLiteral(l13);
    c7.addLiteral(l14);

    auto nodes5_children = tree.add_node(c7, Literal(), false, nodes5[0]);
    ASSERT_EQ(nodes5_children.size(), 2);
    tree.print_tree(kb);

    // 尝试对祖先节点和后代节点进行t-factoring
    std::cout << "Attempting t-factoring between ancestor and descendant" << std::endl;
    result = tree.t_factoring(nodes5[0], nodes5_children[0]);
    EXPECT_FALSE(result); // 应该失败，因为是直接祖先关系

    // 测试场景6：其他无效的节点组合
    std::cout << "\nTest Scenario 6: Invalid Node Combinations" << std::endl;

    // 测试空节点
    result = tree.t_factoring(nullptr, nodes4[0]);
    EXPECT_FALSE(result);

    result = tree.t_factoring(nodes4[0], nullptr);
    EXPECT_FALSE(result);

    // 测试非活动节点
    nodes4[1]->is_active = false;
    result = tree.t_factoring(nodes4[0], nodes4[1]);
    EXPECT_FALSE(result);

    // 测试场景7：复杂的替换情况
    std::cout << "\nTest Scenario 7: Complex Substitution" << std::endl;
    // P(x,y) ∨ P(y,x)
    Clause c8;
    Literal l15(pred_P, {var_x, var_y}, false);
    Literal l16(pred_P, {var_y, var_x}, false);
    c8.addLiteral(l15);
    c8.addLiteral(l16);

    auto nodes7 = tree.add_node(c8, Literal(), false, tree.getRoot());
    ASSERT_EQ(nodes7.size(), 2);

    tree.print_tree(kb);
    result = tree.t_factoring(nodes7[0], nodes7[1]);
    EXPECT_TRUE(result);
    tree.print_tree(kb);
}

TEST_F(SLITreeTest, TAncestry)
{
    SLITree tree(kb);

    // 测试场景1：基本的t-ancestry操作
    std::cout << "\nTest Scenario 1: Basic T-Ancestry" << std::endl;

    // 创建一个基础子句：P(x,a)
    Clause c1;
    Literal l1(pred_P, {var_x, const_a}, false);
    c1.addLiteral(l1);

    auto nodes1 = tree.add_node(c1, Literal(), false, tree.getRoot());
    ASSERT_EQ(nodes1.size(), 1);
    std::cout << "add node P(x,a)" << std::endl;
    tree.print_tree(kb);

    // 创建子节点：~P(x,a) ∨ Q(b)
    Clause c2;
    Literal l2(pred_P, {var_x, const_a}, true);
    Literal l3(pred_Q, {const_b}, false);
    c2.addLiteral(l2);
    c2.addLiteral(l3);

    std::cout << "result of add_node1 " << std::endl;
    nodes1[0]->print(kb);
    auto nodes2 = tree.add_node(c2, l2, false, nodes1[0]);
    for (const auto &node : nodes2)
    {
        node->print(kb);
    }
    ASSERT_EQ(nodes2.size(), 1);
    std::cout << "add node ~P(x,a) ∨ Q(b)" << std::endl;
    tree.print_tree(kb);

    // 创建子节点：~Q(b) ∨ ~P(x,a)
    Clause c3;
    Literal l4(pred_Q, {const_b}, true);
    Literal l5(pred_P, {var_y, var_x}, true);
    c3.addLiteral(l4);
    c3.addLiteral(l5);

    auto nodes3 = tree.add_node(c3, l4, false, nodes2[0]);
    std::cout << "add node ~Q(b) ∨ ~P(y,x)  " << std::endl;
    tree.print_tree(kb);
    bool result = tree.t_ancestry(nodes1[0], nodes3[0]);
    EXPECT_TRUE(result);
    std::cout << "after t-ancestry" << std::endl;
    tree.print_tree(kb);

    tree.rollback();
    std::cout << "after rollback()" << std::endl;
    tree.print_tree(kb);

    // // 测试场景2：非互补文字
    // std::cout << "\nTest Scenario 2: Non-complementary Literals" << std::endl;

    // // 创建根节点：P(x,a)
    // Clause c4;
    // Literal l6(pred_P, {var_x, const_a}, false);
    // c4.addLiteral(l6);

    // auto nodes4 = tree.add_node(c4, Literal(), false, tree.getRoot());
    // ASSERT_EQ(nodes4.size(), 1);
    // std::cout << "add node P(x,a)" << std::endl;
    // tree.print_tree(kb);

    // // 创建子节点：P(b,y) - 注意这里是正文字而非否定文字
    // Clause c5;
    // Literal l7(pred_P, {const_b, var_y}, false);
    // c5.addLiteral(l7);

    // auto nodes5 = tree.add_node(c5, l7, false, nodes4[0]);
    // ASSERT_EQ(nodes5.size(), 1);
    // std::cout << "add node P(b,y)" << std::endl;
    // tree.print_tree(kb);

    // result = tree.t_ancestry(nodes4[0], nodes5[0]);
    // EXPECT_FALSE(result);
}

TEST_F(SLITreeTest, TTruncate)
{
    SLITree tree(kb);

    // 测试场景1：基本的t-truncate操作 - 单个A-literal叶子节点
    std::cout << "\nTest Scenario 1: Basic T-Truncate with single A-literal leaf" << std::endl;

    // 创建基础子句：P(x,a)
    Clause c1;
    Literal l1(pred_P, {var_x, const_a}, false);
    c1.addLiteral(l1);

    auto nodes1 = tree.add_node(c1, Literal(), false, tree.getRoot());
    ASSERT_EQ(nodes1.size(), 1);

    // 标记为A-literal
    nodes1[0]->is_A_literal = true;

    std::cout << "Before truncate:" << std::endl;
    tree.print_tree(kb);

    tree.truncate(nodes1[0]);

    std::cout << "After truncate:" << std::endl;
    tree.print_tree(kb);

    EXPECT_FALSE(nodes1[0]->is_active);
    EXPECT_EQ(nodes1[0]->rule_applied, "t-truncate");

    // 测试场景2：向上传播的t-truncate
    std::cout << "\nTest Scenario 2: Upward propagation of T-Truncate" << std::endl;

    // 创建新的树结构：P(x,a) -> Q(b) -> R(c)
    Clause c2;
    Literal l2(pred_P, {var_x, const_a}, false);
    c2.addLiteral(l2);

    auto parent = tree.add_node(c2, Literal(), false, tree.getRoot())[0];
    parent->is_A_literal = true;

    Clause c3;
    Literal l3(pred_Q, {const_b}, false);
    c3.addLiteral(l3);

    auto middle = tree.add_node(c3, Literal(), false, parent)[0];
    middle->is_A_literal = true;

    Clause c4;
    Literal l4(pred_R, {const_c}, false);
    c4.addLiteral(l4);

    auto leaf = tree.add_node(c4, Literal(), false, middle)[0];
    leaf->is_A_literal = true;

    std::cout << "Before truncate cascade:" << std::endl;
    tree.print_tree(kb);

    tree.truncate(leaf);

    std::cout << "After truncate cascade:" << std::endl;
    tree.print_tree(kb);

    EXPECT_FALSE(leaf->is_active);
    EXPECT_FALSE(middle->is_active);
    EXPECT_FALSE(parent->is_active);

    // 测试场景3：撤销truncate操作
    std::cout << "\nTest Scenario 3: Undo T-Truncate" << std::endl;

    tree.rollback();

    std::cout << "After rollback:" << std::endl;
    tree.print_tree(kb);

    EXPECT_TRUE(leaf->is_active);
    EXPECT_TRUE(middle->is_active);
    EXPECT_TRUE(parent->is_active);

    // 测试场景4：非A-literal节点
    std::cout << "\nTest Scenario 4: Non-A-literal node" << std::endl;

    Clause c5;
    Literal l5(pred_P, {var_x, const_a}, false);
    c5.addLiteral(l5);

    auto non_a_literal = tree.add_node(c5, Literal(), false, tree.getRoot())[0];
    non_a_literal->is_A_literal = false; // 确保不是A-literal

    std::cout << "Before truncate on non-A-literal:" << std::endl;
    tree.print_tree(kb);

    tree.truncate(non_a_literal);

    std::cout << "After truncate attempt on non-A-literal:" << std::endl;
    tree.print_tree(kb);

    EXPECT_TRUE(non_a_literal->is_active); // 应该仍然是活动的

    // 测试场景5：有活跃子节点的A-literal节点
    std::cout << "\nTest Scenario 5: A-literal node with active children" << std::endl;

    Clause c6;
    Literal l6(pred_P, {var_x, const_a}, false);
    c6.addLiteral(l6);

    auto parent_with_children = tree.add_node(c6, Literal(), false, tree.getRoot())[0];
    parent_with_children->is_A_literal = true;

    Clause c7;
    Literal l7(pred_Q, {const_b}, false);
    c7.addLiteral(l7);

    auto active_child = tree.add_node(c7, Literal(), false, parent_with_children)[0];
    active_child->is_A_literal = false; // 子节点不是A-literal

    std::cout << "Before truncate on parent with active child:" << std::endl;
    tree.print_tree(kb);

    tree.truncate(parent_with_children);

    std::cout << "After truncate attempt on parent with active child:" << std::endl;
    tree.print_tree(kb);

    EXPECT_TRUE(parent_with_children->is_active); // 因为有活跃子节点，所以不应该被truncate
    EXPECT_TRUE(active_child->is_active);
}

TEST_F(SLITreeTest, GetGammaLAndDeltaL)
{
    // 子句1: A(x) ∨ B(x)
    Clause c1;
    Literal l1a(pred_A, {var_x}, false); // A(x)
    Literal l1b(pred_B, {var_x}, false); // B(x)
    c1.addLiteral(l1a);
    c1.addLiteral(l1b);

    // 子句2: ¬B(x) ∨ C(x) ∨ D(x)
    Clause c2;
    Literal l2a(pred_B, {var_x}, true);  // ¬B(x)
    Literal l2b(pred_C, {var_x}, false); // C(x)
    Literal l2c(pred_D, {var_x}, false); // D(x)
    c2.addLiteral(l2a);
    c2.addLiteral(l2b);
    c2.addLiteral(l2c);

    // 子句3: ¬C(x) E(x) F(x)
    Clause c3;
    Literal l3a(pred_C, {var_x}, true);  // ¬C(x)
    Literal l3b(pred_E, {var_x}, false); // E(x)
    Literal l3c(pred_F, {var_x}, false); // F(x)
    c3.addLiteral(l3a);
    c3.addLiteral(l3b);
    c3.addLiteral(l3c);

    SLITree tree(kb);

    // add A(x) ∨ B(x)
    auto add_nodes = tree.add_node(c1, Literal(), false, tree.getRoot());
    auto lit_A = add_nodes[0];
    tree.print_tree(kb);
    // add ¬B(x) ∨ C(x) D(x)
    add_nodes = tree.add_node(c2, l2a, false, add_nodes[1]);
    auto lit_D = add_nodes[1];
    tree.print_tree(kb);
    // add ¬C(x)∨E(x)∨F(x)
    add_nodes = tree.add_node(c3, l3a, false, add_nodes[0]);
    auto lit_E = add_nodes[0];
    auto lit_F = add_nodes[1];
    // Print tree for visualization
    std::cout << "\nTest Tree Structure:" << std::endl;
    tree.print_tree(kb);

    //        e*
    //     /     \
//    a       b*
    //          /    \
//         c*     d
    //       /   \
//      e     f

    // 获取节点E(x) e的gammal_l = {f,d,a}
    auto gamma_L = tree.get_gamma_L(lit_E);
    ASSERT_EQ(gamma_L.size(), 3);
    for (const auto node : gamma_L)
    {
        std::cout << node->literal.toString(kb) << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(gamma_L[2]->literal, l1a); // F
    ASSERT_EQ(gamma_L[1]->literal, l2c); // D
    ASSERT_EQ(gamma_L[0]->literal, l3c); // A

    // 获取节点F(x) 的gammal_l = {e,d,a}
    gamma_L = tree.get_gamma_L(lit_F);
    ASSERT_EQ(gamma_L.size(), 3);
    for (const auto node : gamma_L)
    {
        std::cout << node->literal.toString(kb) << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(gamma_L[2]->literal, l1a); // A
    ASSERT_EQ(gamma_L[1]->literal, l2c); // D
    ASSERT_EQ(gamma_L[0]->literal, l3b); // E

    // 获取节点D(x) 的gammal_l = {a}
    gamma_L = tree.get_gamma_L(lit_D);
    ASSERT_EQ(gamma_L.size(), 1);
    for (const auto node : gamma_L)
    {
        std::cout << node->literal.toString(kb) << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(gamma_L[0]->literal, l1a); // A

    // 获取节点A(x) 的gammal_l = {}
    gamma_L = tree.get_gamma_L(lit_A);
    ASSERT_EQ(gamma_L.size(), 0);
    // for(const auto lit : gamma_L)
    // {
    //     std::cout << lit.toString(kb) << " ";
    // }
    // std::cout << std::endl;

    // 获取节点E(x) delta_l = {f,d,a}
    auto delta_L = tree.get_delta_L(lit_E);
    ASSERT_EQ(delta_L.size(), 2);
    for (const auto node : delta_L)
    {
        std::cout << node->literal.toString(kb) << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(delta_L[0]->literal, l2b); // C
    ASSERT_EQ(delta_L[1]->literal, l1b); // B
}

TEST_F(SLITreeTest, ACandMCCheckTest)
{
    // 子句1: A(x) ∨ B(x)
    Clause c1;
    Literal l1a(pred_A, {var_x}, false); // A(x)
    Literal l1b(pred_B, {var_x}, false); // B(x)
    c1.addLiteral(l1a);
    c1.addLiteral(l1b);

    // 子句2: ¬B(x) ∨ C(x) ∨ D(x)
    Clause c2;
    Literal l2a(pred_B, {var_x}, true);  // ¬B(x)
    Literal l2b(pred_C, {var_x}, false); // C(x)
    Literal l2c(pred_D, {var_x}, false); // D(x)
    c2.addLiteral(l2a);
    c2.addLiteral(l2b);
    c2.addLiteral(l2c);

    // 子句3: ¬C(x) E(x) F(x)
    Clause c3;
    Literal l3a(pred_C, {var_x}, true);  // ¬C(x)
    Literal l3b(pred_E, {var_x}, false); // E(x)
    Literal l3c(pred_F, {var_x}, false); // F(x)
    c3.addLiteral(l3a);
    c3.addLiteral(l3b);
    c3.addLiteral(l3c);

    SLITree tree(kb);

    // add A(x) ∨ B(x)
    auto add_nodes = tree.add_node(c1, Literal(), false, tree.getRoot());
    auto lit_A = add_nodes[0];
    tree.print_tree(kb);
    // add ¬B(x) ∨ C(x) D(x)
    add_nodes = tree.add_node(c2, l2a, false, add_nodes[1]);
    auto lit_D = add_nodes[1];
    tree.print_tree(kb);
    // add ¬C(x)∨E(x)∨F(x)
    add_nodes = tree.add_node(c3, l3a, false, add_nodes[0]);
    auto lit_E = add_nodes[0];
    auto lit_F = add_nodes[1];
    // Print tree for visualization
    std::cout << "\nTest Tree Structure:" << std::endl;
    tree.print_tree(kb);

    //        e*
    //     /     \
//    a       b*
    //          /    \
//         c*     d
    //       /   \
//      e     f


    ASSERT_TRUE(tree.check_all_nodes_AC());
    ASSERT_TRUE(tree.check_all_nodes_MC());
}