#include <gtest/gtest.h>
#include "SLITree.h"
#include "KnowledgeBase.h"

using namespace LogicSystem;

class SLITreeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置知识库
        // 添加谓词、常量和变量
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

TEST_F(SLITreeTest, BasicSLIResolution) {
    SLITree tree(kb);
    
    // 创建测试文字
    // P(x,a)
    Literal l1(pred_P, {var_x, const_a}, false);
    // P(b,y)
    Literal l2(pred_P, {const_b, var_y}, false);
    // ~P(b,a)
    Literal l3(pred_P, {const_b, const_a}, true);

    std::cout << "\n=== 初始状态 ===\n";
    
    // 添加节点并测试
    auto node1 = tree.add_node(l1, true);  // A-literal
    std::cout << "添加节点1后的树 [" << l1.toString(kb) << "]:\n";
    tree.print_tree(kb);
    
    auto node2 = tree.add_node(l2, false);
    std::cout << "\n添加节点2后的树 [" << l2.toString(kb) << "]:\n";
    tree.print_tree(kb);

    auto node3 = tree.add_node(l3, false);
    std::cout << "\n添加节点3后的树 [" << l3.toString(kb) << "]:\n";
    tree.print_tree(kb);

    // 测试T-factoring
    std::cout << "\n测试T-factoring (node1 和 node2):\n";
    bool factoring_result = tree.t_factoring(node1, node2);
    std::cout << "T-factoring 结果: " << (factoring_result ? "成功" : "失败") << "\n";
    tree.print_tree(kb);

    // 测试T-ancestry
    std::cout << "\n测试T-ancestry (node1 和 node3):\n";
    bool ancestry_result = tree.t_ancestry(node1, node3);
    std::cout << "T-ancestry 结果: " << (ancestry_result ? "成功" : "失败") << "\n";
    tree.print_tree(kb);

    // 测试回滚
    std::cout << "\n测试回滚操作:\n";
    tree.rollback();
    std::cout << "回滚后的树:\n";
    tree.print_tree(kb);

    // 测试截断
    std::cout << "\n测试截断操作:\n";
    tree.truncate(node2);
    std::cout << "截断后的树:\n";
    tree.print_tree(kb);
}

TEST_F(SLITreeTest, ComplexSLIResolution) {
    SLITree tree(kb);
    
    // 添加更多符号
    int pred_R = kb.addPredicate("R");    // 三元谓词 R
    SymbolId var_z = kb.addVariable("z");

    // 创建复杂的测试文字
    // R(x,y,a)
    Literal l1(pred_R, {var_x, var_y, const_a}, false);
    // R(b,x,y)
    Literal l2(pred_R, {const_b, var_x, var_y}, false);
    // ~R(b,a,z)
    Literal l3(pred_R, {const_b, const_a, var_z}, true);

    std::cout << "\n=== 复杂场景测试 ===\n";

    auto node1 = tree.add_node(l1, true);
    std::cout << "添加第一个复杂节点后 [" << l1.toString(kb) << "]:\n";
    tree.print_tree(kb);

    auto node2 = tree.add_node(l2, false);
    std::cout << "\n添加第二个复杂节点后 [" << l2.toString(kb) << "]:\n";
    tree.print_tree(kb);

    auto node3 = tree.add_node(l3, false);
    std::cout << "\n添加第三个复杂节点后 [" << l3.toString(kb) << "]:\n";
    tree.print_tree(kb);

    // 测试复杂的统一场景
    std::cout << "\n尝试统一 node1 和 node2:\n";
    auto unified = tree.try_unify(l1, l2);
    if (unified) {
        std::cout << "统一成功，结果: " << unified->toString(kb) << "\n";
    } else {
        std::cout << "统一失败\n";
    }

    // 测试多重操作
    bool factoring_result = tree.t_factoring(node1, node2);
    bool ancestry_result = tree.t_ancestry(node2, node3);
    
    std::cout << "\n执行多个操作后的树:\n";
    tree.print_tree(kb);

    std::cout << "\n执行多重回滚:\n";
    tree.rollback();
    tree.rollback();
    tree.print_tree(kb);
}