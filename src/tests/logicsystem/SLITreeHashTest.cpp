#include <gtest/gtest.h>
#include "SLITree.h"
#include "KnowledgeBase.h"

namespace LogicSystem {
namespace testing {

class SLITreeHashTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 添加谓词
        pred_E = kb.addPredicate("E");     // 边关系谓词 E(x,y)
        pred_P = kb.addPredicate("P");     // 其他谓词 P(x)
        
        // 添加变量
        var_x = kb.addVariable("x");
        var_y = kb.addVariable("y");
        var_z = kb.addVariable("z");
        
        // 添加常量
        const_a = kb.addConstant("a");
        const_b = kb.addConstant("b");
    }

    KnowledgeBase kb;
    int pred_E, pred_P;
    SymbolId var_x, var_y, var_z, const_a, const_b;
};

// 测试节点的hash值计算
TEST_F(SLITreeHashTest, SameLiteralDifferentNodeIdHash) {
    SLITree tree1(kb);

    
    // 创建相同的文字和子句
    Literal lit1(pred_E, {var_x, var_y}, false);  // E(x,y)
    Literal lit2(pred_P, {var_x, var_y}, false);
    Clause clause1;
    clause1.addLiteral(lit1);
    Clause clause2;
    clause2.addLiteral(lit2);
    
    // 向两棵树添加相同的节点
    auto node1 = SLINode(lit1, false, 0);
    auto node2 = SLINode(lit1, false, 0);
    auto node3 = SLINode(lit2, false, 0);

    // 计算并比较节点的哈希值
    size_t hash1 = tree1.computeNodeHash(std::make_shared<SLINode>(node1));
    size_t hash2 = tree1.computeNodeHash(std::make_shared<SLINode>(node2));
    size_t hash3 = tree1.computeNodeHash(std::make_shared<SLINode>(node3));
    std::cout << "Node 1" << std::endl;
    node1.print(kb);
    std::cout << "Node1 hash: " << hash1 << std::endl;
    std::cout << "Node 2" << std::endl;
    node2.print(kb);
    std::cout << "Node2 hash: " << hash2 << std::endl;
    std::cout << "Node 3" << std::endl;
    node3.print(kb);
    std::cout << "Node3 hash: " << hash3 << std::endl;
    
    // 确保节点编号不同
    // std::cout << "Node1 ID: " << node1->node_id << std::endl;
    // std::cout << "Node2 ID: " << node2->node_id << std::endl;
    // EXPECT_NE(node1->node_id, node2->node_id);
    // node1->print(kb);
    // node2->print(kb);
    
    
    // size_t hash1 = tree1.computeNodeHash(node1);
    // size_t hash2 = tree2.computeNodeHash(node2);
    
    
    
    // 由于computeNodeHash当前的实现没有考虑node_id，
    // 具有相同文字和属性的节点应该产生相同的哈希值
    EXPECT_EQ(hash1, hash2);
    EXPECT_NE(hash1, hash3);
    
    // // 打印节点的详细信息以便调试
    // std::cout << "Node1 properties:" << std::endl;
    // std::cout << "Literal: " << node1->literal.toString(kb) << std::endl;
    // std::cout << "is_A_literal: " << node1->is_A_literal << std::endl;
    // std::cout << "is_active: " << node1->is_active << std::endl;
    // std::cout << "depth: " << node1->depth << std::endl;
    
    // std::cout << "\nNode2 properties:" << std::endl;
    // std::cout << "Literal: " << node2->literal.toString(kb) << std::endl;
    // std::cout << "is_A_literal: " << node2->is_A_literal << std::endl;
    // std::cout << "is_active: " << node2->is_active << std::endl;
    // std::cout << "depth: " << node2->depth << std::endl;
}

// 测试空树的hash值
TEST_F(SLITreeHashTest, EmptyTreeHash) {
    SLITree tree1(kb);
    SLITree tree2(kb);
    
    size_t hash1 = tree1.computeStateHash();
    size_t hash2 = tree2.computeStateHash();
    
    std::cout << "Empty tree1 hash: " << hash1 << std::endl;
    std::cout << "Empty tree2 hash: " << hash2 << std::endl;
    
    EXPECT_EQ(hash1, hash2);
}

// 测试添加单个节点后的hash值
TEST_F(SLITreeHashTest, SingleNodeHash) {
    SLITree tree1(kb);
    SLITree tree2(kb);
    
    // 创建两个相同的文字和子句
    Literal lit1(pred_E, {var_x, var_y}, false);  // E(x,y)
    Clause clause1;
    clause1.addLiteral(lit1);
    
    // 分别向两棵树添加相同的节点
    tree1.add_node(clause1, Literal(), true, tree1.getRoot());
    tree2.add_node(clause1, Literal(), true, tree2.getRoot());
    
    size_t hash1 = tree1.computeStateHash();
    size_t hash2 = tree2.computeStateHash();
    
    std::cout << "Tree1 with single node hash: " << hash1 << std::endl;
    std::cout << "Tree2 with single node hash: " << hash2 << std::endl;
    
    //结构一样，但是节点ID不同，不可以判断为不同的树
    EXPECT_EQ(hash1, hash2);
}

// 测试不同结构树的hash值
TEST_F(SLITreeHashTest, DifferentStructureHash) {
    SLITree tree1(kb);
    SLITree tree2(kb);
    
    // 创建不同的文字和子句
    Literal lit1(pred_E, {var_x, var_y}, false);  // E(x,y)
    Literal lit2(pred_P, {var_x}, false);         // P(x)
    
    Clause clause1;
    clause1.addLiteral(lit1);
    
    Clause clause2;
    clause2.addLiteral(lit2);
    
    // 向树1添加E(x,y)
    tree1.add_node(clause1, Literal(), true, tree1.getRoot());
    
    // 向树2添加P(x)
    tree2.add_node(clause2, Literal(), true, tree2.getRoot());
    
    size_t hash1 = tree1.computeStateHash();
    size_t hash2 = tree2.computeStateHash();
    
    std::cout << "Tree1 with E(x,y) hash: " << hash1 << std::endl;
    std::cout << "Tree2 with P(x) hash: " << hash2 << std::endl;
    
    EXPECT_NE(hash1, hash2);
}

// 测试多节点树的hash值
TEST_F(SLITreeHashTest, MultipleNodesHash) {
    SLITree tree1(kb);
    SLITree tree2(kb);
    
    // 创建多个文字和子句
    Literal lit1(pred_E, {var_x, var_y}, false);  // E(x,y)
    Literal lit2(pred_E, {var_y, var_z}, false);  // E(y,z)
    
    Clause clause1;
    clause1.addLiteral(lit1);
    
    Clause clause2;
    clause2.addLiteral(lit2);
    
    // 向两棵树添加相同的结构
    auto node1_1 = tree1.add_node(clause1, Literal(), true, tree1.getRoot())[0];
    auto node2_1 = tree2.add_node(clause1, Literal(), true, tree2.getRoot())[0];
    
    tree1.add_node(clause2, Literal(), true, node1_1);
    tree2.add_node(clause2, Literal(), true, node2_1);
    
    size_t hash1 = tree1.computeStateHash();
    size_t hash2 = tree2.computeStateHash();
    
    std::cout << "Tree1 with multiple nodes hash: " << hash1 << std::endl;
    std::cout << "Tree2 with multiple nodes hash: " << hash2 << std::endl;
    
    EXPECT_EQ(hash1, hash2);
}

// 测试节点状态变化后的hash值
TEST_F(SLITreeHashTest, NodeStateChangeHash) {
    SLITree tree(kb);
    
    // 创建文字和子句
    Literal lit(pred_E, {var_x, var_y}, false);  // E(x,y)
    Clause clause;
    clause.addLiteral(lit);
    
    // 添加节点并获取其引用
    auto nodes = tree.add_node(clause, Literal(), true, tree.getRoot());
    auto node = nodes[0];
    
    size_t hash1 = tree.computeStateHash();
    std::cout << "Tree hash before node deactivation: " << hash1 << std::endl;
    
    // 改变节点状态
    node->is_active = false;
    
    size_t hash2 = tree.computeStateHash();
    std::cout << "Tree hash after node deactivation: " << hash2 << std::endl;
    
    EXPECT_NE(hash1, hash2);
}

}  // namespace testing
}  // namespace LogicSystem