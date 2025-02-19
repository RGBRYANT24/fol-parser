#include <gtest/gtest.h>
#include "SLIMCTSState.h"
#include "SLITree.h"
#include "KnowledgeBase.h"
#include "Clause.h"
#include "Literal.h"
#include "SLIMCTSAction.h"

// 本测试文件用于验证 MCTS 框架中对接 SLITree 深拷贝的相关过程
namespace LogicSystem
{
    // 测试1：验证通过 SLIMCTSState 拷贝构造函数得到的深拷贝，其内部 SLITree 节点属性一致但指针指向不同
    TEST(SLIMCTSDeepCopyTest, StructureAndIndependence)
    {
        // 1. 创建知识库，并添加一个测试谓词
        KnowledgeBase kb;
        int pred_Test = kb.addPredicate("TestPredicate");

        // 2. 构造一个 Clause（用于扩展 SLITree 的节点）
        Clause clause;
        // 这里添加一个文字，假设 Literal 提供 toString 接口，第三个参数表示是否取反
        clause.addLiteral(Literal(pred_Test, std::vector<SymbolId>{}, false));

        // 3. 创建 SLITree
        auto tree = std::make_shared<SLITree>(kb);
        // 向根节点添加一个子节点
        auto children = tree->add_node(clause, Literal(), false, tree->getRoot());
        ASSERT_FALSE(children.empty()) << "添加子节点失败，children为空";

        // 为了构造多层结构，向第一个子节点再添加一个子节点
        auto childNode = children.front();
        tree->add_node(clause, Literal(), false, childNode);

        std::cout << "origin tree " << std::endl;
        tree->print_tree(kb);

        // 4. 利用该 SLITree 构造原始状态
        LogicSystem::SLIMCTSState original_state(tree);

        // 5. 通过拷贝构造函数生成深拷贝状态
        LogicSystem::SLIMCTSState copied_state(original_state);

        // 6. 验证深拷贝后的 SLITree 指针不同
        EXPECT_NE(original_state.sli_tree.get(), copied_state.sli_tree.get())
            << "深拷贝后的 SLITree 与原始 SLITree 指针相同，拷贝未生效";

        // 7. 对比原始树和深拷贝树的结构
        const auto &orig_depth_map = original_state.sli_tree->getDepthMap();
        const auto &copy_depth_map = copied_state.sli_tree->getDepthMap();
        EXPECT_EQ(orig_depth_map.size(), copy_depth_map.size())
            << "深拷贝后的树层数与原始树不一致";

        // 遍历各层节点，依次比较各个节点的属性（内容相同，但内存地址不同）
        for (size_t level = 0; level < orig_depth_map.size(); level++)
        {
            EXPECT_EQ(orig_depth_map[level].size(), copy_depth_map[level].size())
                << "深拷贝后，第" << level << "层节点数不一致";
            for (size_t idx = 0; idx < orig_depth_map[level].size(); idx++)
            {
                auto orig_node = orig_depth_map[level][idx];
                auto copy_node = copy_depth_map[level][idx];
                std::cout << "orig_node " << std::endl;
                orig_node->print(kb);
                std::cout << orig_node << std::endl;
                std::cout << "copy_node " << std::endl;
                copy_node->print(kb);
                std::cout << copy_node << std::endl;

                // 指针地址必须不同
                EXPECT_NE(orig_node.get(), copy_node.get())
                    << "深拷贝后的节点与原节点指向相同";

                // 检查基本属性是否一致
                EXPECT_EQ(orig_node->node_id, copy_node->node_id);
                EXPECT_EQ(orig_node->depth, copy_node->depth);
                EXPECT_EQ(orig_node->is_A_literal, copy_node->is_A_literal);
                EXPECT_EQ(orig_node->is_active, copy_node->is_active);
                EXPECT_EQ(orig_node->rule_applied, copy_node->rule_applied);

                // 检查文字是否一致（假设 Literal 提供 toString(const KnowledgeBase&) 接口）
                EXPECT_EQ(orig_node->literal.toString(kb), copy_node->literal.toString(kb));
            }
        }

        // 8. 测试深拷贝的独立性：修改原始树中某个节点的属性，验证深拷贝树不受影响
        if (!orig_depth_map.empty() && !orig_depth_map[0].empty())
        {
            auto orig_root = orig_depth_map[0][0];
            std::string old_rule = orig_root->rule_applied;
            orig_root->rule_applied = "ModifiedRule";
            auto copy_root = copy_depth_map[0][0];
            EXPECT_EQ(copy_root->rule_applied, old_rule)
                << "拷贝树根节点的 rule_applied 属性受原始树修改影响";
        }
    }

    // 测试2：验证通过 SLIMCTSState::next_state 生成新状态时，内部 SLITree 的深拷贝能够使得新旧状态完全独立
    TEST(SLIMCTSDeepCopyTest, NextStateDeepCopy)
    {
        // 1. 创建知识库和测试谓词
        KnowledgeBase kb;
        int pred_Test = kb.addPredicate("TestPredicate");

        // 2. 构造 Clause
        Clause clause;
        clause.addLiteral(Literal(pred_Test, std::vector<SymbolId>{}, false));

        // 3. 构造 SLITree，并扩展根节点
        auto tree = std::make_shared<SLITree>(kb);
        auto children = tree->add_node(clause, Literal(), false, tree->getRoot());
        ASSERT_FALSE(children.empty()) << "添加子节点失败，children为空";
        children[0]->is_A_literal = true;

        // 4. 构造初始 SLIMCTSState
        LogicSystem::SLIMCTSState original_state(tree);

        // 5. 构造一个简单的 SLIMCTSAction
        //    此处我们使用 TRUNCATE 操作，目标节点选取根节点
        SLIMCTSAction action;
        action.action = SLIActionType::TRUNCATE;
        action.lit1_node = children[0];
        // 第二个操作数设为 nullptr（封装在 SecondOperand 中）
        action.second_op = SecondOperand(std::shared_ptr<SLINode>(nullptr));
        // kb_clause 保持默认（空）状态
        action.kb_clause = Clause();

        std::cout << "action " << std::endl;
        std::cout << action.to_string(kb) << std::endl;

        // 6. 利用 next_state 生成新状态（内部会进行深拷贝并在新状态上应用 action）
        LogicSystem::SLIMCTSState new_state = original_state.next_state(action);

        std::cout << "origin tree " << std::endl;
        original_state.sli_tree->print_tree(kb);
        std::cout << "next state tree " << std::endl;
        new_state.sli_tree->print_tree(kb);

        // 7. 验证新状态与原状态的 SLITree 是独立的
        EXPECT_NE(original_state.sli_tree.get(), new_state.sli_tree.get())
            << "通过 next_state 生成的新状态未进行深拷贝";

        // 8. 假设 TRUNCATE 操作会修改目标节点的状态（例如设置 is_active 为 false），
        //    检查原始状态的根节点未受到影响
        EXPECT_TRUE(original_state.sli_tree->getRoot()->is_active)
            << "原始树的根节点状态被错误修改";
    }
}
