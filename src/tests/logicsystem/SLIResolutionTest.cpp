#include <gtest/gtest.h>
#include "SLIResolution.h"
#include "BFSStrategy.h"
#include "KnowledgeBase.h"

namespace LogicSystem
{

    class SLIResolutionTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // 设置知识库的基本谓词和符号
            pred_P = kb.addPredicate("P");
            pred_Q = kb.addPredicate("Q");
            pred_R = kb.addPredicate("R");

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
            var_z = kb.addVariable("z");
        }

        // 辅助函数：创建一个基本的BFS策略
        BFSStrategy createStrategy(int maxDepth = 5)
        {
            return BFSStrategy(maxDepth, 60.0, 1024 * 1024 * 100);
        }

        KnowledgeBase kb;
        int pred_P;
        int pred_Q;
        int pred_R;
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
        SymbolId var_z;
    };

    TEST_F(SLIResolutionTest, SimpleOneStepResolution)
    {
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

    // 辅助函数：构建一个具有特定结构的SLITree
    std::shared_ptr<SLITree> buildSampleTree(KnowledgeBase &kb,
                                             int pred_A, int pred_B, int pred_C, int pred_D, int pred_E, int pred_F,
                                             SymbolId var_x, SymbolId var_y, SymbolId var_z,
                                             bool includeInactive = false)
    {
        // 创建树
        auto tree = std::make_shared<SLITree>(kb);

        // 添加 A(x) ∨ B(x)
        Clause c1;
        Literal l1a(pred_A, {var_x}, false); // A(x)
        Literal l1b(pred_B, {var_x}, false); // B(x)
        c1.addLiteral(l1a);
        c1.addLiteral(l1b);
        auto nodes1 = tree->add_node(c1, Literal(), false, tree->getRoot());
        auto lit_A = nodes1[0];
        auto lit_B = nodes1[1];

        // 添加 ¬B(x) ∨ C(x) ∨ D(x)
        Clause c2;
        Literal l2a(pred_B, {var_x}, true);  // ¬B(x)
        Literal l2b(pred_C, {var_x}, false); // C(x)
        Literal l2c(pred_D, {var_x}, false); // D(x)
        c2.addLiteral(l2a);
        c2.addLiteral(l2b);
        c2.addLiteral(l2c);
        auto nodes2 = tree->add_node(c2, l2a, false, lit_B);
        auto lit_C = nodes2[0];
        auto lit_D = nodes2[1];

        // 添加 ¬C(x) ∨ E(x) ∨ F(x)
        Clause c3;
        Literal l3a(pred_C, {var_x}, true);  // ¬C(x)
        Literal l3b(pred_E, {var_x}, false); // E(x)
        Literal l3c(pred_F, {var_x}, false); // F(x)
        c3.addLiteral(l3a);
        c3.addLiteral(l3b);
        c3.addLiteral(l3c);
        auto nodes3 = tree->add_node(c3, l3a, false, lit_C);
        auto lit_E = nodes3[1];
        auto lit_F = nodes3[2];

        // // 可选：设置某些节点为不活动
        // if (includeInactive)
        // {
        //     lit_B->is_active = false;
        //     lit_C->is_active = false;
        // }

        return tree;
    }
    // ... 其他测试用例类似修改 ...

    TEST_F(SLIResolutionTest, BacktrackingTest)
    {
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

    TEST_F(SLIResolutionTest, GenerateExtensionStatesTest)
    {
        // 设置基本知识库
        // P(x,a)
        Clause kb_clause1;
        kb_clause1.addLiteral(Literal(pred_P, {var_x, const_a}, false));
        kb.addClause(kb_clause1);

        // ~P(x,y) ∨ Q(x)
        Clause kb_clause2;
        kb_clause2.addLiteral(Literal(pred_P, {var_x, var_y}, true));
        kb_clause2.addLiteral(Literal(pred_Q, {var_x}, false));
        kb.addClause(kb_clause2);

        // ~Q(x) ∨ R(x)
        Clause kb_clause3;
        kb_clause3.addLiteral(Literal(pred_Q, {var_x}, true));
        kb_clause3.addLiteral(Literal(pred_R, {var_x}, false));
        kb.addClause(kb_clause3);

        // 创建初始SLI树
        auto initial_tree = std::make_shared<SLITree>(kb);

        // 添加初始节点 P(x,a)
        auto initial_nodes = initial_tree->add_node(kb_clause1, Literal(), false, initial_tree->getRoot());
        ASSERT_EQ(initial_nodes.size(), 1);

        // 创建初始状态
        auto current_state = std::make_shared<SLIOperation::OperationState>(
            initial_tree,
            SLIActionType::EXTENSION,
            initial_nodes[0],
            SecondOperand(Literal()),
            Clause(),
            nullptr);

        std::stack<std::shared_ptr<SLIOperation::OperationState>> state_stack;

        std::cout << "\nInitial tree state:" << std::endl;
        initial_tree->print_tree(kb);

        // 获取所有B-literals
        auto b_literals = initial_tree->get_all_B_literals();
        ASSERT_FALSE(b_literals.empty());
        std::vector<std::shared_ptr<SLIOperation::OperationState>> available_ops;

        // 生成Extension状态
        SLIResolution::generateExtensionStates(kb, b_literals, current_state, state_stack, available_ops);

        // 验证生成的状态
        ASSERT_FALSE(state_stack.empty());

        // 测试每个生成的状态
        size_t expected_states = 0;
        for (const auto &kb_clause : kb.getClauses())
        {
            for (const auto &lit : kb_clause.getLiterals())
            {
                for (const auto &b_node : b_literals)
                {
                    if (Resolution::isComplementary(b_node->literal, lit))
                    {
                        expected_states++;
                    }
                }
            }
        }

        ASSERT_EQ(state_stack.size(), expected_states);

        // 验证生成的状态的正确性
        while (!state_stack.empty())
        {
            auto state = state_stack.top();
            state_stack.pop();

            // 验证状态属性
            EXPECT_EQ(state->action, SLIActionType::EXTENSION);
            EXPECT_TRUE(state->sli_tree != nullptr);
            EXPECT_TRUE(state->lit1_node != nullptr);
            EXPECT_TRUE(std::holds_alternative<Literal>(state->second_op));
            EXPECT_FALSE(state->kb_clause.isEmpty());

            // 验证补充文字的关系
            auto lit2 = std::get<Literal>(state->second_op);
            EXPECT_TRUE(Resolution::isComplementary(state->lit1_node->literal, lit2));

            std::cout << "\nValidating Extension state:" << std::endl;
            std::cout << "First literal: " << state->lit1_node->literal.toString(kb) << std::endl;
            std::cout << "Second literal: " << lit2.toString(kb) << std::endl;
            std::cout << "KB clause: " << state->kb_clause.toString(kb) << std::endl;
        }

        // 测试非活动节点
        b_literals[0]->is_active = false;
        state_stack = std::stack<std::shared_ptr<SLIOperation::OperationState>>();
        SLIResolution::generateExtensionStates(kb, b_literals, current_state, state_stack, available_ops);
        EXPECT_LT(state_stack.size(), expected_states);

        // 测试A-literal节点
        b_literals[0]->is_active = true;
        b_literals[0]->is_A_literal = true;
        state_stack = std::stack<std::shared_ptr<SLIOperation::OperationState>>();
        SLIResolution::generateExtensionStates(kb, b_literals, current_state, state_stack, available_ops);
        EXPECT_LT(state_stack.size(), expected_states);

        // 测试空的B-literals列表
        std::vector<std::shared_ptr<SLINode>> empty_b_literals;
        state_stack = std::stack<std::shared_ptr<SLIOperation::OperationState>>();
        SLIResolution::generateExtensionStates(kb, empty_b_literals, current_state, state_stack, available_ops);
        EXPECT_TRUE(state_stack.empty());
    }

    TEST_F(SLIResolutionTest, FindPotentialFactoringPairsTest)
    {
        // Test 1: Empty tree
        {
            auto tree = std::make_shared<SLITree>(kb);
            auto pairs = SLIResolution::findPotentialFactoringPairs(tree);
            EXPECT_TRUE(pairs.empty());
        }

        // Test 2: Single B-literal
        {
            auto tree = std::make_shared<SLITree>(kb);
            Clause clause;
            clause.addLiteral(Literal(pred_P, {var_x, const_a}, false));
            tree->add_node(clause, Literal(), false, tree->getRoot());

            auto pairs = SLIResolution::findPotentialFactoringPairs(tree);
            EXPECT_TRUE(pairs.empty());
        }

        // Test 3: Two unifiable B-literals
        {
            auto tree = std::make_shared<SLITree>(kb);

            // First literal: P(x,a)
            Clause clause1;
            clause1.addLiteral(Literal(pred_P, {var_x, const_a}, false));
            auto nodes1 = tree->add_node(clause1, Literal(), false, tree->getRoot());

            // Second literal: P(y,a)
            Clause clause2;
            clause2.addLiteral(Literal(pred_P, {var_y, const_a}, false));
            auto nodes2 = tree->add_node(clause2, Literal(), false, tree->getRoot());

            auto pairs = SLIResolution::findPotentialFactoringPairs(tree);
            ASSERT_EQ(pairs.size(), 1);
            EXPECT_EQ(pairs[0].first->literal.getPredicateId(), pred_P);
            EXPECT_EQ(pairs[0].second->literal.getPredicateId(), pred_P);
        }

        // Test 4: Different predicates (should not pair)
        {
            auto tree = std::make_shared<SLITree>(kb);

            // P(x,a)
            Clause clause1;
            clause1.addLiteral(Literal(pred_P, {var_x, const_a}, false));
            auto nodes1 = tree->add_node(clause1, Literal(), false, tree->getRoot());

            // Q(x,a)
            Clause clause2;
            clause2.addLiteral(Literal(pred_Q, {var_x, const_a}, false));
            auto nodes2 = tree->add_node(clause2, Literal(), false, tree->getRoot());

            auto pairs = SLIResolution::findPotentialFactoringPairs(tree);
            EXPECT_TRUE(pairs.empty());
        }

        // Test 5: Different negation (should not pair)
        {
            auto tree = std::make_shared<SLITree>(kb);

            // P(x,a)
            Clause clause1;
            clause1.addLiteral(Literal(pred_P, {var_x, const_a}, false));
            auto nodes1 = tree->add_node(clause1, Literal(), false, tree->getRoot());

            // ¬P(x,a)
            Clause clause2;
            clause2.addLiteral(Literal(pred_P, {var_x, const_a}, true));
            auto nodes2 = tree->add_node(clause2, Literal(), false, tree->getRoot());

            auto pairs = SLIResolution::findPotentialFactoringPairs(tree);
            EXPECT_TRUE(pairs.empty());
        }

        // Test 6: Different argument count (should not pair)
        {
            auto tree = std::make_shared<SLITree>(kb);

            // P(x,a)
            Clause clause1;
            clause1.addLiteral(Literal(pred_P, {var_x, const_a}, false));
            auto nodes1 = tree->add_node(clause1, Literal(), false, tree->getRoot());

            // P(x)
            Clause clause2;
            clause2.addLiteral(Literal(pred_P, {var_x}, false));
            auto nodes2 = tree->add_node(clause2, Literal(), false, tree->getRoot());

            auto pairs = SLIResolution::findPotentialFactoringPairs(tree);
            EXPECT_TRUE(pairs.empty());
        }

        // Test 7: Multiple potential pairs
        {
            auto tree = std::make_shared<SLITree>(kb);

            // P(x,a)
            Clause clause1;
            clause1.addLiteral(Literal(pred_P, {var_x, const_a}, false));
            auto nodes1 = tree->add_node(clause1, Literal(), false, tree->getRoot());

            // P(y,a)
            Clause clause2;
            clause2.addLiteral(Literal(pred_P, {var_y, const_a}, false));
            auto nodes2 = tree->add_node(clause2, Literal(), false, tree->getRoot());

            // P(z,a)
            Clause clause3;
            clause3.addLiteral(Literal(pred_P, {var_z, const_a}, false));
            auto nodes3 = tree->add_node(clause3, Literal(), false, tree->getRoot());

            auto pairs = SLIResolution::findPotentialFactoringPairs(tree);
            EXPECT_EQ(pairs.size(), 3); // Should have 3 pairs: (1,2), (1,3), (2,3)
        }

        // Test 8: Inactive nodes (should not pair)
        {
            auto tree = std::make_shared<SLITree>(kb);

            // P(x,a)
            Clause clause1;
            clause1.addLiteral(Literal(pred_P, {var_x, const_a}, false));
            auto nodes1 = tree->add_node(clause1, Literal(), false, tree->getRoot());

            // P(y,a) (A-lit)
            Clause clause2;
            clause2.addLiteral(Literal(pred_P, {var_y, const_a}, false));
            auto nodes2 = tree->add_node(clause2, Literal(), false, tree->getRoot());
            nodes2[0]->is_active = false;

            auto pairs = SLIResolution::findPotentialFactoringPairs(tree);
            std::cout << "pair size " << pairs.size() << std::endl;
            EXPECT_TRUE(pairs.empty());
        }

        // Test 9: multiple nodes
        {
            // 子句1: A(x) ∨ B(x)
            Clause c1;
            Literal l1a(pred_A, {var_x}, false); // A(x)
            Literal l1b(pred_B, {var_x}, false); // B(x)
            c1.addLiteral(l1a);
            c1.addLiteral(l1b);

            // 子句2: ¬B(x) ∨ C(x) ∨ E(x)
            Clause c2;
            Literal l2a(pred_B, {var_x}, true);  // ¬B(x)
            Literal l2b(pred_C, {var_x}, false); // C(x)
            Literal l2c(pred_E, {var_x}, false); // E(x)
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

            // 子句4：¬A(x) E(x)
            Clause c4;
            Literal l4a(pred_A, {var_z}, true);
            Literal l4e(pred_E, {var_z}, false);
            c4.addLiteral(l4a);
            c4.addLiteral(l4e);

            // 使用make_shared创建树
            auto tree = std::make_shared<SLITree>(kb);

            // 构建树结构
            // add A(x) ∨ B(x)
            auto add_nodes = tree->add_node(c1, Literal(), false, tree->getRoot());
            auto lit_A = add_nodes[0];
            tree->add_node(c4, l4a, false, lit_A);
            tree->print_tree(kb);

            // add ¬B(x) ∨ C(x) E(x)
            add_nodes = tree->add_node(c2, l2a, false, add_nodes[1]);
            auto lit_D = add_nodes[1];
            tree->print_tree(kb);

            // add ¬C(x)∨E(x)∨F(x)
            add_nodes = tree->add_node(c3, l3a, false, add_nodes[0]);
            auto lit_E = add_nodes[0];
            auto lit_F = add_nodes[1];

            // Print tree for visualization
            std::cout << "\nTest Tree Structure:" << std::endl;
            tree->print_tree(kb);

            //        e*
            //     /     \
            //    a*      b*
            //    |     /    \
            //    e     c*     e
            //       /   \
            //      e     f

            // 查找可能的factoring对
            auto factoring_pairs = SLIResolution::findPotentialFactoringPairs(tree);

            // 打印结果
            std::cout << "factoring pairs size " << factoring_pairs.size() << std::endl;
            if (factoring_pairs.size() > 0)
            {
                for (const auto &p : factoring_pairs)
                {
                    if (p.first && p.second)
                    {
                        std::cout << p.first->node_id << " " << p.second->node_id << std::endl;
                    }
                }
            }

            // 验证结果
            ASSERT_EQ(factoring_pairs.size(), 1);
        }
    }

    // 新增的 findPotentialAncestryPairs 测试
    TEST_F(SLIResolutionTest, FindPotentialAncestryPairs_EmptyTree)
    {
        auto tree = std::make_shared<SLITree>(kb);
        auto pairs = SLIResolution::findPotentialAncestryPairs(tree);
        EXPECT_TRUE(pairs.empty());
    }

    TEST_F(SLIResolutionTest, FindPotentialAncestryPairs_NoBLiterals)
    {
        auto tree = std::make_shared<SLITree>(kb);

        // 添加一个不含 B-literals 的节点
        Clause c1;
        Literal l1a(pred_A, {var_x}, false); // A(x)
        c1.addLiteral(l1a);
        auto new_node_A = tree->add_node(c1, Literal(), false, tree->getRoot());
        tree->print_tree(kb);

        auto pairs = SLIResolution::findPotentialAncestryPairs(tree);
        EXPECT_TRUE(pairs.empty());
    }
    TEST_F(SLIResolutionTest, FindPotentialAncestryPairs_NoMatchingPairs)
    {
        auto tree = std::make_shared<SLITree>(kb);

        // 添加 B(x)
        Clause c1;
        Literal l1b(pred_B, {var_x}, false); // B(x)
        c1.addLiteral(l1b);
        auto nodes1 = tree->add_node(c1, Literal(), false, tree->getRoot());
        auto lit_B = nodes1[0];

        // 添加另一个 B(x) 但不符合条件（相同否定状态）
        // ¬B(y) \lor B(y)
        Clause c2;
        Literal l2a(pred_B, {var_y}, true);  // ¬B(y)
        Literal l2b(pred_B, {var_y}, false); // B(y)
        c2.addLiteral(l2a);
        c2.addLiteral(l2b);
        auto nodes2 = tree->add_node(c2, l2a, false, lit_B);
        auto lit_B2 = nodes2[0];

        auto pairs = SLIResolution::findPotentialAncestryPairs(tree);
        EXPECT_TRUE(pairs.empty());
        ASSERT_FALSE(tree->check_all_nodes_AC());
    }

    TEST_F(SLIResolutionTest, FindPotentialAncestryPairs_WithMatchingPairs)
    {
        auto tree = std::make_shared<SLITree>(kb);

        // 添加 B(x)
        Clause c1;
        Literal l1b(pred_B, {var_x}, false); // B(x)
        c1.addLiteral(l1b);
        auto nodes1 = tree->add_node(c1, Literal(), false, tree->getRoot());
        auto lit_B = nodes1[0];

        // 添加 ¬B(x) ∨ C(x) ∨ D(x)
        Clause c2;
        Literal l2a(pred_B, {var_x}, true);  // ¬B(x)
        Literal l2b(pred_C, {var_x}, false); // C(x)
        Literal l2c(pred_D, {var_x}, false); // D(x)
        c2.addLiteral(l2a);
        c2.addLiteral(l2b);
        c2.addLiteral(l2c);
        auto nodes2 = tree->add_node(c2, l2a, false, lit_B);
        auto lit_C = nodes2[0];
        auto lit_D = nodes2[1];

        // 添加 ¬C(x) ∨ E(x) ∨ F(x)
        Clause c3;
        Literal l3a(pred_C, {var_x}, true);  // ¬C(x)
        Literal l3b(pred_E, {var_x}, false); // E(x)
        Literal l3c(pred_F, {var_x}, false); // F(x)
        c3.addLiteral(l3a);
        c3.addLiteral(l3b);
        c3.addLiteral(l3c);
        auto nodes3 = tree->add_node(c3, l3a, false, lit_C);
        auto lit_E = nodes3[0];
        auto lit_F = nodes3[1];

        auto pairs = SLIResolution::findPotentialAncestryPairs(tree);
        EXPECT_TRUE(pairs.empty()); //
    }

    TEST_F(SLIResolutionTest, FindPotentialAncestryPairs_MultipleMatchingPairs)
    {
        auto tree = std::make_shared<SLITree>(kb);

        // 添加 B(x)
        Clause c1;
        Literal l1b(pred_B, {var_x}, false); // B(x)
        c1.addLiteral(l1b);
        auto nodes1 = tree->add_node(c1, Literal(), false, tree->getRoot());
        auto lit_B1 = nodes1[0];

        // 添加 ¬B(x) ∨ C(x) ∨ D(x)
        Clause c2;
        Literal l2a(pred_B, {var_x}, true);  // ¬B(x)
        Literal l2b(pred_C, {var_x}, false); // C(x)
        Literal l2c(pred_D, {var_x}, false); // D(x)
        c2.addLiteral(l2a);
        c2.addLiteral(l2b);
        c2.addLiteral(l2c);
        auto nodes2 = tree->add_node(c2, l2a, false, lit_B1);
        auto lit_C = nodes2[0];
        auto lit_D = nodes2[1];

        // 添加 ¬C(x) ∨ E(x) ∨ F(x)
        Clause c3;
        Literal l3a(pred_C, {var_x}, true);  // ¬C(x)
        Literal l3b(pred_E, {var_x}, false); // E(x)
        Literal l3c(pred_F, {var_x}, false); // F(x)
        c3.addLiteral(l3a);
        c3.addLiteral(l3b);
        c3.addLiteral(l3c);
        auto nodes3 = tree->add_node(c3, l3a, false, lit_C);
        auto lit_E = nodes3[0];
        auto lit_F = nodes3[1];

        // 添加另一个 B(y)
        Clause c4;
        Literal l4b(pred_B, {var_y}, false); // B(y)
        c4.addLiteral(l4b);
        auto nodes4 = tree->add_node(c4, Literal(), false, tree->getRoot());
        auto lit_B2 = nodes4[0];

        // 添加 ¬B(y) ∨ G(y)
        Clause c5;
        Literal l5a(pred_B, {var_y}, true);  // ¬B(y)
        Literal l5b(pred_G, {var_y}, false); // G(y)
        c5.addLiteral(l5a);
        c5.addLiteral(l5b);
        auto nodes5 = tree->add_node(c5, l5a, false, lit_B2);
        auto lit_G = nodes5[0];

        // 现在有两个 B-literals: lit_B1 和 lit_B2
        auto pairs = SLIResolution::findPotentialAncestryPairs(tree);
        // 期望有两个对: (root B1, B1 itself does not pair), (root B2)
        // 根据函数实现，应该找出对 (root B1's ancestor nodes matching conditions, which might not exist)
        // 需要更复杂的结构来确保存在匹配对

        // 重新设计树结构，确保有匹配对
        // 添加一个 B-literal with different predicate condition
        // 这里假设 lit_B1 和 lit_B2 不会形成对，因为他们属于不同变量

        // 为确保有匹配对，添加一个具有相同谓词和相反否定状态的 B-literal
        Clause c6;
        Literal l6d(pred_D, {var_y}, true); // ¬D(y)
        Literal l6b(pred_B, {var_x}, true); // ¬B(x)
        c6.addLiteral(l6d);
        c6.addLiteral(l6b);
        auto nodes6 = tree->add_node(c6, l6d, false, lit_D); // parent is lit_D
        auto lit_B3 = nodes6[0];
        tree->print_tree(kb);

        // 现在 lit_B1 和 lit_B3 应该形成一个祖先对
        auto updated_pairs = SLIResolution::findPotentialAncestryPairs(tree);
        // ASSERT_FALSE(tree->check_all_nodes_AC());
        ASSERT_EQ(updated_pairs.size(), 1);
        EXPECT_EQ(updated_pairs[0].first, lit_B1);  // upper_node
        EXPECT_EQ(updated_pairs[0].second, lit_B3); // lower_node
    }
    // 新增的 findPotentialTruncateNodes 测试
    TEST_F(SLIResolutionTest, FindPotentialTruncateNodes_EmptyTree)
    {
        auto tree = std::make_shared<SLITree>(kb);
        auto truncate_nodes = SLIResolution::findPotentialTruncateNodes(tree);
        EXPECT_TRUE(truncate_nodes.empty());
    }

    TEST_F(SLIResolutionTest, FindPotentialTruncateNodes_NoALiterals)
    {
        auto tree = std::make_shared<SLITree>(kb);

        // 添加 B(x)
        Clause c1;
        Literal l1b(pred_B, {var_x}, false); // B(x)
        c1.addLiteral(l1b);
        tree->add_node(c1, Literal(), false, tree->getRoot());

        auto truncate_nodes = SLIResolution::findPotentialTruncateNodes(tree);
        EXPECT_TRUE(truncate_nodes.empty());
    }

    TEST_F(SLIResolutionTest, FindPotentialTruncateNodes_AllALiteralsHaveChildren)
    {
        auto tree = std::make_shared<SLITree>(kb);

        // 添加 A(x) ∨ B(x)
        Clause c1;
        Literal l1a(pred_A, {var_x}, false); // A(x)
        Literal l1b(pred_B, {var_x}, false); // B(x)
        c1.addLiteral(l1a);
        c1.addLiteral(l1b);
        auto nodes1 = tree->add_node(c1, Literal(), false, tree->getRoot());
        auto lit_A = nodes1[0];
        auto lit_B = nodes1[1];

        // 添加 ¬B(x) ∨ C(x)
        Clause c2;
        Literal l2a(pred_B, {var_x}, true);  // ¬B(x)
        Literal l2b(pred_C, {var_x}, false); // C(x)
        c2.addLiteral(l2a);
        c2.addLiteral(l2b);
        auto nodes2 = tree->add_node(c2, l2a, false, lit_B);
        auto lit_C = nodes2[0];

        // 添加 ¬C(x) ∨ E(x)
        Clause c3;
        Literal l3a(pred_C, {var_x}, true);  // ¬C(x)
        Literal l3b(pred_E, {var_x}, false); // E(x)
        c3.addLiteral(l3a);
        c3.addLiteral(l3b);
        tree->add_node(c3, l3a, false, lit_C);

        auto truncate_nodes = SLIResolution::findPotentialTruncateNodes(tree);
        EXPECT_TRUE(truncate_nodes.empty());
    }
    TEST_F(SLIResolutionTest, FindPotentialTruncateNodes_MixedALiterals)
    {
        auto tree = std::make_shared<SLITree>(kb);

        // 添加 A(x) ∨ B(x)
        Clause c1;
        Literal l1a(pred_A, {var_x}, false); // A(x)
        Literal l1b(pred_B, {var_x}, false); // B(x)
        c1.addLiteral(l1a);
        c1.addLiteral(l1b);
        auto nodes1 = tree->add_node(c1, Literal(), false, tree->getRoot());
        auto lit_A = nodes1[0];
        auto lit_B = nodes1[1];

        // 添加 ¬B(x) ∨ C(x)
        Clause c2;
        Literal l2a(pred_B, {var_x}, true);  // ¬B(x)
        Literal l2b(pred_C, {var_x}, false); // C(x)
        c2.addLiteral(l2a);
        c2.addLiteral(l2b);
        auto nodes2 = tree->add_node(c2, l2a, false, lit_B);
        auto lit_C = nodes2[0];

        // 添加 ¬C(x) ∨ E(x)
        Clause c3;
        Literal l3a(pred_C, {var_x}, true);  // ¬C(x)
        Literal l3b(pred_E, {var_x}, false); // E(x)
        c3.addLiteral(l3a);
        c3.addLiteral(l3b);
        tree->add_node(c3, l3a, false, lit_C);

        // 添加另一个 A(y) 没有子节点
        Clause c4;
        Literal l4a(pred_A, {var_y}, false); // A(y)
        c4.addLiteral(l4a);
        auto nodes4 = tree->add_node(c4, Literal(), false, tree->getRoot());
        auto lit_A2 = nodes4[0];
        // lit_A2->is_A_literal = true;

        // 添加另一个 A(z) 有子节点
        Clause c5;
        Literal l5a(pred_A, {var_z}, true); // A(z)
        c5.addLiteral(l5a);
        auto nodes5 = tree->add_node(c5, l5a, false, lit_A2);
        // auto lit_A3 = nodes5[0];

        // // 添加一个子节点到 A(z)
        // Clause c6;
        // Literal l6b(pred_F, {var_z}, false); // F(z)
        // c6.addLiteral(l6b);
        // tree->add_node(c6, Literal(), false, lit_A3);

        auto truncate_nodes = SLIResolution::findPotentialTruncateNodes(tree);
        ASSERT_EQ(truncate_nodes.size(), 1);
        EXPECT_EQ(truncate_nodes[0], lit_A2);
    }

} // namespace LogicSystem