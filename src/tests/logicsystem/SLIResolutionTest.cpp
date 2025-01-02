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

        // 生成Extension状态
        SLIResolution::generateExtensionStates(kb, b_literals, current_state, state_stack);

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
        SLIResolution::generateExtensionStates(kb, b_literals, current_state, state_stack);
        EXPECT_LT(state_stack.size(), expected_states);

        // 测试A-literal节点
        b_literals[0]->is_active = true;
        b_literals[0]->is_A_literal = true;
        state_stack = std::stack<std::shared_ptr<SLIOperation::OperationState>>();
        SLIResolution::generateExtensionStates(kb, b_literals, current_state, state_stack);
        EXPECT_LT(state_stack.size(), expected_states);

        // 测试空的B-literals列表
        std::vector<std::shared_ptr<SLINode>> empty_b_literals;
        state_stack = std::stack<std::shared_ptr<SLIOperation::OperationState>>();
        SLIResolution::generateExtensionStates(kb, empty_b_literals, current_state, state_stack);
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

} // namespace LogicSystem