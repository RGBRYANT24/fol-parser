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
            std::cout << "KB clause: "<<state->kb_clause.toString(kb) <<std::endl;
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

} // namespace LogicSystem