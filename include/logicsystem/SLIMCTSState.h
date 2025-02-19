#ifndef LOGIC_SYSTEM_SLI_MCTS_STATE_H
#define LOGIC_SYSTEM_SLI_MCTS_STATE_H

#include "SLITree.h"
#include "SLIMCTSAction.h" // 注意这里包含的是改进后的 SLIMCTSAction.h
#include "SLIResolution.h" // 假定静态辅助函数都在这里声明
#include "Resolution.h"    // 用于 isComplementary 等判断
#include "Unifier.h"
#include "KnowledgeBase.h"
// 假设你有或借用 IState 接口要求的头文件
#include "IState.h"
#include <vector>
#include <random>
#include <optional>
#include <iostream> // 用于调试输出

namespace LogicSystem
{

    /**
     * @brief 专门用于 MCTS 搜索的 SLI 状态类
     *
     * 该状态类封装了整个证明问题的搜索状态，通过内部持有独立的 SLITree
     * 实例表示。状态转换时采用深拷贝策略，保证子状态与父状态独立。
     */
    class SLIMCTSState /* : public IState */
    {
    public:
        // 当前 SLI 算法搜索状态，由 SLITree 表示（保存部分证明状态）
        std::shared_ptr<SLITree> sli_tree;

        // 拷贝构造函数，确保深拷贝 SLITree
        SLIMCTSState(const SLIMCTSState &other)
        {
            sli_tree = other.sli_tree->deepCopy();
        }
        // 构造函数：从给定 SLITree 创建状态（深拷贝）
        SLIMCTSState(std::shared_ptr<SLITree> tree)
        {
            if (tree)
            {
                sli_tree = tree->deepCopy();
            }
        }

        SLIMCTSState &operator=(const SLIMCTSState &other)
        {
            if (this != &other)
            {
                if (other.sli_tree)
                    sli_tree = other.sli_tree->deepCopy();
                else
                    sli_tree.reset();
            }
            return *this;
        }

        // 新增方法：根据当前状态和动作生成下一个状态，保证深拷贝
        SLIMCTSState next_state(const SLIMCTSAction &action) const
        {
            SLIMCTSState new_state(*this);
            new_state.apply_action(action);
            return new_state;
        }

        // 原始终局状态判断函数（保留原有逻辑，用于其他用途）
        bool is_terminal_original() const
        {
            if (this->sli_tree->get_all_active_nodes().empty())
                return true;
            if (!this->sli_tree->validateAllNodes())
                return true;
            std::vector<SLIMCTSAction> actions;
            get_actions(actions);
            return actions.empty();
        }

        /**
         * @brief 计算当前状态是否为终局，并返回对应的奖励（如果不是终局则返回空值）
         *
         * 逻辑为：
         * - 如果没有活动节点，则认为终局，奖励为 +1；
         * - 如果节点不合法，则认为终局，奖励为 -1；
         * - 如果无候选动作，则认为终局，奖励为 -1；
         * - 否则返回空值，表示非终局状态。
         */
        std::optional<float> compute_terminal_reward() const
        {
            // 条件1：没有活动节点 → 奖励 +1
            auto active_nodes = this->sli_tree->get_all_active_nodes();
            if (active_nodes.empty())
            {
                return 1.0f;
            }

            // 条件2：节点不合法 → 奖励 -1
            if (!this->sli_tree->validateAllNodes())
            {
                return -1.0f;
            }

            // 条件3：无候选动作 → 奖励 -1
            std::vector<SLIMCTSAction> actions;
            get_actions(actions);
            if (actions.empty())
            {
                return -1.0f;
            }

            // 非终局状态
            return std::nullopt;
        }

        /**
         * @brief 通过辅助函数判断当前状态是否为终局状态
         */
        bool is_terminal() const
        {
            return compute_terminal_reward().has_value();
        }

        /**
         * @brief 评估状态并返回奖励
         *
         * 如果终局，则返回对应奖励；否则返回 0。
         */
        std::vector<float> evaluate() const
        {
            std::vector<float> rewards(1, 0.0f);
            auto terminalReward = compute_terminal_reward();
            if (terminalReward.has_value())
                rewards[0] = terminalReward.value();
            else
                rewards[0] = 0.0f;
            return rewards;
        }

        int agent_id() const
        {
            return 0;
        }

        /**
         * @brief 应用动作（改进版）
         *
         * 通过动作中的节点ID，利用 SLITree::findNodeById 转换成当前状态有效的节点指针，
         * 然后执行具体操作。
         */
        void apply_action(const SLIMCTSAction &action)
        {
            auto kb = sli_tree->getKB();
            // 根据动作中的 lit1_node_id 获取当前状态下对应的节点指针
            auto parent_node = sli_tree->findNodeById(action.lit1_node_id);
            if (!parent_node)
            {
                std::cerr << "apply_action: 找不到节点ID " << action.lit1_node_id << "\n";
                return;
            }

            switch (action.action)
            {
            case SLIActionType::EXTENSION:
            {
                if (std::holds_alternative<Literal>(action.second_op))
                {
                    Literal lit = std::get<Literal>(action.second_op);
                    auto kb_lit = SLIOperation::getLiteral(lit);
                    auto new_nodes = sli_tree->add_node(action.kb_clause,
                                                        kb_lit,
                                                        true,
                                                        parent_node);
                }
                break;
            }
            case SLIActionType::FACTORING:
            {
                if (std::holds_alternative<int>(action.second_op))
                {
                    int node_id = std::get<int>(action.second_op);
                    auto second_node = sli_tree->findNodeById(node_id);
                    if (second_node)
                    {
                        // std::cout << "before apply_action (factoring): " << action.to_string(kb) << std::endl;
                        // sli_tree->print_tree(kb);
                        sli_tree->t_factoring(parent_node, second_node);
                        // std::cout << "after apply_action (factoring): " << std::endl;
                        // sli_tree->print_tree(kb);
                    }
                }
                break;
            }
            case SLIActionType::ANCESTRY:
            {
                if (std::holds_alternative<int>(action.second_op))
                {
                    int node_id = std::get<int>(action.second_op);
                    auto second_node = sli_tree->findNodeById(node_id);
                    if (second_node)
                    {
                        sli_tree->t_ancestry(parent_node, second_node);
                    }
                }
                break;
            }
            case SLIActionType::TRUNCATE:
            {
                sli_tree->truncate(parent_node);
                break;
            }
            default:
                break;
            }
        }

        // 生成候选动作，下面函数中均使用节点 ID 替代指针传递
        void generateMCTSExtensionStates(std::vector<SLIMCTSAction> &actions) const
        {
            KnowledgeBase kb = sli_tree->getKB();
            auto b_lit_nodes = sli_tree->get_all_B_literals();
            int count = 1;
            for (auto &node : b_lit_nodes)
            {
                if (!node->is_active || node->is_A_literal)
                    continue;
                for (const auto &kb_clause : kb.getClauses())
                {
                    for (const auto &lit : kb_clause.getLiterals())
                    {
                        if (Resolution::isComplementary(node->literal, lit) &&
                            Unifier::findMGU(node->literal, lit, kb))
                        {

                            // std::cout << "generateMCTSExtensionStates found complementary count " << count++ << std::endl;
                            //         std::cout << "generateMCTSExtensionStates sli node " << std::endl;
                            //         node->print(kb);
                            //         std::cout << "generateMCTSExtensionStates kb lit " << lit.toString(kb) << " clause " << kb_clause.toString(kb) << std::endl;
                            // 改为使用节点ID进行传递, 同时构造时传入 MCTSSecondOperand（后者底层类型为 std::variant<int, Literal>）
                            actions.emplace_back(SLIActionType::EXTENSION,
                                                 node->node_id,
                                                 MCTSSecondOperand(lit),
                                                 kb_clause);
                        }
                    }
                }
            }
        }

        void generateMCTSFactoringStates(std::vector<SLIMCTSAction> &actions) const
        {
            auto factoring_pairs = SLIResolution::findPotentialFactoringPairs(sli_tree);
            for (const auto &pair : factoring_pairs)
            {
                actions.emplace_back(SLIActionType::FACTORING,
                                     pair.first->node_id,
                                     MCTSSecondOperand(pair.second->node_id),
                                     Clause());
            }
        }

        void generateMCTSAncestryStates(std::vector<SLIMCTSAction> &actions) const
        {
            auto ancestry_pairs = SLIResolution::findPotentialAncestryPairs(sli_tree);
            for (const auto &pair : ancestry_pairs)
            {
                actions.emplace_back(SLIActionType::ANCESTRY,
                                     pair.first->node_id,
                                     MCTSSecondOperand(pair.second->node_id),
                                     Clause());
            }
        }

        void generateMCTSTruncateStates(std::vector<SLIMCTSAction> &actions) const
        {
            auto truncate_nodes = SLIResolution::findPotentialTruncateNodes(sli_tree);
            for (auto &node : truncate_nodes)
            {
                // TRUNCATE 操作只依赖于第一个节点，因此第二操作数可用占位值（例如0）表示
                actions.emplace_back(SLIActionType::TRUNCATE,
                                     node->node_id,
                                     MCTSSecondOperand(0),
                                     Clause());
            }
        }

        void get_actions(std::vector<SLIMCTSAction> &actions) const
        {
            bool AC_result = sli_tree->check_all_nodes_AC();
            bool MC_result = sli_tree->check_all_nodes_MC();
            if (AC_result && MC_result)
            {
                generateMCTSExtensionStates(actions);
                generateMCTSFactoringStates(actions);
                generateMCTSAncestryStates(actions);
                generateMCTSTruncateStates(actions);
            }
            else if (MC_result)
            {
                generateMCTSFactoringStates(actions);
                generateMCTSAncestryStates(actions);
            }
            else if (AC_result)
            {
                generateMCTSTruncateStates(actions);
            }
        }

        /**
         * @brief 从候选动作中随机返回一个动作（用于模拟阶段）
         */
        bool get_random_action(SLIMCTSAction &action) const
        {
            std::vector<SLIMCTSAction> actions;
            get_actions(actions);
            if (actions.empty())
                return false;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<size_t> dis(0, actions.size() - 1);
            action = actions[dis(gen)];
            return true;
        }

        std::string to_string() const
        {
            return "SLIMCTSState: " + sli_tree->printBLiteralsAsClause();
        }
    };

} // namespace LogicSystem

#endif // LOGIC_SYSTEM_SLI_MCTS_STATE_H