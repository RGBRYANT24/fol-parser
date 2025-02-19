// SLIMCTSState.h (部分摘录)
#ifndef LOGIC_SYSTEM_SLI_MCTS_STATE_H
#define LOGIC_SYSTEM_SLI_MCTS_STATE_H

#include "SLITree.h"
#include "SLIMCTSAction.h"
#include "SLIResolution.h" // 假定静态辅助函数都在这里声明
#include "Resolution.h"    // 用于 isComplementary 等判断
#include "Unifier.h"
#include "KnowledgeBase.h"
// 假设你有或借用 IState 接口要求的头文件
#include "IState.h"
#include <vector>
#include <random>
#include <optional>

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
            // std::cout<< "SLIMCTSSTATE DEEPCOPY USED " << std::endl;
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
                // 如果 other 的 sli_tree 非空，则调用 deepCopy()；否则，置空当前的 sli_tree
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
            // 利用拷贝构造函数进行深拷贝
            SLIMCTSState new_state(*this);
            // 在新状态上应用动作
            new_state.apply_action(action);
            return new_state;
        }

        // 判断是否为终局状态
        bool is_terminal() const
        {
            bool basic_check = (this->sli_tree->get_all_B_literals().empty() && this->sli_tree->validateAllNodes());
            if (!basic_check)
            {
                return false;
            }
            std::vector<SLIMCTSAction> actions;
            get_actions(actions);
            return actions.empty();
        }

        int agent_id() const
        {
            return 0;
        }

        // 修改 apply_action，不再在原状态上“逐步”修改状态，而应当通过 next_state 创建新状态
        void apply_action(const SLIMCTSAction &action)
        {
            auto kb = sli_tree->getKB();
            switch (action.action)
            {
            case SLIActionType::EXTENSION:
            {
                // std::cout << "before apply_action specifically extension " << std::endl;
                // sli_tree->print_tree(kb);
                if (SLIOperation::isLiteral(action.second_op))
                {
                    auto kb_lit = SLIOperation::getLiteral(action.second_op);
                    auto new_nodes = sli_tree->add_node(action.kb_clause,
                                                        kb_lit,
                                                        true,
                                                        action.lit1_node);
                }
                // std::cout << "after apply_action specifically extension " << std::endl;
                // sli_tree->print_tree(kb);
                break;
            }
            case SLIActionType::FACTORING:
            {
                if (SLIOperation::isNode(action.second_op))
                {
                    std::cout << "before apply_action specifically factoring " << std::endl;
                    std::cout << action.to_string(kb) << std::endl;
                    sli_tree->print_tree(kb);
                    auto second_node = SLIOperation::getNode(action.second_op);
                    sli_tree->t_factoring(action.lit1_node, second_node);
                    std::cout << "after apply_action specifically factoring " << std::endl;
                    sli_tree->print_tree(kb);
                }
                break;
            }
            case SLIActionType::ANCESTRY:
            {
                if (SLIOperation::isNode(action.second_op))
                {
                    auto second_node = SLIOperation::getNode(action.second_op);
                    sli_tree->t_ancestry(action.lit1_node, second_node);
                }
                break;
            }
            case SLIActionType::TRUNCATE:
            {
                // std::cout << "apply action truncate " << std::endl;
                sli_tree->truncate(action.lit1_node);
                break;
            }
            default:
                break;
            }
        }

        // 生成候选动作（原有代码）
        void generateMCTSExtensionStates(std::vector<SLIMCTSAction> &actions) const
        {
            KnowledgeBase kb = sli_tree->getKB();
            auto b_lit_nodes = sli_tree->get_all_B_literals();
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
                            actions.emplace_back(SLIActionType::EXTENSION,
                                                 node,
                                                 SecondOperand(lit),
                                                 kb_clause);
                        }
                    }
                }
            }
        }

        // 生成 FACTORING、ANCESTRY、TRUNCATE 的候选状态方法同原代码...
        void generateMCTSFactoringStates(std::vector<SLIMCTSAction> &actions) const
        {
            auto factoring_pairs = SLIResolution::findPotentialFactoringPairs(sli_tree);
            for (const auto &pair : factoring_pairs)
            {
                actions.emplace_back(SLIActionType::FACTORING,
                                     pair.first,
                                     SecondOperand(pair.second),
                                     Clause());
            }
        }

        void generateMCTSAncestryStates(std::vector<SLIMCTSAction> &actions) const
        {
            auto ancestry_pairs = SLIResolution::findPotentialAncestryPairs(sli_tree);
            for (const auto &pair : ancestry_pairs)
            {
                actions.emplace_back(SLIActionType::ANCESTRY,
                                     pair.first,
                                     SecondOperand(pair.second),
                                     Clause());
            }
        }

        void generateMCTSTruncateStates(std::vector<SLIMCTSAction> &actions) const
        {
            auto truncate_nodes = SLIResolution::findPotentialTruncateNodes(sli_tree);
            for (auto &node : truncate_nodes)
            {
                actions.emplace_back(SLIActionType::TRUNCATE,
                                     node,
                                     SecondOperand(std::shared_ptr<SLINode>(nullptr)),
                                     Clause());
            }
        }

        void get_actions(std::vector<SLIMCTSAction> &actions) const
        {
            bool AC_result = sli_tree->check_all_nodes_AC();
            bool MC_result = sli_tree->check_all_nodes_MC();
            if (AC_result && MC_result)
            {
                // std::cout << "AC && MC" << std::endl;
                generateMCTSExtensionStates(actions);
                generateMCTSFactoringStates(actions);
                generateMCTSAncestryStates(actions);
                generateMCTSTruncateStates(actions);
            }
            else if (MC_result)
            {
                // std::cout << "MC" << std::endl;
                generateMCTSFactoringStates(actions);
                generateMCTSAncestryStates(actions);
            }
            else if (AC_result)
            {
                // std::cout << "AC" << std::endl;
                generateMCTSTruncateStates(actions);
            }
        }

        /**
         * @brief 从候选动作中随机返回一个动作，用于模拟阶段。
         *
         * @param action 随机选取的动作通过引用返回
         * @return true 如果存在候选动作；false 如果当前状态下没有可选动作
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

        std::vector<float> evaluate() const
        {
            std::vector<float> rewards(1, 0.0f);
            if (is_terminal())
                rewards[0] = 1.0f;
            else
                rewards[0] = 0.0f;
            return rewards;
        }

        std::string to_string() const
        {
            return "SLIMCTSState: " + sli_tree->printBLiteralsAsClause();
        }
    };

} // namespace LogicSystem

#endif // LOGIC_SYSTEM_SLI_MCTS_STATE_H