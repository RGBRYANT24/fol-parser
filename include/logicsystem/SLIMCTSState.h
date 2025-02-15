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
        /**
         * @brief 深拷贝构造函数：调用 SLITree::deepCopy 实现状态独立性。
         *
         * @param other 原状态
         */
        SLIMCTSState(const SLIMCTSState &other)
        {
            sli_tree = other.sli_tree->deepCopy();
        }
        /**
         * @brief 新增构造函数，接受一个 SLITree 的 shared_ptr，并深拷贝该树。
         *
         * @param tree 需要深拷贝的 SLITree
         */
        SLIMCTSState(std::shared_ptr<SLITree> tree)
        {
            if (tree)
            {
                sli_tree = tree->deepCopy();
            }
        }

        /**
         * @brief 判断是否为终局状态。
         *
         * 根据具体的 SLI 算法，当 SLITree 中没有扩展候选（例如 get_all_B_literals() 返回空）
         * 或者验证节点合法性失败（例如出现自环）时，可认为当前状态为终局状态。
         *
         * @return true 如果当前状态为终局状态；false 否则
         */
        bool is_terminal() const
        {
            // 1. 首先可以判断基本条件，比如候选扩展是否为空、节点是否合法
            bool basic_check = (this->sli_tree->get_all_B_literals().empty() && this->sli_tree->validateAllNodes());

            // 如果基本条件不满足，则直接返回 false（状态还可继续扩展）
            if (!basic_check)
            {
                return false;
            }

            // 2. 再生成当前状态下所有可执行的候选动作
            std::vector<SLIMCTSAction> actions;
            get_actions(actions);

            // 3. 根据候选动作集是否为空来判断是否处于终局状态
            return actions.empty();
        }

        /**
         * @brief 返回决策者 id。
         *
         * 对于证明问题通常只有一方，返回 0 即可。
         *
         * @return int 决策者 id
         */
        int agent_id() const
        {
            return 0;
        }

        /**
         * @brief 应用动作，将当前状态根据传入的 SLIMCTSAction 更新为新的状态。
         *
         * 为了保证父状态不被修改，扩展子节点时应在深拷贝后的状态上执行动作。
         * 注意不同操作对应的参数类型：
         * - EXTENSION 操作要求 second_op 为 Literal 类型
         * - FACTORING / ANCESTRY 操作要求 second_op 为 std::shared_ptr<SLINode>
         * - TRUNCATE 操作通常只依赖 lit1_node（此处参数为空）
         *
         * @param action 组合动作，包含 action、lit1_node、second_op 和 kb_clause
         */
        void apply_action(const SLIMCTSAction &action)
        {
            switch (action.action)
            {
            case SLIActionType::EXTENSION:
            {
                if (SLIOperation::isLiteral(action.second_op))
                {
                    auto kb_lit = SLIOperation::getLiteral(action.second_op);
                    auto new_nodes = sli_tree->add_node(action.kb_clause,
                                                        kb_lit,
                                                        true,
                                                        action.lit1_node);
                }
                break;
            }
            case SLIActionType::FACTORING:
            {
                if (SLIOperation::isNode(action.second_op))
                {
                    auto second_node = SLIOperation::getNode(action.second_op);
                    sli_tree->t_factoring(action.lit1_node, second_node);
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
                sli_tree->truncate(action.lit1_node);
                break;
            }
            default:
                break;
            }
        }
        // 此处还可以更新 sli_tree 其他状态信息（例如深度、节点计数等）

        /**
         * @brief 生成当前状态下所有可选的组合动作集合 A(s) = {(op,p) | op ∈ 𝒪, p ∈ P₍op₎(s)}.
         *
         * 本方法分别调用扩展、factoring、ancestry、truncate 对应的候选生成逻辑：
         *
         * 1. 对于 EXTENSION：
         *    遍历 sli_tree->get_all_B_literals() 得到候选 b-lit 节点，
         *    若节点活跃且不是 A-literal，再遍历 KnowledgeBase 中所有 Clause 及其 Literals，
         *    对满足 Resolution::isComplementary 与 Unifier::findMGU 条件的候选，
         *    生成动作，参数为：
         *      - action: EXTENSION
         *      - lit1_node: 候选 b-lit 节点
         *      - second_op: Literal（目标文字）
         *      - kb_clause: 对应 Clause
         *
         * 2. 对于 FACTORING：
         *    调用 SLIResolution::findPotentialFactoringPairs(sli_tree) 得到候选对，
         *    对于每个候选对 (upper, lower)，生成动作：
         *      - action: FACTORING
         *      - lit1_node: upper
         *      - second_op: lower
         *      - kb_clause: 空 Clause()
         *
         * 3. 对于 ANCESTRY：
         *    类似于 FACTORING，调用 SLIResolution::findPotentialAncestryPairs(sli_tree)；
         *
         * 4. 对于 TRUNCATE：
         *    调用 SLIResolution::findPotentialTruncateNodes(sli_tree) 得到候选节点，
         *    对每个候选生成动作：
         *      - action: TRUNCATE
         *      - lit1_node: 该节点
         *      - second_op: 空（即 nullptr）
         *      - kb_clause: 空 Clause()
         *
         * @param actions 用传引用方式返回所有生成的 SLIMCTSAction 动作
         */
        // 生成 EXTENSION 操作的状态
        void generateMCTSExtensionStates(std::vector<SLIMCTSAction> &actions) const
        {
            // 从 SLITree 获取 KnowledgeBase
            KnowledgeBase kb = sli_tree->getKB();
            // 获取所有候选 b-lit 节点
            auto b_lit_nodes = sli_tree->get_all_B_literals();
            for (auto &node : b_lit_nodes)
            {
                if (!node->is_active || node->is_A_literal)
                    continue;
                // 遍历知识库中的所有 Clause
                for (const auto &kb_clause : kb.getClauses())
                {
                    // 遍历 Clause 中所有 Literal
                    for (const auto &lit : kb_clause.getLiterals())
                    {
                        if (Resolution::isComplementary(node->literal, lit) &&
                            Unifier::findMGU(node->literal, lit, kb))
                        {
                            // 生成 EXTENSION 操作：用候选 b-lit 节点作为 lit1_node，
                            // 目标文字作为 second_op，kb_clause 为当前的 Clause
                            actions.emplace_back(SLIActionType::EXTENSION,
                                                 node,
                                                 SecondOperand(lit),
                                                 kb_clause);
                        }
                    }
                }
            }
        }

        // 生成 FACTORING 操作的状态
        void generateMCTSFactoringStates(std::vector<SLIMCTSAction> &actions) const
        {
            auto factoring_pairs = SLIResolution::findPotentialFactoringPairs(sli_tree);
            for (const auto &pair : factoring_pairs)
            {
                actions.emplace_back(SLIActionType::FACTORING,
                                     pair.first,                 // upper_node 作为 lit1_node
                                     SecondOperand(pair.second), // lower_node 作为 second_op
                                     Clause());                  // kb_clause 为空
            }
        }

        // 生成 ANCESTRY 操作的状态
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

        // 针对 MCTS 的 TRUNCATE 操作生成函数，采用所有 active 节点作为候选
        void generateMCTSTruncateStates(std::vector<SLIMCTSAction> &actions) const
        {
            auto active_nodes = sli_tree->get_all_active_nodes();
            for (auto &node : active_nodes)
            {
                actions.emplace_back(SLIActionType::TRUNCATE,
                                     node,
                                     SecondOperand(std::shared_ptr<SLINode>(nullptr)),
                                     Clause());
            }
        }

        // 根据当前状态的 AC 与 MC 条件生成所有候选操作
        void get_actions(std::vector<SLIMCTSAction> &actions) const
        {
            // 检查所有节点的 AC 与 MC 条件
            bool AC_result = sli_tree->check_all_nodes_AC();
            bool MC_result = sli_tree->check_all_nodes_MC();

            if (AC_result && MC_result)
            {
                // 同时满足 AC 与 MC 条件：生成 EXTENSION、FACTORING、ANCESTRY 与 MCTS-TRUNCATE 操作
                generateMCTSExtensionStates(actions);
                generateMCTSFactoringStates(actions);
                generateMCTSAncestryStates(actions);
                generateMCTSTruncateStates(actions);
            }
            else if (MC_result)
            {
                // 仅满足 MC 条件：只生成 FACTORING 与 ANCESTRY 操作
                generateMCTSFactoringStates(actions);
                generateMCTSAncestryStates(actions);
            }
            else if (AC_result)
            {
                // 仅满足 AC 条件：只生成 MCTS-TRUNCATE 操作
                generateMCTSTruncateStates(actions);
            }
            else
            {
                // 当既不满足 AC 也不满足 MC 条件时，不生成任何操作
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

        /**
         * @brief 返回叶子状态的原始评价（奖励向量）。
         *
         * 例如，在证明成功时返回 +1，在证明失败时返回 -1，其它情况返回 0
         * （奖励向量大小为 1）。
         *
         * @return std::vector<float> 奖励向量
         */
        std::vector<float> evaluate() const
        {
            std::vector<float> rewards(1, 0.0f);
            if (is_terminal())
                rewards[0] = 1.0f;
            else
                rewards[0] = 0.0f;
            return rewards;
        }

        /**
         * @brief 返回状态的字符串描述，用于调试输出。
         *
         * @return std::string 状态描述字符串
         */
        std::string to_string() const
        {
            return "SLIMCTSState: " + sli_tree->printBLiteralsAsClause();
        }
    };

} // namespace LogicSystem

#endif // LOGIC_SYSTEM_SLI_MCTS_STATE_H