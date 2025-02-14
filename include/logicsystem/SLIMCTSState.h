#ifndef LOGIC_SYSTEM_SLI_MCTS_STATE_H
#define LOGIC_SYSTEM_SLI_MCTS_STATE_H

#include "SLITree.h"
#include "SLIMCTSAction.h"
#include "SLIResolution.h"  // 假定静态辅助函数都在这里声明
#include "Resolution.h"     // 用于 isComplementary 等判断
#include "Unifier.h"
#include "KnowledgeBase.h"
// 假设你有或借用 IState 接口要求的头文件
#include "IState.h" 
#include <vector>
#include <random>
#include <optional>

namespace LogicSystem {

    /**
     * @brief 专门用于 MCTS 搜索的 SLI 状态类
     *
     * 该状态类封装了整个证明问题的搜索状态，通过内部持有独立的 SLITree
     * 实例表示。状态转换时采用深拷贝策略，保证子状态与父状态独立。
     */
    class SLIMCTSState /* : public IState */ {
    public:
        // 当前 SLI 算法搜索状态，由 SLITree 表示（保存部分证明状态）
        std::shared_ptr<SLITree> sli_tree;

        /**
         * @brief 基于 KnowledgeBase 构造初始状态。
         *
         * @param kb 知识库对象
         */
        SLIMCTSState(KnowledgeBase &kb) {
            sli_tree = std::make_shared<SLITree>(kb);
        }

        /**
         * @brief 深拷贝构造函数：调用 SLITree::deepCopy 实现状态独立性。
         *
         * @param other 原状态
         */
        SLIMCTSState(const SLIMCTSState &other) {
            sli_tree = other.sli_tree->deepCopy();
        }

        /**
         * @brief 判断是否为终局状态。
         *
         * 根据具体的 SLI 算法，当 SLITree 中没有扩展候选（例如 get_all_B_literals() 返回空）
         * 或者验证节点合法性失败（例如出现自环）时，可认为当前状态为终局状态。
         *
         * @return true 如果当前状态为终局状态；false 否则
         */
        bool is_terminal() const {
            // 1. 首先可以判断基本条件，比如候选扩展是否为空、节点是否合法
            bool basic_check = (this->sli_tree->get_all_B_literals().empty() && this->sli_tree->validateAllNodes());
        
            // 如果基本条件不满足，则直接返回 false（状态还可继续扩展）
            if (!basic_check) {
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
        int agent_id() const {
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
        void apply_action(const SLIMCTSAction &action) {
            switch (action.action) {
            case SLIActionType::EXTENSION:
            {
                if (std::holds_alternative<Literal>(action.second_op)) {
                    Literal targetLiteral = std::get<Literal>(action.second_op);
                    auto new_nodes = sli_tree->add_node(action.kb_clause, targetLiteral, true, action.lit1_node);
                }
                break;
            }
            case SLIActionType::FACTORING:
            {
                if (std::holds_alternative<std::shared_ptr<SLINode>>(action.second_op)) {
                    auto second_node = std::get<std::shared_ptr<SLINode>>(action.second_op);
                    sli_tree->t_factoring(action.lit1_node, second_node);
                }
                break;
            }
            case SLIActionType::ANCESTRY:
            {
                if (std::holds_alternative<std::shared_ptr<SLINode>>(action.second_op)) {
                    auto second_node = std::get<std::shared_ptr<SLINode>>(action.second_op);
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
            // 此处还可以更新 sli_tree 其他状态信息（例如深度、节点计数等）
        }

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
        void get_actions(std::vector<SLIMCTSAction> &actions) const {
            // 1. EXTENSION 动作生成
            {
                auto b_lit_nodes = sli_tree->get_all_B_literals();
                // 从 SLITree 获取 KnowledgeBase（假定 SLITree 提供 getKB() 方法）
                KnowledgeBase kb = sli_tree->getKB();
                for (auto &node : b_lit_nodes) {
                    if (!node->is_active || node->is_A_literal)
                        continue;
                    // 遍历知识库中的所有 Clause
                    for (const auto &kb_clause : kb.getClauses()) {
                        // 遍历 Clause 中所有 Literal
                        for (const auto &lit : kb_clause.getLiterals()) {
                            if (Resolution::isComplementary(node->literal, lit) &&
                                Unifier::findMGU(node->literal, lit, kb)) {
                                // 生成 EXTENSION 动作：用候选 b-lit 节点作为 lit1_node，目标文字作为 second_op，
                                // kb_clause 为当前 Clause
                                actions.emplace_back(SLIActionType::EXTENSION,
                                                     node,
                                                     SecondOperand(lit),
                                                     kb_clause);
                            }
                        }
                    }
                }
            }

            // 2. FACTORING 动作生成
            {
                auto factoring_pairs = SLIResolution::findPotentialFactoringPairs(sli_tree);
                for (const auto &pair : factoring_pairs) {
                    // 根据 SLIResolution 的设计：候选对 (upper, lower)
                    actions.emplace_back(SLIActionType::FACTORING,
                                         pair.first,       // upper_node 作为 lit1_node
                                         SecondOperand(pair.second), // lower_node 作为 second_op
                                         Clause());        // kb_clause 为空
                }
            }

            // 3. ANCESTRY 动作生成
            {
                auto ancestry_pairs = SLIResolution::findPotentialAncestryPairs(sli_tree);
                for (const auto &pair : ancestry_pairs) {
                    actions.emplace_back(SLIActionType::ANCESTRY,
                                         pair.first, 
                                         SecondOperand(pair.second),
                                         Clause());
                }
            }

            // 4. TRUNCATE 动作生成
            {
                auto truncate_nodes = SLIResolution::findPotentialTruncateNodes(sli_tree);
                for (auto &node : truncate_nodes) {
                    actions.emplace_back(SLIActionType::TRUNCATE,
                                         node,
                                         SecondOperand(std::shared_ptr<SLINode>(nullptr)),
                                         Clause());
                }
            }
        }

        /**
         * @brief 从候选动作中随机返回一个动作，用于模拟阶段。
         *
         * @param action 随机选取的动作通过引用返回
         * @return true 如果存在候选动作；false 如果当前状态下没有可选动作
         */
        bool get_random_action(SLIMCTSAction &action) const {
            std::vector<SLIMCTSAction> actions;
            get_actions(actions);
            if (actions.empty()) return false;
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
        std::vector<float> evaluate() const {
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
        std::string to_string() const {
            return "SLIMCTSState: " + sli_tree->printBLiteralsAsClause();
        }
    };

} // namespace LogicSystem

#endif // LOGIC_SYSTEM_SLI_MCTS_STATE_H