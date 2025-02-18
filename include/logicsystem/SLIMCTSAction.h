#ifndef LOGIC_SYSTEM_SLI_MCTS_ACTION_H
#define LOGIC_SYSTEM_SLI_MCTS_ACTION_H

#include "SLIOperation.h" // 包含 SLIActionType, SecondOperand, SLI_Action_to_string 等定义
#include "SLINode.h"
#include "Clause.h"
#include "KnowledgeBase.h"
#include <memory>
#include <sstream>
#include <string>

namespace LogicSystem
{

    /**
     * @brief MCTS 搜索中使用的组合动作。
     *
     * 该动作封装了 SLIOperation::OperationState 中需要的所有信息：
     * - SLIActionType action:         操作类型（如 t-extension、t-factoring 等）
     * - std::shared_ptr<SLINode> lit1_node:  第一个节点
     * - SecondOperand second_op:      第二个操作数（可以是 Literal 或 SLINode 指针）
     * - Clause kb_clause:             用于扩展操作的知识库子句
     */
    class SLIMCTSAction
    {
    public:
        SLIActionType action;               // 操作类型
        std::shared_ptr<SLINode> lit1_node; // 第一个节点
        SecondOperand second_op;            // 第二个操作数
        Clause kb_clause;                   // 知识库子句（用于 extension 操作）

        /// 构造函数
        SLIMCTSAction(SLIActionType action,
                      std::shared_ptr<SLINode> lit1_node,
                      const SecondOperand &second_op,
                      const Clause &kb_clause)
            : action(action), lit1_node(lit1_node), second_op(second_op), kb_clause(kb_clause)
        {
        }

        /// 默认构造函数
        SLIMCTSAction() = default;

        SLIMCTSAction &operator=(const SLIMCTSAction &rhs)
        {
            if (this != &rhs)
            {
                // 可以在这里加入调试信息，例如：
                if (!rhs.lit1_node)
                {
                    std::cerr << "Warning: rhs.lit1_node is NULL! with action type " << SLI_Action_to_string(rhs.action) << std::endl;
                }

                // 检查 second_op 类型
                if (std::holds_alternative<std::shared_ptr<SLINode>>(rhs.second_op))
                {
                    auto ptr = std::get<std::shared_ptr<SLINode>>(rhs.second_op);
                    if (!ptr && rhs.action != SLIActionType::TRUNCATE)
                    {
                        std::cerr << "Warning: rhs.second_op holds a nullptr for SLINode! with action type " << SLI_Action_to_string(rhs.action) << std::endl;
                    }
                }
                else if (std::holds_alternative<Literal>(rhs.second_op))
                {
                    // 如果需要，可以对 Literal 做额外检查
                    const auto &lit = std::get<Literal>(rhs.second_op);
                    // 例如根据某些判断条件检测 lit 是否处于合理状态
                }

                action = rhs.action;
                lit1_node = rhs.lit1_node;
                second_op = rhs.second_op;
                kb_clause = rhs.kb_clause;
            }
            return *this;
        }

        /**
         * @brief 获取动作的字符串表示（可用于调试）。
         *
         * @param kb 知识库对象，用以转换 Literal 和 Clause 为字符串。
         * @return std::string 动作信息字符串
         */
        std::string to_string(const KnowledgeBase &kb) const
        {
            std::ostringstream oss;
            oss << "Action: " << SLI_Action_to_string(action);
            oss << ", lit1_node: ";
            if (lit1_node)
                oss << lit1_node->node_id;
            else
                oss << "NULL";
            oss << ", second_op: ";
            if (std::holds_alternative<Literal>(second_op))
            {
                oss << std::get<Literal>(second_op).toString(kb);
            }
            else if (std::holds_alternative<std::shared_ptr<SLINode>>(second_op))
            {
                auto node = std::get<std::shared_ptr<SLINode>>(second_op);
                oss << (node ? std::to_string(node->node_id) : "NULL");
            }
            oss << ", kb_clause: " << kb_clause.toString(kb);
            return oss.str();
        }
    };

} // namespace LogicSystem

#endif // LOGIC_SYSTEM_SLI_MCTS_ACTION_H