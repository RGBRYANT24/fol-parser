#ifndef LOGIC_SYSTEM_SLI_MCTS_ACTION_H
#define LOGIC_SYSTEM_SLI_MCTS_ACTION_H

#include "SLIOperation.h" // 包含 SLIActionType, SLI_Action_to_string 等定义
#include "Clause.h"
#include "KnowledgeBase.h"
#include <variant>
#include <sstream>
#include <string>

namespace LogicSystem
{
    // 定义新的操作数类型：使用 int 表示节点ID，Literal 保持不变
    using MCTSSecondOperand = std::variant<int, Literal>;

    /**
     * @brief MCTS 搜索中使用的组合动作（改进版）
     *
     * 该动作不再直接保存节点指针，而是保存节点的 ID，
     * 从而在 apply_action 时通过 SLITree 的查找函数转换成当前状态下有效的节点指针。
     */
    class SLIMCTSAction
    {
    public:
        SLIActionType action;  // 操作类型
        int lit1_node_id;      // 第一个节点的 ID（替代 shared_ptr<SLINode>）
        MCTSSecondOperand second_op; // 第二个操作数：可以是节点 ID（int）或 Literal
        Clause kb_clause;      // 用于 extension 操作的知识库子句

        /// 构造函数
        SLIMCTSAction(SLIActionType action,
                      int lit1_node_id,
                      const MCTSSecondOperand &second_op,
                      const Clause &kb_clause)
            : action(action),
              lit1_node_id(lit1_node_id),
              second_op(second_op),
              kb_clause(kb_clause)
        {
        }

        /// 默认构造函数
        SLIMCTSAction() = default;

        // 默认的拷贝赋值操作符
        SLIMCTSAction &operator=(const SLIMCTSAction &rhs) = default;

        /**
         * @brief 获取动作的字符串表示（用于调试）。
         */
        std::string to_string(const KnowledgeBase &kb) const
        {
            std::ostringstream oss;
            oss << "Action: " << SLI_Action_to_string(action);
            oss << ", lit1_node_id: " << lit1_node_id;
            oss << ", second_op: ";
            if (std::holds_alternative<Literal>(second_op))
            {
                oss << std::get<Literal>(second_op).toString(kb);
            }
            else if (std::holds_alternative<int>(second_op))
            {
                oss << std::get<int>(second_op);
            }
            oss << ", kb_clause: " << kb_clause.toString(kb);
            return oss.str();
        }
    };

} // namespace LogicSystem

#endif // LOGIC_SYSTEM_SLI_MCTS_ACTION_H