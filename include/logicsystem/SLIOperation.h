#ifndef LOGIC_SYSTEM_SLI_OPERATION_H
#define LOGIC_SYSTEM_SLI_OPERATION_H

#include "SLITree.h"
#include "Literal.h"
#include "Clause.h"
#include <memory>
#include <variant>

namespace LogicSystem
{

    // SLI操作类型枚举
    enum class SLIActionType
    {
        EXTENSION,
        FACTORING,
        ANCESTRY,
        TRUNCATE
    };

    inline std::string SLI_Action_to_string(SLIActionType action)
    {
        switch (action)
        {
        case SLIActionType::EXTENSION:
            return "EXTENSION";
        case SLIActionType::FACTORING:
            return "FACTORING";
        case SLIActionType::ANCESTRY:
            return "ANCESTRY";
        case SLIActionType::TRUNCATE:
            return "TRUNCATE";
        default:
            return "UNKNOWN"; // 防止未定义的行为
        }
    }

    // 第二操作数的变体类型：可以是SLI节点指针或文字
    using SecondOperand = std::variant<std::shared_ptr<SLINode>, Literal>;

    class SLIOperation
    {
    public:
        // 操作状态结构
        struct OperationState
        {
            std::shared_ptr<SLITree> sli_tree;      // SLI树
            SLIActionType action;                   // 操作类型
            std::shared_ptr<SLINode> lit1_node;     // 第一个节点
            SecondOperand second_op;                // 第二个操作数
            Clause kb_clause;                       // 知识库子句（用于extension）
            std::shared_ptr<OperationState> parent; // 父状态
            int state_id;                           // 状态ID

            OperationState(std::shared_ptr<SLITree> tree,
                           SLIActionType act,
                           std::shared_ptr<SLINode> l1,
                           SecondOperand second,
                           Clause clause = Clause(), // 默认是一个空的Clause
                           std::shared_ptr<OperationState> p = nullptr)
                : sli_tree(tree),
                  action(act),
                  lit1_node(l1),
                  second_op(second),
                  kb_clause(clause),
                  parent(p)
            {
                static int next_id = 0;
                state_id = next_id++;
            }
        };
        static std::shared_ptr<OperationState> deepCopyOperationState(
            const std::shared_ptr<OperationState> &original_state)
        {
            // 深拷贝 SLITree
            auto new_sli_tree = original_state->sli_tree->deepCopy();

            // 在新树中找到对应的新节点
            auto new_lit1_node = new_sli_tree->findNodeById(original_state->lit1_node->node_id);

            // 深拷贝第二个操作数
            SecondOperand new_second_op;

            if (isNode(original_state->second_op))
            {
                // 如果是 SLINode，进行深拷贝
                auto original_node = getNode(original_state->second_op);
                auto new_node = new_sli_tree->findNodeById(original_node->node_id);
                new_second_op = SecondOperand(new_node);
            }
            else if (isLiteral(original_state->second_op))
            {
                // 如果是 Literal，直接拷贝
                auto original_literal = getLiteral(original_state->second_op);
                new_second_op = SecondOperand(original_literal);
            }

            // 创建新的 OperationState
            auto new_state = std::make_shared<OperationState>(
                new_sli_tree,
                original_state->action,
                new_lit1_node,
                new_second_op,
                original_state->kb_clause,
                original_state->parent);
            return new_state;
        }

        // 创建extension操作状态
        static std::shared_ptr<OperationState> createExtensionState(
            std::shared_ptr<SLITree> tree,
            std::shared_ptr<SLINode> resolving_node,
            const Literal &kb_literal,
            Clause clause,
            std::shared_ptr<OperationState> parent = nullptr)
        {
            return std::make_shared<OperationState>(
                tree,
                SLIActionType::EXTENSION,
                resolving_node,
                SecondOperand(kb_literal),
                clause,
                parent);
        }

        // 创建factoring操作状态
        static std::shared_ptr<OperationState> createFactoringState(
            std::shared_ptr<SLITree> tree,
            std::shared_ptr<SLINode> upper_node,
            std::shared_ptr<SLINode> lower_node,
            std::shared_ptr<OperationState> parent = nullptr)
        {
            return std::make_shared<OperationState>(
                tree,
                SLIActionType::FACTORING,
                upper_node,
                SecondOperand(lower_node),
                Clause(),
                parent);
        }

        // 创建ancestry操作状态
        static std::shared_ptr<OperationState> createAncestryState(
            std::shared_ptr<SLITree> tree,
            std::shared_ptr<SLINode> upper_node,
            std::shared_ptr<SLINode> lower_node,
            std::shared_ptr<OperationState> parent = nullptr)
        {
            return std::make_shared<OperationState>(
                tree,
                SLIActionType::ANCESTRY,
                upper_node,
                SecondOperand(lower_node),
                Clause(),
                parent);
        }

        // 创建truncate操作状态
        static std::shared_ptr<OperationState> createTruncateState(
            std::shared_ptr<SLITree> tree,
            std::shared_ptr<SLINode> truncate_node,
            std::shared_ptr<OperationState> parent = nullptr)
        {
            return std::make_shared<OperationState>(
                tree,
                SLIActionType::TRUNCATE,
                truncate_node,
                SecondOperand(std::shared_ptr<SLINode>(nullptr)), // 空节点
                Clause(),
                parent);
        }

        // 辅助函数：检查第二个操作数类型
        static bool isNode(const SecondOperand &op)
        {
            return std::holds_alternative<std::shared_ptr<SLINode>>(op);
        }

        static bool isLiteral(const SecondOperand &op)
        {
            return std::holds_alternative<Literal>(op);
        }

        // 辅助函数：获取第二个操作数的值
        static std::shared_ptr<SLINode> getNode(const SecondOperand &op)
        {
            return std::get<std::shared_ptr<SLINode>>(op);
        }

        static Literal getLiteral(const SecondOperand &op)
        {
            return std::get<Literal>(op);
        }

        // 获取操作类型的字符串表示
        static std::string getActionString(SLIActionType action)
        {
            switch (action)
            {
            case SLIActionType::EXTENSION:
                return "Extension";
            case SLIActionType::FACTORING:
                return "Factoring";
            case SLIActionType::ANCESTRY:
                return "Ancestry";
            case SLIActionType::TRUNCATE:
                return "Truncate";
            default:
                return "Unknown";
            }
        }

        // 用于序列化的方法
        static std::string serializeState(const std::shared_ptr<OperationState> &state)
        {
            std::string result = "State " + std::to_string(state->state_id) + ": ";
            // result += getActionString(state->action) + " | ";
            // result += "Node1: " + std::to_string(state->lit1_node->node_id) + " | ";

            // if (isNode(state->second_op))
            // {
            //     auto node = getNode(state->second_op);
            //     result += "Node2: " + std::to_string(node ? node->node_id : -1);
            // }
            // else
            // {
            //     result += "Literal: " + getLiteral(state->second_op).toString(*(state->sli_tree->getKnowledgeBase()));
            // }

            return result;
        }

        // 获取操作路径（从根到当前状态）
        static std::vector<std::shared_ptr<OperationState>> getOperationPath(
            const std::shared_ptr<OperationState> &state)
        {
            std::vector<std::shared_ptr<OperationState>> path;
            auto current = state;
            while (current)
            {
                path.push_back(current);
                current = current->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }
    };

} // namespace LogicSystem

#endif // LOGIC_SYSTEM_SLI_OPERATION_H