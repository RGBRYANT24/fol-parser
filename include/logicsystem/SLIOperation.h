#ifndef LOGIC_SYSTEM_SLI_OPERATION_H
#define LOGIC_SYSTEM_SLI_OPERATION_H

#include "SLITree.h"
#include "Literal.h"
#include "Clause.h"
#include <memory>
#include <iostream>
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
            int depth;                              // 深度

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
                depth = (parent) ? parent->depth + 1 : 1;
                // std::cout << "[OperationState] Created state_id: " << state_id << "\n";
            }
        };

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

        // 打印操作路径的函数
        static void printOperationPath(const std::shared_ptr<OperationState> &state, const KnowledgeBase &kb)
        {
            if (!state)
            {
                std::cout << "No operations to display.\n";
                return;
            }

            // Get operation path from root to current state
            std::vector<std::shared_ptr<OperationState>> path = getOperationPath(state);

            std::cout << "\n====== Operation Path ======\n";

            for (size_t i = 0; i < path.size(); ++i)
            {
                const auto &op = path[i];
                std::cout << "Step " << i << ":\n";
                std::cout << "State id " << op->state_id;
                if (op->parent)
                    std::cout << " Parent id " << op->parent->state_id << std::endl;
                else
                    std::cout << " No Parent " << std::endl;
                std::cout << "  Operation Type: " << SLI_Action_to_string(op->action) << "\n";
                std::cout << "  Node1 ID: " << (op->lit1_node ? std::to_string(op->lit1_node->node_id) : "NULL") << "\n";

                if (isNode(op->second_op))
                {
                    auto node = getNode(op->second_op);
                    std::cout << "  Node2 ID: " << (node ? std::to_string(node->node_id) : "NULL") << "\n";
                }
                else if (isLiteral(op->second_op))
                {
                    auto lit = getLiteral(op->second_op);
                    std::cout << "  Literal: " << lit.toString(kb) << "\n";
                }
                // 打印 kb_clause（如果存在）
                if (!op->kb_clause.isEmpty())
                {
                    std::cout << "  KB Clause: " << op->kb_clause.toString(kb) << "\n";
                }

                // Print current tree state
                std::cout << "  Current Tree State:\n";
                op->sli_tree->print_tree(kb);
                std::cout << "----------------------\n";
            }

            std::cout << "====== End of Operation Path ======\n";
        }

        static void printOperationPathAsClause(const std::shared_ptr<OperationState> &state, const KnowledgeBase &kb)
        {
            if (!state)
            {
                std::cout << "No operations to display.\n";
                return;
            }

            // Get operation path from root to current state
            std::vector<std::shared_ptr<OperationState>> path = getOperationPath(state);

            std::cout << "\n====== Operation Path (Clause Form) ======\n";

            for (size_t i = 0; i < path.size(); ++i)
            {
                const auto &op = path[i];
                std::cout << "Step " << i << ":\n";
                std::cout << "Operation Type: " << SLI_Action_to_string(op->action) << "\n";

                std::cout << "Node1: " << (op->lit1_node ? op->lit1_node->literal.toString(kb) : "NULL") << "\n";

                if (isNode(op->second_op))
                {
                    auto node = getNode(op->second_op);
                    std::cout << "Node2: " << (node ? node->literal.toString(kb) : "NULL") << "\n";
                }
                else if (isLiteral(op->second_op))
                {
                    auto lit = getLiteral(op->second_op);
                    std::cout << "Literal: " << lit.toString(kb) << "\n";
                }

                if (!op->kb_clause.isEmpty())
                {
                    std::cout << "KB Clause: " << op->kb_clause.toString(kb) << "\n";
                }

                std::cout << "Current Clause: " << op->sli_tree->printBLiteralsAsClause() << "\n";
                std::cout << "----------------------\n";
            }

            std::cout << "====== End of Operation Path ======\n";
        }

        static void printCurrentState(const std::shared_ptr<OperationState> &state, const KnowledgeBase &kb)
        {
            if (!state)
            {
                std::cout << "Null state.\n";
                return;
            }

            std::cout << "\n=== Current State Details ===\n";
            std::cout << "State ID: " << state->state_id << "\n";
            std::cout << "Parent ID: " << (state->parent ? std::to_string(state->parent->state_id) : "None") << "\n";
            std::cout << "Operation Type: " << SLI_Action_to_string(state->action) << "\n";

            // 打印第一个节点信息
            std::cout << "First Node: ";
            if (state->lit1_node)
            {
                std::cout << "ID=" << state->lit1_node->node_id
                          << ", Literal=" << state->lit1_node->literal.toString(kb) << "\n";
            }
            else
            {
                std::cout << "NULL\n";
            }

            // 打印第二个操作数信息
            std::cout << "Second Operand: ";
            if (isNode(state->second_op))
            {
                auto node = getNode(state->second_op);
                if (node)
                {
                    std::cout << "Node(ID=" << node->node_id
                              << ", Literal=" << node->literal.toString(kb) << ")\n";
                }
                else
                {
                    std::cout << "NULL Node\n";
                }
            }
            else if (isLiteral(state->second_op))
            {
                auto lit = getLiteral(state->second_op);
                std::cout << "Literal(" << lit.toString(kb) << ")\n";
            }

            // 打印知识库子句（如果存在）
            if (!state->kb_clause.isEmpty())
            {
                std::cout << "KB Clause: " << state->kb_clause.toString(kb) << "\n";
            }

            // 打印当前SLI树状态
            std::cout << "\nCurrent Tree State:\n";
            state->sli_tree->print_tree(kb);

            std::cout << "=========================\n";
        }

        static std::shared_ptr<OperationState> deepCopyOperationState(
            const std::shared_ptr<OperationState> &original_state)
        {
            // 深拷贝 SLITree
            auto new_sli_tree = original_state->sli_tree->deepCopy();

            // 在新树中找到对应的新节点
            std::shared_ptr<SLINode> new_lit1_node;
            if (original_state->lit1_node)
            {
                new_lit1_node = new_sli_tree->findNodeById(original_state->lit1_node->node_id);
                if (new_lit1_node == nullptr)
                {
                    throw std::runtime_error(
                        "Failed to find the corresponding node in the new tree. Node ID: " + std::to_string(original_state->lit1_node->node_id));
                }
            }
            else
            {
                new_lit1_node = nullptr;
            }
            // std::cout << "finish copy new_lit1_node \n";

            // 深拷贝第二个操作数
            SecondOperand new_second_op;

            if (isNode(original_state->second_op))
            {
                // 如果是 SLINode，进行深拷贝
                auto original_node = getNode(original_state->second_op);
                if (!original_node)
                {
                    // 处理 second_op 是 null 的情况
                    new_second_op = SecondOperand(std::shared_ptr<SLINode>(nullptr));
                }
                else
                {
                    auto new_node = new_sli_tree->findNodeById(original_node->node_id);
                    if (!new_node)
                    {
                        throw std::runtime_error(
                            "Failed to find the corresponding second node in the new tree. Node ID: " + std::to_string(original_node->node_id));
                    }
                    new_second_op = SecondOperand(new_node);
                }
            }
            else if (isLiteral(original_state->second_op))
            {
                // 如果是 Literal，直接拷贝
                auto original_literal = getLiteral(original_state->second_op);
                new_second_op = SecondOperand(original_literal);
            }
            else
            {
                throw std::runtime_error("second_op holds an unknown type for state_id: " + std::to_string(original_state->state_id));
            }
            // std::cout << "finish copy new_second_op \n";

            // 创建新的 OperationState，保持 parent 指向原始 parent
            auto new_state = std::make_shared<OperationState>(
                new_sli_tree,
                original_state->action,
                new_lit1_node,
                new_second_op,
                original_state->kb_clause,
                original_state->parent); // 这里保持 parent 指向原始 parent

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
        // static std::vector<std::shared_ptr<OperationState>> getOperationPath(
        //     const std::shared_ptr<OperationState> &state)
        // {
        //     std::vector<std::shared_ptr<OperationState>> path;
        //     auto current = state;
        //     while (current)
        //     {
        //         path.push_back(current);
        //         current = current->parent;
        //     }
        //     std::reverse(path.begin(), path.end());
        //     return path;
        // }
    };

} // namespace LogicSystem

#endif // LOGIC_SYSTEM_SLI_OPERATION_H