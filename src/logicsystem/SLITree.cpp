// SLITree.cpp
#include "SLITree.h"
#include <iostream>

namespace LogicSystem
{

    void TruncateOperation::undo()
    {
        for (size_t i = 0; i < affected_nodes.size(); ++i)
        {
            affected_nodes[i]->is_active = previous_states[i];
        }
    }

    // SLITree.cpp
    void AddOperation::undo()
    {
        if (node)
        {
            // 从 literal_map_ref 中移除节点
            size_t node_hash = node->literal.hash();
            literal_map_ref.erase(node_hash);

            // 从 depth_map_ref 中移除节点
            if (node->depth < depth_map_ref.size())
            {
                auto &nodes_at_depth = depth_map_ref[node->depth];
                nodes_at_depth.erase(
                    std::remove(nodes_at_depth.begin(), nodes_at_depth.end(), node),
                    nodes_at_depth.end());
            }
        }
    }

    std::shared_ptr<SLINode> SLITree::add_node(const Literal &literal, bool is_A_literal,
                                               std::shared_ptr<SLINode> parent)
    {
        auto node = std::make_shared<SLINode>(literal, is_A_literal, next_node_id++);
        node->rule_applied = "extension"; // node t-extension

        if (parent)
        {
            node->parent = parent;
            node->depth = parent->depth + 1;
            parent->children.push_back(node);
        }
        else
        {
            root = node;
            node->depth = 0;
        }

        if (depth_map.size() <= node->depth)
        {
            depth_map.resize(node->depth + 1);
        }
        depth_map[node->depth].push_back(node);

        literal_map[literal.hash()] = node;

        // 创建并保存Operation
        auto op = std::make_unique<AddOperation>(node, literal_map, depth_map);
        operation_stack.push(std::move(op));

        return node;
    }

    bool SLITree::is_ancestor(std::shared_ptr<SLINode> potential_ancestor,
                              std::shared_ptr<SLINode> potential_descendant)
    {
        if (!potential_ancestor || !potential_descendant)
            return false;
        if (potential_ancestor->depth >= potential_descendant->depth)
            return false;

        auto current = potential_descendant;
        while (current && current->depth > potential_ancestor->depth)
        {
            if (auto parent = current->parent.lock())
            {
                current = parent;
            }
            else
            {
                return false;
            }
        }
        return current == potential_ancestor;
    }

    void SLITree::truncate(std::shared_ptr<SLINode> node)
    {
        if (!node || !node->is_active)
            return;

        auto op = std::make_unique<TruncateOperation>(node);
        std::stack<std::shared_ptr<SLINode>> stack;
        stack.push(node);

        while (!stack.empty())
        {
            auto current = stack.top();
            stack.pop();

            if (current->is_active)
            {
                dynamic_cast<TruncateOperation *>(op.get())->save_state(current);
                current->is_active = false;
                current->rule_applied = "truncate"; // 添加规则信息

                for (const auto &child : current->children)
                {
                    stack.push(child);
                }
            }
        }

        operation_stack.push(std::move(op));
    }

    bool SLITree::t_factoring(std::shared_ptr<SLINode> node1, std::shared_ptr<SLINode> node2)
    {
        if (!node1 || !node2 || !node1->is_active || !node2->is_active)
        {
            return false;
        }

        // 尝试统一两个节点的文字
        auto unified_literal = try_unify(node1->literal, node2->literal);
        if (!unified_literal)
        {
            return false;
        }

        // 如果统一成功，创建新的节点
        auto new_node = add_node(*unified_literal, true, node1);
        new_node->rule_applied = "t_factoring"; // 添加规则信息

        // 将新节点的substitution存储起来
        auto mgu = Unifier::findMGU(node1->literal, node2->literal, kb);
        if (mgu)
        {
            new_node->substitution = *mgu;
        }

        return true;
    }

    bool SLITree::t_ancestry(std::shared_ptr<SLINode> node1, std::shared_ptr<SLINode> node2)
    {
        if (!node1 || !node2 || !node1->is_active || !node2->is_active)
        {
            return false;
        }

        // 检查祖先关系
        if (!is_ancestor(node1, node2))
        {
            return false;
        }

        // 尝试统一文字
        auto unified_literal = try_unify(node1->literal, node2->literal);
        if (!unified_literal)
        {
            return false;
        }

        // 如果统一成功，创建新的节点
        auto new_node = add_node(*unified_literal, true, node2);
        new_node->rule_applied = "t_ancestry"; // 添加规则信息

        // 存储substitution
        auto mgu = Unifier::findMGU(node1->literal, node2->literal, kb);
        if (mgu)
        {
            new_node->substitution = *mgu;
        }

        return true;
    }

    void SLITree::rollback()
    {
        if (!operation_stack.empty())
        {
            operation_stack.top()->undo();
            operation_stack.pop();
        }
    }
    void SLITree::print_tree(const KnowledgeBase &kb) const
    {
        if (!root)
        {
            std::cout << "Empty tree\n";
            return;
        }
        print_node_info(root, kb, "", true);
        for (size_t i = 0; i < root->children.size(); ++i)
        {
            print_node(root->children[i], kb, "", i == root->children.size() - 1);
        }
    }

    void SLITree::print_node(std::shared_ptr<SLINode> node, const KnowledgeBase &kb,
                             std::string prefix, bool is_last) const
    {
        if (!node || !node->is_active)
        {
            return;
        }

        print_node_info(node, kb, prefix, is_last);

        prefix += is_last ? "  " : "| ";
        for (size_t i = 0; i < node->children.size(); ++i)
        {
            print_node(node->children[i], kb, prefix, i == node->children.size() - 1);
        }
    }

    void SLITree::print_node_info(std::shared_ptr<SLINode> node, const KnowledgeBase &kb,
                                  std::string prefix, bool is_last) const
    {
        // 第一行：基本树结构和文字信息
        std::cout << prefix << (prefix.empty() ? "" : get_branch_str(is_last))
                  << node->literal.toString(kb)
                  << (node->is_A_literal ? "*" : "");

        // 节点基本信息
        std::cout << " [" << node->node_id << "|d:" << node->depth << "]";

        // 替换信息
        if (!node->substitution.empty())
        {
            std::cout << " subst:{";
            bool first = true;
            for (const auto &[var, term] : node->substitution)
            {
                if (!first)
                    std::cout << ",";
                std::cout << kb.getSymbolName(var) << "/" << kb.getSymbolName(term);
                first = false;
            }
            std::cout << "}";
        }

        // 状态信息
        std::cout << " (";
        std::vector<std::string> status;
        if (!node->is_active)
            status.push_back("inactive");
        // if (node->is_closed)
        //     status.push_back("closed");
        // if (node->is_blocked)
        //     status.push_back("blocked");
        if (node->is_A_literal)
            status.push_back("A-lit");
        if (status.empty())
            status.push_back("active");
        std::cout << join(status, ",") << ")";

        // 其他调试信息
        if (node->parent.lock())
        {
            std::cout << " parent:" << node->parent.lock()->node_id;
        }
        std::cout << " children:" << node->children.size();

        // 可选：显示使用的规则或生成原因
        if (!node->rule_applied.empty())
        {
            std::cout << " rule:" << node->rule_applied;
        }

        std::cout << "\n";
    }

} // namespace LogicSystem