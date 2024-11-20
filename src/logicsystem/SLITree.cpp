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
        for (const auto &node : added_nodes)
        {
            // 从父节点的children中移除
            if (auto parent = node->parent.lock())
            {
                auto &children = parent->children;
                children.erase(
                    std::remove(children.begin(), children.end(), node),
                    children.end());
            }

            // 从深度图中移除
            if (node->depth < depth_map.size())
            {
                auto &depth_level = depth_map[node->depth];
                depth_level.erase(
                    std::remove(depth_level.begin(), depth_level.end(), node),
                    depth_level.end());
            }

            // 从文字映射中移除
            literal_map.erase(node->literal.hash());
        }
    }

    std::vector<std::shared_ptr<SLINode>> SLITree::add_node(const Clause &input_clause, const Literal &resolving_literal,
                                                            bool is_A_literal, std::shared_ptr<SLINode> parent)
    {
        // 基础检查
        if (!parent)
        {
            throw std::invalid_argument("Parent node must be specified");
        }

        if (input_clause.getLiterals().empty())
        {
            std::cout << "Warning: Empty input clause" << std::endl;
            return {};
        }

        std::optional<Substitution> mgu;
        Literal substituted_parent_lit = parent->literal;

        std::cout << "Processing clause: " << input_clause.toString(this->kb) << std::endl;
        std::cout << "Resolving literal: " << resolving_literal.toString(this->kb) << std::endl;
        std::cout << "Parent literal: " << parent->literal.toString(this->kb) << std::endl;

        // MGU计算和应用
        if (parent != this->root && !resolving_literal.isEmpty())
        {
            mgu = Unifier::findMGU(resolving_literal, parent->literal, kb);
            if (!mgu)
            {
                std::cout << "MGU unification failed" << std::endl;
                return {};
            }
            else
            {
                std::cout << "print MGU "<< std::endl;
                Unifier::printSubstitution(mgu.value(), this->kb);
            }

            try
            {
                // 对父节点应用替换
                substituted_parent_lit = Unifier::applySubstitutionToLiteral(parent->literal, *mgu, kb);

                // 更新父节点的字面量
                parent->literal = substituted_parent_lit;
                // 标记父节点为A-literal（因为它参与了消解）
                parent->is_A_literal = true;

                std::cout << "Parent literal after MGU: " << substituted_parent_lit.toString(this->kb) << std::endl;

                // 对父节点的所有祖先节点也应用相同的替换
                /*auto current = parent;
                while (current->parent && current->parent != root)
                {
                    current->parent->literal = Unifier::applySubstitutionToLiteral(current->parent->literal, *mgu, kb);
                    current = current->parent;
                }*/
            }
            catch (const std::exception &e)
            {
                std::cout << "Error applying substitution to parent: " << e.what() << std::endl;
                return {};
            }
        }
        else
        {
            mgu = Substitution();
            std::cout << "Using empty substitution (root node or empty resolving literal)" << std::endl;
        }

        std::vector<std::shared_ptr<SLINode>> added_nodes;

        // 添加节点，但跳过消解文字
        for (const Literal &lit : input_clause.getLiterals())
        {
            if (lit != resolving_literal)
            {
                std::cout << "Processing literal for addition: " << lit.toString(this->kb) << std::endl;

                try
                {
                    // 应用替换到新的文字
                    Literal substituted_lit = (parent == this->root)
                                                  ? lit
                                                  : Unifier::applySubstitutionToLiteral(lit, *mgu, kb);

                    if (substituted_lit.isEmpty())
                    {
                        std::cout << "Warning: Substitution resulted in empty literal, skipping" << std::endl;
                        continue;
                    }

                    std::cout << "Creating node with substituted literal: " << substituted_lit.toString(this->kb) << std::endl;

                    auto child = std::make_shared<SLINode>(substituted_lit, is_A_literal=false, next_node_id++);
                    child->parent = parent;
                    child->depth = parent->depth + 1;
                    child->substitution = *mgu;

                    if (depth_map.size() <= child->depth)
                    {
                        depth_map.resize(child->depth + 1);
                    }

                    depth_map[child->depth].push_back(child);
                    literal_map[substituted_lit.hash()] = child;
                    parent->children.push_back(child);
                    added_nodes.push_back(child);

                    std::cout << "Successfully added node at depth " << child->depth << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cout << "Error processing literal: " << e.what() << std::endl;
                    continue;
                }
            }
            else
            {
                std::cout << "Skipping resolving literal: " << lit.toString(this->kb) << std::endl;
            }
        }

        if (added_nodes.empty())
        {
            std::cout << "No nodes were added to the tree" << std::endl;
            return {};
        }

        try
        {
            auto op = std::make_unique<AddOperation>(added_nodes, literal_map, depth_map);
            operation_stack.push(std::move(op));
            std::cout << "Successfully created and stored operation for " << added_nodes.size() << " nodes" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cout << "Error creating operation: " << e.what() << std::endl;
            throw;
        }

        return added_nodes;
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

    /*bool SLITree::t_factoring(std::shared_ptr<SLINode> node1, std::shared_ptr<SLINode> node2)
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
    }*/
    void SLITree::print_tree(const KnowledgeBase &kb) const
    {
        if (!root)
        {
            std::cout << "Empty tree\n";
            return;
        }

        // 按深度遍历所有活跃节点
        for (size_t depth = 0; depth < depth_map.size(); ++depth)
        {
            std::cout << "Depth " << depth << ":\n";
            for (const auto &node : depth_map[depth])
            {
                if (node && node->is_active)
                {
                    std::string prefix = "  ";
                    print_node_info(node, kb, prefix, false);

                    // 可选：显示与父节点的关系
                    if (auto parent = node->parent.lock())
                    {
                        std::cout << prefix << "  └─ Parent: " << parent->node_id << "\n";
                    }
                }
            }
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
        if (node->literal.isEmpty())
        {
            std::cout << prefix << get_branch_str(is_last) << "ROOT" << std::endl;
            return;
        }
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