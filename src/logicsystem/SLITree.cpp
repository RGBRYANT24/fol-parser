// SLITree.cpp
#include "SLITree.h"
#include <iostream>

namespace LogicSystem
{


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

        // 不能在这里检查是不是互补的，一定要在调用之前检查
        /*if(parent->literal.isNegated() == resolving_literal.isNegated())
        {
            throw std::invalid_argument("Not Complementary in add_node");
        }**/

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
                std::cout << "print MGU " << std::endl;
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

                    auto child = std::make_shared<SLINode>(substituted_lit, is_A_literal = false, next_node_id++);
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
                std::cout << "Skipping add literal: " << lit.toString(this->kb) << std::endl;
            }
        }

        /*if (added_nodes.empty())
        {
            std::cout << "No nodes were added to the tree" << std::endl;
            return {};
        }*/

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
        {
            return;
        }

        // Create operation for potential undo
        auto op = std::make_unique<TruncateOperation>();
        bool truncation_performed = false;

        // Process current node and potentially propagate upwards
        std::shared_ptr<SLINode> current = node;
        while (current && current->is_active)
        {
            // Case 1: Node is a leaf and is an A-literal
            if (current->children.empty() && current->is_A_literal)
            {
                dynamic_cast<TruncateOperation *>(op.get())->save_state(current);
                current->is_active = false;
                current->rule_applied = "t-truncate";
                truncation_performed = true;

                // Move to parent for potential further truncation
                current = current->parent.lock();
            }
            // Case 2: Check if all children are inactive
            else if (!current->children.empty())
            {
                bool all_children_inactive = true;
                for (const auto &child : current->children)
                {
                    if (child->is_active)
                    {
                        all_children_inactive = false;
                        break;
                    }
                }

                if (all_children_inactive && current->is_A_literal)
                {
                    dynamic_cast<TruncateOperation *>(op.get())->save_state(current);
                    current->is_active = false;
                    current->rule_applied = "t-truncate";
                    truncation_performed = true;
                    current = current->parent.lock();
                }
                else
                {
                    break; // Stop propagation if conditions aren't met
                }
            }
            else
            {
                break; // Stop propagation if conditions aren't met
            }
        }

        // Only add the operation to stack if any truncation was performed
        if (truncation_performed)
        {
            operation_stack.push(std::move(op));
        }
    }

    bool SLITree::t_factoring(std::shared_ptr<SLINode> upper_node, std::shared_ptr<SLINode> lower_node)
    {
        // 基础检查
        if (!upper_node || !lower_node || !upper_node->is_active || !lower_node->is_active)
        {
            std::cout << "basic check failed in t-factoring " << std::endl;
            return false;
        }

        // 检查正负号是否相同
        if (upper_node->literal.isNegated() != lower_node->literal.isNegated())
        {
            std::cout << "try factoring a negative and positive literal" << std::endl;
            return false;
        }

        // 检查深度是否正确, upper >=lower
        if (upper_node->depth > lower_node->depth)
        {
            std::cout << "deepth wrong in t-factoring" << std::endl;
            std::cout << "upper_node depth " << upper_node->depth << " lower_node_deepth " << lower_node->depth << std::endl;
            return false;
        }

        // 检查是不是直接祖先关系
        if (is_ancestor(upper_node, lower_node))
        {
            std::cout << "upper_node is lower_node's ancestor. t-factoring failed" << std::endl;
            return false;
        }

        // 尝试统一两个节点的文字
        auto mgu = Unifier::findMGU(upper_node->literal, lower_node->literal, kb);
        if (!mgu)
        {
            std::cout << "Find MGU Failed " << std::endl;
            return false;
        }

        try
        {
            std::cout << "try t-factoring " << std::endl;
            // 对upper_node应用替换
            Literal substituted_lit = Unifier::applySubstitutionToLiteral(upper_node->literal, *mgu, kb);

            // 保存原始状态用于可能的回滚
            auto previous_lit = upper_node->literal;
            auto previous_substitution = upper_node->substitution;

            // 更新upper_node
            upper_node->literal = substituted_lit;
            // 合并替换
            for (const auto &[var, term] : *mgu)
            {
                upper_node->substitution[var] = term;
            }

            // 创建截断操作用于处理lower_node
            truncate(lower_node);

            // 创建操作记录
            auto op = std::make_unique<FactoringOperation>(
                upper_node,
                lower_node,
                previous_lit,
                previous_substitution,
                *mgu);
            operation_stack.push(std::move(op));

            upper_node->rule_applied = "t_factoring";
            return true;
        }
        catch (const std::exception &e)
        {
            std::cout << "Error in t_factoring: " << e.what() << std::endl;
            return false;
        }
    }

    bool SLITree::t_ancestry(std::shared_ptr<SLINode> upper_node, std::shared_ptr<SLINode> lower_node)
    {
        // 基础检查
        if (!upper_node || !lower_node || !upper_node->is_active || !lower_node->is_active)
        {
            std::cout << "basic check failed in t-ancestry" << std::endl;
            return false;
        }

        // 检查文字是否互补(一个是否定形式，一个是肯定形式)
        if (upper_node->literal.isNegated() == lower_node->literal.isNegated())
        {
            std::cout << "literals are not complementary" << std::endl;
            return false;
        }

        // 检查是否为祖先关系
        if (!is_ancestor(upper_node, lower_node))
        {
            std::cout << "nodes are not in ancestor relationship" << std::endl;
            return false;
        }

        // 尝试统一两个节点的文字
        auto mgu = Unifier::findMGU(upper_node->literal, lower_node->literal, kb);
        if (!mgu)
        {
            std::cout << "Find MGU Failed in t-ancestry" << std::endl;
            return false;
        }

        try
        {
            std::cout << "try t-ancestry" << std::endl;
            // 对upper_node应用替换
            Literal substituted_lit = Unifier::applySubstitutionToLiteral(upper_node->literal, *mgu, kb);

            // 保存原始状态用于可能的回滚
            auto previous_lit = upper_node->literal;
            auto previous_substitution = upper_node->substitution;

            // 更新upper_node
            upper_node->literal = substituted_lit;

            // 合并替换
            for (const auto &[var, term] : *mgu)
            {
                upper_node->substitution[var] = term;
            }

            // 创建截断操作用于处理lower_node及其子树
            truncate(lower_node);

            // 创建操作记录
            auto op = std::make_unique<AncestryOperation>(
                upper_node,
                lower_node,
                previous_lit,
                previous_substitution,
                *mgu);
            operation_stack.push(std::move(op));

            upper_node->rule_applied = "t_ancestry";
            return true;
        }
        catch (const std::exception &e)
        {
            std::cout << "Error in t_ancestry: " << e.what() << std::endl;
            return false;
        }
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

    std::vector<std::shared_ptr<SLINode>> SLITree::get_active_nodes_at_depth(int depth) const {
    std::vector<std::shared_ptr<SLINode>> active_nodes;
    
    // 检查深度是否有效
    if (depth < 0 || depth >= depth_map.size()) {
        return active_nodes;
    }
    
    // 遍历指定深度的所有节点
    for (const auto& node : depth_map[depth]) {
        // 检查节点是否为活跃的且未参与过消解
        if (node && node->is_active && !node->is_A_literal) {
            active_nodes.push_back(node);
        }
    }
    
    return active_nodes;
}

} // namespace LogicSystem