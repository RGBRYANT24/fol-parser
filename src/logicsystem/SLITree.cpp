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

            // // 从文字映射中移除
            // literal_map.erase(node->literal.hash());
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

        // 获取要参与消解的litreal index 防止rename之后丢失
        int resolbing_literal_index = input_clause.findLiteralIndex(resolving_literal);

        // 对输入子句进行变量重命名
        Clause renamed_clause = VariableRenamer::renameClauseVariables(input_clause, *this, kb);

        // 根据保存的index获取重命名后的resolving literal
        Literal renamed_resolving_literal;
        if (resolbing_literal_index >= 0)
        {
            renamed_resolving_literal = renamed_clause.getLiterals()[resolbing_literal_index];
        }

        std::optional<Substitution> mgu;
        Literal substituted_parent_lit = parent->literal;

        // std::cout << "Original clause: " << input_clause.toString(this->kb) << std::endl;
        // std::cout << "Renamed clause: " << renamed_clause.toString(this->kb) << std::endl;
        // std::cout << "Resolving literal: " << renamed_resolving_literal.toString(this->kb) << std::endl;
        // std::cout << "Parent literal: " << parent->literal.toString(this->kb) << std::endl;
        // std::cout << "Tree before add nodes " <<std::endl;
        // this->print_tree(kb);
        // MGU计算和应用
        if (!resolving_literal.isEmpty())
        {
            mgu = Unifier::findMGU(renamed_resolving_literal, parent->literal, kb);
            if (!mgu)
            {
                std::cout << "MGU unification failed" << std::endl;
                return {};
            }
            // else
            // {
            //     std::cout << "print MGU " << std::endl;
            //     Unifier::printSubstitution(mgu.value(), this->kb);
            // }

            try
            {
                // 保存替换前的状态用于可能的回滚
                std::vector<std::pair<std::shared_ptr<SLINode>, Literal>> previous_literals;
                std::vector<std::pair<std::shared_ptr<SLINode>, Substitution>> previous_substitutions;
                // 对整棵树应用MGU替换
                for (size_t depth = 0; depth < depth_map.size(); ++depth)
                {
                    for (auto &node : depth_map[depth])
                    {
                        if (node && node != root)
                        {
                            // 保存原始状态
                            previous_literals.emplace_back(node, node->literal);
                            previous_substitutions.emplace_back(node, node->substitution);

                            // // 从literal_map中移除旧的hash
                            // literal_map.erase(node->literal.hash());

                            // 应用替换
                            node->literal = Unifier::applySubstitutionToLiteral(node->literal, *mgu, kb);

                            // 更新substitution
                            for (const auto &[var, term] : *mgu)
                            {
                                node->substitution[var] = term;
                            }

                            // // 更新literal_map
                            // literal_map[node->literal.hash()] = node;
                        }
                    }
                }

                // 标记参与消解的节点为A-literal
                parent->is_A_literal = true;
                // parent->is_active = false;

                // std::cout << "Tree after MGU application:" << std::endl;
                // print_tree(kb);
                // 创建操作记录用于可能的回滚 目前暂时不回滚
                // auto op = std::make_unique<SubstitutionOperation>(
                //     previous_literals,
                //     previous_substitutions);
                // operation_stack.push(std::move(op));
            }
            catch (const std::exception &e)
            {
                // std::cout << "Error applying substitution: " << e.what() << std::endl;
                return {};
            }
        }
        else
        {
            mgu = Substitution();
            // std::cout << "Using empty substitution (root node or empty resolving literal)" << std::endl;
        }

        std::vector<std::shared_ptr<SLINode>> added_nodes;

        // 添加节点，但跳过消解文字
        for (const Literal &lit : renamed_clause.getLiterals())
        {
            // 消解文字是经过rename的renamed_resolving_literal
            if (lit != renamed_resolving_literal)
            {
                // std::cout << "Processing literal for addition: " << lit.toString(this->kb) << std::endl;
                try
                {
                    // 应用替换到新的文字
                    Literal substituted_lit = (parent == this->root)
                                                  ? lit
                                                  : Unifier::applySubstitutionToLiteral(lit, *mgu, kb);
                    // if (substituted_lit.getPredicateId() == kb.getPredicateId("E"))
                    // {
                    //     const auto &args = substituted_lit.getArgumentIds();
                    //     if (args.size() == 2 && args[0] == args[1])
                    //     {
                    //         has_self_loop = true; // 设置标志
                    //         return {};
                    //     }
                    // }

                    if (substituted_lit.isEmpty())
                    {
                        std::cout << "Warning: Substitution resulted in empty literal, skipping" << std::endl;
                        continue;
                    }
                    // std::cout << "Creating node with substituted literal: " << substituted_lit.toString(this->kb) << std::endl;

                    auto child = std::make_shared<SLINode>(substituted_lit,
                                                           is_A_literal = false,
                                                           SLINode::next_node_id++); // 使用 SLINode 的静态计数器
                    child->parent = parent;
                    child->depth = parent->depth + 1;
                    child->substitution = *mgu;

                    if (depth_map.size() <= child->depth)
                    {
                        depth_map.resize(child->depth + 1);
                    }

                    depth_map[child->depth].push_back(child);
                    // literal_map[substituted_lit.hash()] = child;
                    parent->children.push_back(child);
                    added_nodes.push_back(child);

                    // std::cout << "Successfully added node at depth " << child->depth << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cout << "Error processing literal: " << e.what() << std::endl;
                    continue;
                }
            }
            else
            {
                // std::cout << "Skipping add literal: " << lit.toString(this->kb) << std::endl;
            }
        }

        /*if (added_nodes.empty())
        {
            std::cout << "No nodes were added to the tree" << std::endl;
            return {};
        }*/

        // try
        // {
        //     auto op = std::make_unique<AddOperation>(added_nodes, literal_map, depth_map);
        //     operation_stack.push(std::move(op));
        //     // std::cout << "Successfully created and stored operation for " << added_nodes.size() << " nodes" << std::endl;
        // }
        // catch (const std::exception &e)
        // {
        //     std::cout << "Error creating operation: " << e.what() << std::endl;
        //     throw;
        // }

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
            // Case 1: Node is a leaf and dont need it is an A-literal
            // if (current->children.empty() && current->is_A_literal)
            if (current->children.empty())
            {
                // std::cout << "truncate node " << current->literal.toString(kb) << std::endl;
                dynamic_cast<TruncateOperation *>(op.get())->save_state(current);
                current->is_active = false;
                current->rule_applied = "t-truncate";
                truncation_performed = true;

                // 从父节点的 children 向量中移除当前节点
                std::shared_ptr<SLINode> parent = current->parent.lock();
                if (parent)
                {
                    auto &siblings = parent->children;
                    siblings.erase(
                        std::remove(siblings.begin(), siblings.end(), current),
                        siblings.end());

                    // 保存父节点引用
                    std::shared_ptr<SLINode> old_parent = parent;

                    // 断开当前节点与父节点的关系
                    current->parent.reset();

                    // 将 current 指向父节点，继续向上截断
                    current = old_parent;
                }
                else
                {
                    // 没有父节点，停止截断
                    current = nullptr;
                }
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

                    // 从父节点的 children 向量中移除当前节点
                    std::shared_ptr<SLINode> parent = current->parent.lock();
                    if (parent)
                    {
                        auto &siblings = parent->children;
                        siblings.erase(
                            std::remove(siblings.begin(), siblings.end(), current),
                            siblings.end());

                        // 保存父节点引用
                        std::shared_ptr<SLINode> old_parent = parent;

                        // 断开当前节点与父节点的关系
                        current->parent.reset();

                        // 将 current 指向父节点，继续向上截断
                        current = old_parent;
                    }
                    else
                    {
                        // 没有父节点，停止截断
                        current = nullptr;
                    }
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
            cleanup_empty_depths(); // 清理空层
        }
    }

    bool SLITree::t_factoring(std::shared_ptr<SLINode> upper_node, std::shared_ptr<SLINode> lower_node)
    {
        // 基础检查
        // is_active表示内存还存在的节点 is_A_literal表示是否是A-lit
        if (!upper_node || !lower_node || !upper_node->is_active || !lower_node->is_active || upper_node->is_A_literal || lower_node->is_A_literal)
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

        // 检查深度是否正确, upper > lower
        if (upper_node->depth > lower_node->depth)
        {
            std::cout << "deepth wrong in t-factoring" << std::endl;
            std::cout << "upper_node depth " << upper_node->depth << " lower_node_deepth " << lower_node->depth << std::endl;
            std::cout << "upper node " << upper_node->node_id << " lower node " << lower_node->node_id << std::endl;
            this->print_tree(kb);
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
            // 保存整棵树当前状态用于可能的回滚
            std::vector<std::pair<std::shared_ptr<SLINode>, Literal>> previous_literals;
            std::vector<std::pair<std::shared_ptr<SLINode>, Substitution>> previous_substitutions;
            
            std::cout << "before factoring in SLITree::t_factoring " << std::endl;
            this->print_tree(kb);

            // 对整棵树应用MGU替换
            for (auto &level : depth_map)
            {
                for (auto &node : level)
                {
                    if (node && node != this->root)
                    {
                        // 保存原始状态
                        previous_literals.emplace_back(node, node->literal);
                        previous_substitutions.emplace_back(node, node->substitution);

                        // 应用替换到节点的文字
                        node->literal = Unifier::applySubstitutionToLiteral(node->literal, *mgu, kb);
                        // 合并替换
                        for (const auto &[var, term] : *mgu)
                        {
                            node->substitution[var] = term;
                        }
                    }
                }
            }

            // 创建截断操作用于处理lower_node
            truncate(lower_node);

            // 创建操作记录（需要修改FactoringOperation类以支持全局替换）
            auto op = std::make_unique<FactoringOperation>(
                upper_node,
                lower_node,
                previous_literals,
                previous_substitutions,
                *mgu);
            operation_stack.push(std::move(op));

            upper_node->rule_applied = "t_factoring";
            std::cout << "after factoring in SLITree::t_factoring " << std::endl;
            this->print_tree(kb);
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
        if (lower_node->parent.lock() == upper_node || lower_node->depth - upper_node->depth == 1)
        {
            std::cout << "basic check in t_ancestry failed, cannot ancestry with direct parent" << std::endl;
            return false;
        }

        // 基础检查
        // is_active表示内存还存在的节点 is_A_literal表示是否是A-lit
        if (!upper_node || !lower_node || !upper_node->is_active || !lower_node->is_active || !upper_node->is_A_literal || lower_node->is_A_literal)
        {
            std::cout << "basic check failed in t-ancestry" << std::endl;
            if(!upper_node->is_A_literal)
            {
                std::cout << "upper node is not A lit " << std::endl;
            }
            return false;
        }

        // 检查文字是否互补
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
            // 保存整棵树当前状态用于可能的回滚
            std::vector<std::pair<std::shared_ptr<SLINode>, Literal>> previous_literals;
            std::vector<std::pair<std::shared_ptr<SLINode>, Substitution>> previous_substitutions;

            // 对整棵树应用MGU替换
            for (auto &level : depth_map)
            {
                for (auto &node : level)
                {
                    if (node && node != this->root)
                    {
                        // 保存原始状态
                        previous_literals.emplace_back(node, node->literal);
                        previous_substitutions.emplace_back(node, node->substitution);

                        // 应用替换到节点的文字
                        node->literal = Unifier::applySubstitutionToLiteral(node->literal, *mgu, kb);
                        // 合并替换
                        for (const auto &[var, term] : *mgu)
                        {
                            node->substitution[var] = term;
                        }
                    }
                }
            }

            // 创建截断操作用于处理lower_node及其子树
            truncate(lower_node);

            // 创建操作记录（需要修改AncestryOperation类以支持全局替换）
            auto op = std::make_unique<AncestryOperation>(
                upper_node,
                lower_node,
                previous_literals,
                previous_substitutions,
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

    // 获取γL集合
    std::vector<std::shared_ptr<SLINode>> SLITree::get_gamma_L(std::shared_ptr<SLINode> L_node) const
    {
        std::vector<std::shared_ptr<SLINode>> gamma_L;

        // 从L节点向上遍历到根节点
        auto current = L_node;
        while (auto parent = current->parent.lock())
        {
            // 对路径上的每个节点,收集其B-literal子节点
            for (auto &sibling : parent->children)
            {
                // 排除L节点本身
                // 添加is_active 限制
                if (!sibling->is_A_literal && sibling != L_node && sibling->is_active)
                {
                    gamma_L.push_back(sibling);
                }
            }
            current = parent;
        }
        return gamma_L;
    }

    // 获取δL集合
    std::vector<std::shared_ptr<SLINode>> SLITree::get_delta_L(std::shared_ptr<SLINode> L_node) const
    {
        std::vector<std::shared_ptr<SLINode>> delta_L;

        // 从L节点向上遍历到根节点
        auto current = L_node;
        while (auto parent = current->parent.lock())
        {
            // 收集路径上的所有A-literal
            if (parent->is_A_literal)
            {
                delta_L.push_back(parent);
            }
            current = parent;
        }
        return delta_L;
    }

    bool SLITree::check_AC(std::shared_ptr<SLINode> L_node) const
    {

        // 获取γL和δL集合
        auto gamma_L = get_gamma_L(L_node);
        auto delta_L = get_delta_L(L_node);

        // 检查条件(i): γL ∪ {L}中不能有相同atom的literals
        // 这一步只检查B-lit
        if (!L_node->is_A_literal)
        {
            gamma_L.push_back(L_node);
            for (size_t i = 0; i < gamma_L.size(); i++)
            {
                for (size_t j = i + 1; j < gamma_L.size(); j++)
                {
                    if (have_same_atom(gamma_L[i], gamma_L[j]))
                    {
                        // std::cout << "check_AC CONDITION 1 FAILED NODES: " << std::endl;
                        // std::cout << "higher node in gamma_L " << gamma_L[i]->literal.toString(kb) << " lower node " << gamma_L[j]->literal.toString(kb) << std::endl;
                        return false;
                    }
                }
            }
        }

        // 检查条件(ii): δL ∪ {L}中不能有相同atom的literals
        delta_L.push_back(L_node);
        for (size_t i = 0; i < delta_L.size(); i++)
        {
            for (size_t j = i + 1; j < delta_L.size(); j++)
            {
                if (have_same_atom(delta_L[i], delta_L[j]))
                {
                    // std::cout << "check_AC CONDITION 2 FAILED NODES: " << std::endl;
                    // std::cout << "higher node in delta_L " << delta_L[i]->literal.toString(kb) << " lower node " << delta_L[j]->literal.toString(kb) << std::endl;
                    return false;
                }
            }
        }

        return true;
    }

    bool SLITree::check_MC(const std::shared_ptr<SLINode> &node) const
    {
        // 如果是叶子节点且是A-literal，则违反MC条件
        if (node->children.empty() && node->is_A_literal)
        {
            // std::cout << "check_MC FAILED NODE: " << std::endl;
            // this->print_node_info(node, kb, "", true);
            return false;
        }

        // 递归检查所有子节点
        for (const auto &child : node->children)
        {
            if (!check_MC(child))
            {
                return false;
            }
        }

        return true;
    }

    bool SLITree::check_all_nodes_AC() const
    {
        // Get all active nodes
        auto all_nodes = get_all_active_nodes();

        // Check AC condition for each node
        for (const auto &node : all_nodes)
        {
            if (!check_AC(node))
            {
                return false; // If any node fails AC check, return false
            }
        }

        return true; // All nodes passed AC check
    }

    bool SLITree::check_all_nodes_MC() const
    {
        // Get all active nodes
        auto all_nodes = get_all_active_nodes();

        // Check MC condition for each node
        for (const auto &node : all_nodes)
        {
            if (!check_MC(node))
            {
                return false; // If any node fails MC check, return false
            }
        }

        return true; // All nodes passed MC check
    }

    // 辅助函数：检查两个literal是否有相同的atom
    bool SLITree::have_same_atom(const std::shared_ptr<SLINode> &node1, const std::shared_ptr<SLINode> &node2) const
    {
        Literal lit1 = node1->literal;
        Literal lit2 = node2->literal;
        // 如果谓词ID相同且参数相同（忽略否定符号），则认为是相同的atom
        return (lit1.getPredicateId() == lit2.getPredicateId() &&
                lit1.getArgumentIds() == lit2.getArgumentIds());
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

    std::vector<std::shared_ptr<SLINode>> SLITree::get_active_nodes_at_depth(int depth) const
    {
        std::vector<std::shared_ptr<SLINode>> active_nodes;

        // 检查深度是否有效
        if (depth < 0 || depth >= depth_map.size())
        {
            return active_nodes;
        }

        // 遍历指定深度的所有节点
        for (const auto &node : depth_map[depth])
        {
            // 检查节点是否为活跃的且未参与过消解
            if (node && node->is_active && !node->is_A_literal)
            {
                active_nodes.push_back(node);
            }
        }

        return active_nodes;
    }

    void SLITree::cleanup_empty_depths()
    {
        // 从后向前遍历，找到第一个非空层
        int last_non_empty = -1;
        for (int i = static_cast<int>(depth_map.size()) - 1; i >= 0; --i)
        {
            bool has_active_nodes = false;
            for (const auto &node : depth_map[i])
            {
                if (node && node->is_active)
                {
                    has_active_nodes = true;
                    break;
                }
            }
            if (has_active_nodes)
            {
                last_non_empty = i;
                break;
            }
        }

        // 如果找到了最后一个非空层，将depth_map调整为该大小
        // 保留根节点所在的层（深度0）
        if (last_non_empty >= 0)
        {
            depth_map.resize(last_non_empty + 1);
        }
        else
        {
            // 如果所有层都是空的，至少保留深度0
            depth_map.resize(1);
        }
    }

    // SLITree.cpp  中实现拷贝构造函数
    SLITree::SLITree(const SLITree &other, std::shared_ptr<SLINode> startNode)
        : kb(other.kb)
    {
        // 创建节点映射表，使用 node_id 作为键
        std::unordered_map<int, std::shared_ptr<SLINode>> nodeMap;

        // 复制根节点
        root = std::make_shared<SLINode>(other.root->literal,
                                         other.root->is_A_literal,
                                         other.root->node_id);
        root->depth = 0;
        root->is_active = other.root->is_active;
        nodeMap[other.root->node_id] = root;

        // 按层复制整棵树
        for (size_t depth = 0; depth < other.depth_map.size(); ++depth)
        {
            for (const auto &oldNode : other.depth_map[depth])
            {
                if (oldNode == other.root)
                    continue; // 根节点已经复制过了

                // 创建新节点
                auto newNode = std::make_shared<SLINode>(oldNode->literal,
                                                         oldNode->is_A_literal,
                                                         oldNode->node_id);
                newNode->depth = oldNode->depth;
                newNode->is_active = oldNode->is_active;
                newNode->substitution = oldNode->substitution;
                newNode->rule_applied = oldNode->rule_applied;
                newNode->is_A_literal = oldNode->is_A_literal;

                // 设置父节点
                if (auto oldParent = oldNode->parent.lock())
                {
                    auto it = nodeMap.find(oldParent->node_id);
                    if (it != nodeMap.end())
                    {
                        newNode->parent = it->second;
                        it->second->children.push_back(newNode);
                    }
                    else
                    {
                        std::cerr << "Parent node_id " << oldParent->node_id << " not found in nodeMap.\n";
                    }
                }

                // 将新节点加入映射表
                nodeMap[oldNode->node_id] = newNode;
            }
        }

        // 复制深度图和文字映射
        depth_map = other.depth_map;
        for (auto &level : depth_map)
        {
            for (auto &node : level)
            {
                if (nodeMap.find(node->node_id) != nodeMap.end())
                {
                    node = nodeMap[node->node_id];
                }
                else
                {
                    std::cerr << "Node with node_id " << node->node_id << " not found in nodeMap.\n";
                    node = nullptr; // 或者其他错误处理
                }
            }
        }

        // // 复制文字映射
        // for (const auto &[hash, oldNode] : other.literal_map)
        // {
        //     if (nodeMap.find(oldNode->node_id) != nodeMap.end())
        //     {
        //         literal_map[hash] = nodeMap[oldNode->node_id];
        //     }
        //     else
        //     {
        //         std::cerr << "Literal map node_id " << oldNode->node_id << " not found in nodeMap.\n";
        //     }
        // }
    }

    std::shared_ptr<SLINode> SLITree::copyNode(const std::shared_ptr<SLINode> &node)
    {
        // 复制时保持原始nodeId
        auto new_node = std::make_shared<SLINode>(node->literal,
                                                  node->is_A_literal,
                                                  node->node_id); // 使用原始节点ID
        new_node->depth = node->depth;
        new_node->is_active = node->is_active;
        new_node->substitution = node->substitution;
        new_node->rule_applied = node->rule_applied;
        return new_node;
    }

    // 在SLITree.cpp中实现
    size_t SLITree::computeNodeHash(const std::shared_ptr<SLINode> &node) const
    {
        if (!node)
            return 0;

        size_t hash = 0;

        // 合并文字的哈希值
        hash ^= node->literal.hash();

        // 合并节点属性的哈希值
        hash ^= std::hash<bool>{}(node->is_A_literal);
        hash ^= std::hash<bool>{}(node->is_active);
        hash ^= std::hash<int>{}(node->depth);

        // 合并替换的哈希值
        for (const auto &[var, term] : node->substitution)
        {
            hash ^= std::hash<SymbolId>{}(var);
            hash ^= std::hash<SymbolId>{}(term);
        }

        return hash;
    }

    std::vector<std::shared_ptr<SLINode>> SLITree::get_all_active_nodes() const
    {
        std::vector<std::shared_ptr<SLINode>> active_nodes;

        // 遍历所有深度层级
        for (const auto &level : depth_map)
        {
            // 遍历当前深度的所有节点
            for (const auto &node : level)
            {
                // 如果节点存在且是active的，加入结果列表
                if (node && node->is_active)
                {
                    active_nodes.push_back(node);
                }
            }
        }

        return active_nodes;
    }

    std::vector<std::shared_ptr<SLINode>> SLITree::get_all_B_literals()
    {
        std::vector<std::shared_ptr<SLINode>> B_literals;

        // 遍历depth_map中的所有层
        for (const auto &level : depth_map)
        {
            // 遍历每一层中的所有节点
            for (const auto &node : level)
            {
                // 检查节点是否有效且不是A-literal
                if (node && node->is_active && !node->is_A_literal && !node->literal.isEmpty())
                {
                    B_literals.push_back(node);
                }
            }
        }
        return B_literals;
    }

    size_t SLITree::computeStateHash() const
    {
        size_t hash = 0;
        const size_t PRIME = 31;

        // std::cout << "\nStarting hash computation..." << std::endl;

        for (const auto &level : depth_map)
        {
            size_t level_hash = 1;
            int current_depth = level.front()->depth;

            // std::cout << "\nProcessing Level " << current_depth << ":" << std::endl;

            for (const auto &node : level)
            {
                if (node && node->is_active)
                {
                    // std::cout << "  Processing node: " << node->literal.toString(kb) << std::endl;

                    size_t node_hash = computeNodeHash(node);
                    // std::cout << "    Initial node_hash: " << node_hash << std::endl;

                    if (auto parent = node->parent.lock())
                    {
                        size_t parent_literal_hash = parent->literal.hash();
                        size_t parent_depth_hash = std::hash<int>{}(parent->depth);
                        size_t parent_info = parent_literal_hash + parent_depth_hash;

                        // std::cout << "    Parent info:" << std::endl;
                        // std::cout << "      Parent predicate: " << parent->literal.toString(kb) << std::endl;
                        // std::cout << "      Parent literal hash: " << parent_literal_hash << std::endl;
                        // std::cout << "      Parent depth hash: " << parent_depth_hash << std::endl;
                        // std::cout << "      Combined parent_info: " << parent_info << std::endl;

                        node_hash = node_hash * PRIME + parent_info;
                        // std::cout << "    Node hash after parent info: " << node_hash << std::endl;
                    }

                    size_t prev_level_hash = level_hash;
                    level_hash = level_hash * PRIME + node_hash;
                    // std::cout << "    Level hash update: " << prev_level_hash
                    //          << " -> " << level_hash << std::endl;
                }
            }

            size_t depth_hash = std::hash<int>{}(current_depth);
            size_t prev_hash = hash;
            hash = hash * PRIME + (level_hash + depth_hash);

            // std::cout << "  Level " << current_depth << " final computation:" << std::endl;
            // std::cout << "    Level hash: " << level_hash << std::endl;
            // std::cout << "    Depth hash: " << depth_hash << std::endl;
            // std::cout << "    Previous total hash: " << prev_hash << std::endl;
            // std::cout << "    New total hash: " << hash << std::endl;
        }

        // std::cout << "\nFinal hash value: " << hash << std::endl;
        return hash;
    }

    bool SLITree::areNodesEquivalent(const std::shared_ptr<SLINode> &node1,
                                     const std::shared_ptr<SLINode> &node2) const
    {
        if (!node1 || !node2)
            return node1 == node2;

        // 检查基本属性
        if (node1->literal != node2->literal ||
            node1->is_A_literal != node2->is_A_literal ||
            node1->is_active != node2->is_active ||
            node1->depth != node2->depth ||
            node1->substitution != node2->substitution)
        {
            return false;
        }

        return true;
    }

    bool SLITree::isEquivalentTo(const SLITree &other) const
    {
        if (depth_map.size() != other.depth_map.size())
            return false;

        // 按层比较活跃节点
        for (size_t i = 0; i < depth_map.size(); ++i)
        {
            std::vector<std::shared_ptr<SLINode>> active1, active2;

            // 收集当前层的活跃节点
            for (const auto &node : depth_map[i])
            {
                if (node && node->is_active)
                    active1.push_back(node);
            }
            for (const auto &node : other.depth_map[i])
            {
                if (node && node->is_active)
                    active2.push_back(node);
            }

            if (active1.size() != active2.size())
                return false;

            // 比较节点属性和结构
            for (size_t j = 0; j < active1.size(); ++j)
            {
                if (!areNodesEquivalent(active1[j], active2[j]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    std::string SLITree::printBLiteralsAsClause() const
    {
        // 获取所有B-literals
        std::vector<Literal> b_literals;
        for (const auto &level : depth_map)
        {
            for (const auto &node : level)
            {
                if (node && node->is_active && !node->is_A_literal)
                {
                    b_literals.push_back(node->literal);
                }
            }
        }

        // 按照子句格式构建输出字符串
        std::string result;
        for (size_t i = 0; i < b_literals.size(); ++i)
        {
            result += b_literals[i].toString(kb);
            if (i < b_literals.size() - 1)
            {
                result += " ∨ ";
            }
        }

        return result;
    }

} // namespace LogicSystem