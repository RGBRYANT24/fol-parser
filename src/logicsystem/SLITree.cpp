// SLITree.cpp
#include "SLITree.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>

// 以下与 Operation 相关的代码已删除/注释掉
/*
void AddOperation::undo()
{
    for (const auto &node : added_nodes)
    {
        if (auto parent = node->parent.lock())
        {
            auto &children = parent->children;
            children.erase(
                std::remove(children.begin(), children.end(), node),
                children.end());
        }

        if (node->depth < depth_map.size())
        {
            auto &depth_level = depth_map[node->depth];
            depth_level.erase(
                std::remove(depth_level.begin(), depth_level.end(), node),
                depth_level.end());
        }
    }
}
*/

namespace LogicSystem
{

    std::vector<std::shared_ptr<SLINode>> SLITree::add_node(const Clause &input_clause, const Literal &resolving_literal,
                                                            bool is_A_literal, std::shared_ptr<SLINode> parent)
    {
        if (!parent)
        {
            throw std::invalid_argument("Parent node must be specified");
        }
        parent = this->findNodeById(parent->node_id);
        if (!parent)
        {
            throw std::runtime_error("父节点在当前SLITree状态中未找到！");
        }

        if (input_clause.getLiterals().empty())
        {
            std::cout << "Warning: Empty input clause" << std::endl;
            return {};
        }

        int resolbing_literal_index = input_clause.findLiteralIndex(resolving_literal);
        Clause renamed_clause = VariableRenamer::renameClauseVariables(input_clause, *this, kb);

        Literal renamed_resolving_literal;
        if (resolbing_literal_index >= 0)
        {
            renamed_resolving_literal = renamed_clause.getLiterals()[resolbing_literal_index];
        }

        std::optional<Substitution> mgu;
        Literal substituted_parent_lit = parent->literal;

        if (!resolving_literal.isEmpty())
        {
            mgu = Unifier::findMGU(renamed_resolving_literal, parent->literal, kb);
            if (!mgu)
            {
                std::cout << "MGU unification failed" << std::endl;
                return {};
            }

            try
            {
                std::vector<std::pair<std::shared_ptr<SLINode>, Literal>> previous_literals;
                std::vector<std::pair<std::shared_ptr<SLINode>, Substitution>> previous_substitutions;
                for (size_t depth = 0; depth < depth_map.size(); ++depth)
                {
                    for (auto &node : depth_map[depth])
                    {
                        if (node && node != root)
                        {
                            previous_literals.emplace_back(node, node->literal);
                            previous_substitutions.emplace_back(node, node->substitution);

                            node->literal = Unifier::applySubstitutionToLiteral(node->literal, *mgu, kb);
                            for (const auto &[var, term] : *mgu)
                            {
                                node->substitution[var] = term;
                            }
                        }
                    }
                }
                parent->is_A_literal = true;
            }
            catch (const std::exception &e)
            {
                return {};
            }
        }
        else
        {
            mgu = Substitution();
        }

        std::vector<std::shared_ptr<SLINode>> added_nodes;
        for (const Literal &lit : renamed_clause.getLiterals())
        {
            if (lit != renamed_resolving_literal)
            {
                try
                {
                    Literal substituted_lit = (parent == this->root)
                                                    ? lit
                                                    : Unifier::applySubstitutionToLiteral(lit, *mgu, kb);
                    if (substituted_lit.isEmpty())
                    {
                        std::cout << "Warning: Substitution resulted in empty literal, skipping" << std::endl;
                        continue;
                    }
                    auto child = std::make_shared<SLINode>(substituted_lit,
                                                           false,
                                                           SLINode::next_node_id++);
                    child->parent = parent;
                    child->depth = parent->depth + 1;
                    child->substitution = *mgu;

                    if (depth_map.size() <= child->depth)
                    {
                        depth_map.resize(child->depth + 1);
                    }

                    depth_map[child->depth].push_back(child);
                    parent->children.push_back(child);
                    added_nodes.push_back(child);
                }
                catch (const std::exception &e)
                {
                    std::cout << "Error processing literal: " << e.what() << std::endl;
                    continue;
                }
            }
        }

        return added_nodes;
    }

    bool SLITree::is_ancestor(std::shared_ptr<SLINode> potential_ancestor,
                              std::shared_ptr<SLINode> potential_descendant)
    {
        if (potential_ancestor)
            potential_ancestor = this->findNodeById(potential_ancestor->node_id);
        if (potential_descendant)
            potential_descendant = this->findNodeById(potential_descendant->node_id);

        if (!potential_ancestor || !potential_descendant)
            return false;
        if (potential_ancestor->depth >= potential_descendant->depth)
            return false;

        auto current = potential_descendant;
        while (current && current->depth > potential_ancestor->depth)
        {
            if (auto parent = current->parent.lock())
            {
                parent = this->findNodeById(parent->node_id);
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
        node = this->findNodeById(node->node_id);
        if (!node || !node->is_active)
        {
            std::cout << "not active node or nullptr in slitree::truncate " << std::endl;
            return;
        }
        // std::cout << "slitree::truncate slitree " << std::endl;
        // this->print_tree(kb);
        // bool flag = true;
        // for (const auto &nodes : this->depth_map)
        // {
        //     for (const auto &n : nodes)
        //     {
        //         std::cout << n << "  " << node << " " << int(n == node) << std::endl;
        //         if (n == node)
        //         {
        //             std::cout << "find truncate node" << std::endl;
        //             flag = false;
        //             break;
        //         }
        //     }
        // }
        // if (flag)
        // {
        //     std::cout << "cannot find truncate node in this tree " << node << std::endl;
        //     return;
        // }

        // 删除操作相关的压栈代码
        // auto op = std::make_unique<TruncateOperation>();
        bool truncation_performed = false;

        std::shared_ptr<SLINode> current = node;
        while (current && current->is_active)
        {
            if (current->children.empty())
            {
                // 不再保存状态：之前为 dynamic_cast<TruncateOperation *>(op.get())->save_state(current);
                current->is_active = false;
                current->rule_applied = "t-truncate";
                truncation_performed = true;

                std::shared_ptr<SLINode> parent = current->parent.lock();
                if (parent)
                {
                    parent = this->findNodeById(parent->node_id);
                    auto &siblings = parent->children;
                    siblings.erase(
                        std::remove(siblings.begin(), siblings.end(), current),
                        siblings.end());

                    std::shared_ptr<SLINode> old_parent = parent;
                    current->parent.reset();
                    current = old_parent;
                }
                else
                {
                    current = nullptr;
                }
            }
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
                    // 不再保存状态：之前为 dynamic_cast<TruncateOperation *>(op.get())->save_state(current);
                    current->is_active = false;
                    current->rule_applied = "t-truncate";
                    truncation_performed = true;
                    current = current->parent.lock();

                    std::shared_ptr<SLINode> parent = current ? current->parent.lock() : nullptr;
                    if (parent)
                    {
                        parent = this->findNodeById(parent->node_id);
                        auto &siblings = parent->children;
                        siblings.erase(
                            std::remove(siblings.begin(), siblings.end(), current),
                            siblings.end());

                        std::shared_ptr<SLINode> old_parent = parent;
                        current->parent.reset();
                        current = old_parent;
                    }
                    else
                    {
                        current = nullptr;
                    }
                }
                else
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }

        if (truncation_performed)
        {
            // 删除操作堆栈的压栈操作
            // operation_stack.push(std::move(op));
            cleanup_empty_depths();
        }
    }

    bool SLITree::t_factoring(std::shared_ptr<SLINode> upper_node, std::shared_ptr<SLINode> lower_node)
    {
        upper_node = this->findNodeById(upper_node->node_id);
        lower_node = this->findNodeById(lower_node->node_id);
        if (!upper_node || !lower_node || !upper_node->is_active || !lower_node->is_active || upper_node->is_A_literal || lower_node->is_A_literal)
        {
            std::cout << "basic check failed in t-factoring " << std::endl;
            return false;
        }

        if (upper_node->literal.isNegated() != lower_node->literal.isNegated())
        {
            std::cout << "try factoring a negative and positive literal" << std::endl;
            return false;
        }

        if (upper_node->depth > lower_node->depth)
        {
            std::cout << "deepth wrong in t-factoring" << std::endl;
            std::cout << "upper_node depth " << upper_node->depth << " lower_node_deepth " << lower_node->depth << std::endl;
            std::cout << "upper node " << upper_node->node_id << " lower node " << lower_node->node_id << std::endl;
            this->print_tree(kb);
            return false;
        }

        if (is_ancestor(upper_node, lower_node))
        {
            std::cout << "upper_node is lower_node's ancestor. t-factoring failed" << std::endl;
            return false;
        }

        auto mgu = Unifier::findMGU(upper_node->literal, lower_node->literal, kb);
        if (!mgu)
        {
            std::cout << "Find MGU Failed " << std::endl;
            return false;
        }

        try
        {
            std::vector<std::pair<std::shared_ptr<SLINode>, Literal>> previous_literals;
            std::vector<std::pair<std::shared_ptr<SLINode>, Substitution>> previous_substitutions;
            
            // std::cout << "before factoring in SLITree::t_factoring " << std::endl;
            // this->print_tree(kb);

            for (auto &level : depth_map)
            {
                for (auto &node : level)
                {
                    if (node && node != this->root)
                    {
                        previous_literals.emplace_back(node, node->literal);
                        previous_substitutions.emplace_back(node, node->substitution);

                        node->literal = Unifier::applySubstitutionToLiteral(node->literal, *mgu, kb);
                        for (const auto &[var, term] : *mgu)
                        {
                            node->substitution[var] = term;
                        }
                    }
                }
            }
            // 调用 truncate() 时依然执行树剪枝，移除操作回溯代码
            truncate(lower_node);

            // 已删除 FactoringOperation 及其压栈操作
            upper_node->rule_applied = "t_factoring";
            // std::cout << "after factoring in SLITree::t_factoring " << std::endl;
            // this->print_tree(kb);
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
        upper_node = this->findNodeById(upper_node->node_id);
        lower_node = this->findNodeById(lower_node->node_id);

        if (!upper_node || !lower_node)
            return false;

        if (lower_node->parent.lock() == upper_node || lower_node->depth - upper_node->depth == 1)
        {
            std::cout << "basic check in t_ancestry failed, cannot ancestry with direct parent" << std::endl;
            return false;
        }

        if (!upper_node || !lower_node || !upper_node->is_active || !lower_node->is_active || !upper_node->is_A_literal || lower_node->is_A_literal)
        {
            std::cout << "basic check failed in t-ancestry" << std::endl;
            if (!upper_node->is_A_literal)
            {
                std::cout << "upper node is not A lit " << std::endl;
            }
            return false;
        }

        if (upper_node->literal.isNegated() == lower_node->literal.isNegated())
        {
            std::cout << "literals are not complementary" << std::endl;
            return false;
        }

        if (!is_ancestor(upper_node, lower_node))
        {
            std::cout << "nodes are not in ancestor relationship" << std::endl;
            return false;
        }

        auto mgu = Unifier::findMGU(upper_node->literal, lower_node->literal, kb);
        if (!mgu)
        {
            std::cout << "Find MGU Failed in t-ancestry" << std::endl;
            return false;
        }

        try
        {
            std::vector<std::pair<std::shared_ptr<SLINode>, Literal>> previous_literals;
            std::vector<std::pair<std::shared_ptr<SLINode>, Substitution>> previous_substitutions;

            for (auto &level : depth_map)
            {
                for (auto &node : level)
                {
                    if (node && node != this->root)
                    {
                        previous_literals.emplace_back(node, node->literal);
                        previous_substitutions.emplace_back(node, node->substitution);

                        node->literal = Unifier::applySubstitutionToLiteral(node->literal, *mgu, kb);
                        for (const auto &[var, term] : *mgu)
                        {
                            node->substitution[var] = term;
                        }
                    }
                }
            }

            truncate(lower_node);

            // 已删除 AncestryOperation 及其压栈操作
            upper_node->rule_applied = "t_ancestry";
            return true;
        }
        catch (const std::exception &e)
        {
            std::cout << "Error in t_ancestry: " << e.what() << std::endl;
            return false;
        }
    }

    // 删除 rollback 方法，因为不再需要通过操作栈回滚
    /*
    void SLITree::rollback()
    {
        if (!operation_stack.empty())
        {
            operation_stack.top()->undo();
            operation_stack.pop();
        }
    }
    */

    void SLITree::print_tree(const KnowledgeBase &kb) const
    {
        if (!root)
        {
            std::cout << "Empty tree\n";
            return;
        }

        for (size_t depth = 0; depth < depth_map.size(); ++depth)
        {
            std::cout << "Depth " << depth << ":\n";
            for (const auto &node : depth_map[depth])
            {
                if (node && node->is_active)
                {
                    std::string prefix = "  ";
                    print_node_info(node, kb, prefix, false);
                    if (auto parent = node->parent.lock())
                    {
                        std::cout << prefix << "  └─ Parent: " << parent->node_id << "\n";
                    }
                }
            }
        }
    }

    std::vector<std::shared_ptr<SLINode>> SLITree::get_gamma_L(std::shared_ptr<SLINode> L_node) const
    {
        L_node = this->findNodeById(L_node->node_id);
        if (!L_node)
            return {};
        std::vector<std::shared_ptr<SLINode>> gamma_L;
        auto current = L_node;
        while (auto parent = current->parent.lock())
        {
            parent = this->findNodeById(parent->node_id);
            for (auto &sibling : parent->children)
            {
                if (!sibling->is_A_literal && sibling != L_node && sibling->is_active)
                {
                    gamma_L.push_back(sibling);
                }
            }
            current = parent;
        }
        return gamma_L;
    }

    std::vector<std::shared_ptr<SLINode>> SLITree::get_delta_L(std::shared_ptr<SLINode> L_node) const
    {
        L_node = this->findNodeById(L_node->node_id);
        if (!L_node)
            return {};
        std::vector<std::shared_ptr<SLINode>> delta_L;
        auto current = L_node;
        while (auto parent = current->parent.lock())
        {
            parent = this->findNodeById(parent->node_id);
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
        L_node = this->findNodeById(L_node->node_id);
        if (!L_node)
            return false;
        auto gamma_L = get_gamma_L(L_node);
        auto delta_L = get_delta_L(L_node);

        if (!L_node->is_A_literal)
        {
            gamma_L.push_back(L_node);
            for (size_t i = 0; i < gamma_L.size(); i++)
            {
                for (size_t j = i + 1; j < gamma_L.size(); j++)
                {
                    if (have_same_atom(gamma_L[i], gamma_L[j]))
                    {
                        std::cout << "SLITree::check_AC have same atom in gammal_L ac failed" << std::endl;
                        gamma_L[i]->print(kb);
                        gamma_L[j]->print(kb);
                        std::cout << gamma_L[i] <<" " << gamma_L[j] << std::endl;
                        return false;
                    }
                }
            }
        }

        delta_L.push_back(L_node);
        for (size_t i = 0; i < delta_L.size(); i++)
        {
            for (size_t j = i + 1; j < delta_L.size(); j++)
            {
                if (have_same_atom(delta_L[i], delta_L[j]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    bool SLITree::check_MC(const std::shared_ptr<SLINode> &node) const
    {
        auto cur = this->findNodeById(node->node_id);
        if (!cur)
            return false;

        if (cur->children.empty() && cur->is_A_literal)
        {
            return false;
        }

        for (const auto &child : cur->children)
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
        auto all_nodes = get_all_active_nodes();
        for (const auto &node : all_nodes)
        {
            if (!check_AC(node))
            {
                return false;
            }
        }
        return true;
    }

    bool SLITree::check_all_nodes_MC() const
    {
        auto all_nodes = get_all_active_nodes();
        for (const auto &node : all_nodes)
        {
            if (!check_MC(node))
            {
                return false;
            }
        }
        return true;
    }

    bool SLITree::have_same_atom(const std::shared_ptr<SLINode> &node1, const std::shared_ptr<SLINode> &node2) const
    {
        Literal lit1 = node1->literal;
        Literal lit2 = node2->literal;
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

        std::cout << prefix << (prefix.empty() ? "" : get_branch_str(is_last))
                  << node->literal.toString(kb)
                  << (node->is_A_literal ? "*" : "");

        std::cout << " [" << node->node_id << "|d:" << node->depth << "]";

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

        std::cout << " (";
        std::vector<std::string> status;
        if (!node->is_active)
            status.push_back("inactive");
        if (node->is_A_literal)
            status.push_back("A-lit");
        if (status.empty())
            status.push_back("active");
        std::cout << join(status, ",") << ")";

        if (node->parent.lock())
        {
            std::cout << " parent:" << node->parent.lock()->node_id;
        }
        std::cout << " children:" << node->children.size();

        if (!node->rule_applied.empty())
        {
            std::cout << " rule:" << node->rule_applied;
        }

        std::cout << "\n";
    }

    std::vector<std::shared_ptr<SLINode>> SLITree::get_active_nodes_at_depth(int depth) const
    {
        std::vector<std::shared_ptr<SLINode>> active_nodes;
        if (depth < 0 || depth >= depth_map.size())
        {
            return active_nodes;
        }
        for (const auto &node : depth_map[depth])
        {
            if (node && node->is_active && !node->is_A_literal)
            {
                active_nodes.push_back(node);
            }
        }
        return active_nodes;
    }

    void SLITree::cleanup_empty_depths()
    {
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
        if (last_non_empty >= 0)
        {
            depth_map.resize(last_non_empty + 1);
        }
        else
        {
            depth_map.resize(1);
        }
    }

    // 拷贝构造函数，根据传入的起始节点进行局部复制
    SLITree::SLITree(const SLITree &other, std::shared_ptr<SLINode> startNode)
        : kb(other.kb)
    {
        std::unordered_map<int, std::shared_ptr<SLINode>> nodeMap;
        root = std::make_shared<SLINode>(other.root->literal,
                                         other.root->is_A_literal,
                                         other.root->node_id);
        root->depth = 0;
        root->is_active = other.root->is_active;
        nodeMap[other.root->node_id] = root;

        for (size_t depth = 0; depth < other.depth_map.size(); ++depth)
        {
            for (const auto &oldNode : other.depth_map[depth])
            {
                if (oldNode == other.root)
                    continue;

                auto newNode = std::make_shared<SLINode>(oldNode->literal,
                                                         oldNode->is_A_literal,
                                                         oldNode->node_id);
                newNode->depth = oldNode->depth;
                newNode->is_active = oldNode->is_active;
                newNode->substitution = oldNode->substitution;
                newNode->rule_applied = oldNode->rule_applied;
                newNode->is_A_literal = oldNode->is_A_literal;

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
                nodeMap[oldNode->node_id] = newNode;
            }
        }

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
                    node = nullptr;
                }
            }
        }
    }

    std::shared_ptr<SLINode> SLITree::copyNode(const std::shared_ptr<SLINode> &node)
    {
        auto new_node = std::make_shared<SLINode>(node->literal,
                                                  node->is_A_literal,
                                                  node->node_id);
        new_node->depth = node->depth;
        new_node->is_active = node->is_active;
        new_node->substitution = node->substitution;
        new_node->rule_applied = node->rule_applied;
        return new_node;
    }

    size_t SLITree::computeNodeHash(const std::shared_ptr<SLINode> &node) const
    {
        if (!node)
            return 0;

        size_t hash = 0;
        hash ^= node->literal.hash();
        hash ^= std::hash<bool>{}(node->is_A_literal);
        hash ^= std::hash<bool>{}(node->is_active);
        hash ^= std::hash<int>{}(node->depth);

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
        for (const auto &level : depth_map)
        {
            for (const auto &node : level)
            {
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
        for (const auto &level : depth_map)
        {
            for (const auto &node : level)
            {
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

        for (const auto &level : depth_map)
        {
            size_t level_hash = 1;
            int current_depth = level.front()->depth;

            for (const auto &node : level)
            {
                if (node && node->is_active)
                {
                    size_t node_hash = computeNodeHash(node);
                    if (auto parent = node->parent.lock())
                    {
                        size_t parent_literal_hash = parent->literal.hash();
                        size_t parent_depth_hash = std::hash<int>{}(parent->depth);
                        size_t parent_info = parent_literal_hash + parent_depth_hash;
                        node_hash = node_hash * PRIME + parent_info;
                    }

                    size_t prev_level_hash = level_hash;
                    level_hash = level_hash * PRIME + node_hash;
                }
            }

            size_t depth_hash = std::hash<int>{}(current_depth);
            size_t prev_hash = hash;
            hash = hash * PRIME + (level_hash + depth_hash);
        }

        return hash;
    }

    bool SLITree::areNodesEquivalent(const std::shared_ptr<SLINode> &node1,
                                     const std::shared_ptr<SLINode> &node2) const
    {
        if (!node1 || !node2)
            return node1 == node2;

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

        for (size_t i = 0; i < depth_map.size(); ++i)
        {
            std::vector<std::shared_ptr<SLINode>> active1, active2;

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