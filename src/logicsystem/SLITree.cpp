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
                // 从root开始，对整棵树应用MGU替换
                std::function<void(std::shared_ptr<SLINode>)> applyMGUToTree;
                applyMGUToTree = [this, &mgu, &applyMGUToTree](std::shared_ptr<SLINode> node)
                {
                    if (node != this->root)
                    { // 跳过root节点，因为它不包含实际的文字
                        // 保存旧的hash用于更新literal_map
                        size_t old_hash = node->literal.hash();

                        // 应用替换
                        node->literal = Unifier::applySubstitutionToLiteral(
                            node->literal, *mgu, this->kb);

                        // 更新literal_map
                        if (old_hash != node->literal.hash())
                        {
                            literal_map.erase(old_hash);
                            literal_map[node->literal.hash()] = node;
                        }
                    }

                    // 递归处理所有子节点
                    for (auto &child : node->children)
                    {
                        applyMGUToTree(child);
                    }
                };

                // 从root开始应用MGU
                applyMGUToTree(root);

                // 标记参与消解的节点为A-literal
                parent->is_A_literal = true;

                std::cout << "Tree after MGU application:" << std::endl;
                print_tree(kb);
            }
            catch (const std::exception &e)
            {
                std::cout << "Error applying substitution: " << e.what() << std::endl;
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
                    if (substituted_lit.getPredicateId() == kb.getPredicateId("E"))
                    {
                        const auto &args = substituted_lit.getArgumentIds();
                        if (args.size() == 2 && args[0] == args[1])
                        {
                            has_self_loop = true; // 设置标志
                            return {};
                        }
                    }

                    if (substituted_lit.isEmpty())
                    {
                        std::cout << "Warning: Substitution resulted in empty literal, skipping" << std::endl;
                        continue;
                    }

                    std::cout << "Creating node with substituted literal: " << substituted_lit.toString(this->kb) << std::endl;

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
            // Case 1: Node is a leaf and dont need it is an A-literal
            // if (current->children.empty() && current->is_A_literal)
            if (current->children.empty())
            {
                std::cout << "truncate node " << current->literal.toString(kb) << std::endl;
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
            std::cout << "Factoring uppper lit1 " << upper_node->literal.toString(kb) << "Factoring lower lit2 " << lower_node->literal.toString(kb) << std::endl;
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

    // 在SLITree.cpp中
    SLITree::SLITree(const SLITree &other, std::shared_ptr<SLINode> startNode)
        : kb(other.kb)
    {
        // 创建节点映射表，用于维护新旧节点的对应关系
        std::unordered_map<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>> nodeMap;

        // 复制根节点
        root = std::make_shared<SLINode>(other.root->literal,
                                         other.root->is_active,
                                         other.root->node_id);
        root->depth = 0;
        root->is_active = other.root->is_active;
        nodeMap[other.root] = root;

        // 从根节点开始，按层复制整棵树
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

                // 找到并设置父节点
                if (auto oldParent = oldNode->parent.lock())
                {
                    auto it = nodeMap.find(oldParent);
                    if (it != nodeMap.end())
                    {
                        newNode->parent = it->second;
                        it->second->children.push_back(newNode);
                    }
                }

                // 将新节点加入映射表
                nodeMap[oldNode] = newNode;
            }
        }

        // 复制深度图和文字映射
        depth_map = other.depth_map;
        for (auto &level : depth_map)
        {
            for (auto &node : level)
            {
                auto it = nodeMap.find(node);
                if (it != nodeMap.end())
                {
                    node = it->second;
                }
            }
        }

        // 复制文字映射
        for (const auto &[hash, oldNode] : other.literal_map)
        {
            auto it = nodeMap.find(oldNode);
            if (it != nodeMap.end())
            {
                literal_map[hash] = it->second;
            }
        }
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

    size_t SLITree::computeStateHash() const
    {
        size_t hash = 0;

        // 按层遍历所有活跃节点
        for (const auto &level : depth_map)
        {
            for (const auto &node : level)
            {
                if (node && node->is_active)
                {
                    hash ^= computeNodeHash(node);
                    // 考虑节点间的关系
                    if (auto parent = node->parent.lock())
                    {
                        hash ^= std::hash<int>{}(parent->node_id);
                    }
                }
            }
        }

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

} // namespace LogicSystem