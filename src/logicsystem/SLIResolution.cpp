// SLIResolution.cpp
#include "SLIResolution.h"
#include <iostream>

namespace LogicSystem
{

    bool SLIResolution::prove(const KnowledgeBase &kb, const Clause &goal, SearchStrategy &strategy)
    {
        // 1. 初始化SLI树，添加否定的目标子句
        SLITree tree(kb);
        auto initial_nodes = tree.add_node(goal, Literal(), false, tree.getRoot());

        if (initial_nodes.empty())
        {
            std::cout << "Failed to add initial goal clause to SLI tree" << std::endl;
            return false;
        }

        // 保存所有活跃节点
        std::vector<std::shared_ptr<SLINode>> active_nodes = initial_nodes;

        int count = 0;

        while (!active_nodes.empty())
        {
            count ++;
            std::cout << "round " << count + 1 << std::endl;
            if(count >= 2)
            {
                return false;
            }
            std::cout << "Current active nodes: " << active_nodes.size() << std::endl;
            tree.print_tree(kb);

            // 2. 子句选择和配对
            for (const auto &node : active_nodes)
            {
                for (const auto &kb_clause : kb.getClauses())
                {
                    for (size_t i = 0; i < kb_clause.getLiterals().size(); ++i)
                    {
                        const auto &lit = kb_clause.getLiterals()[i];
                        if (Resolution::isComplementary(node->literal, lit))
                        {
                            double score = calculateHeuristic(kb_clause, node, lit);
                            if (strategy.shouldTryResolution(score))
                            {
                                strategy.addSLIPair(SLIResolutionPair(node, kb_clause, lit, score));
                            }
                        }
                    }
                }
            }

            if (strategy.isEmpty())
            {
                std::cout << "No more resolution pairs to try" << std::endl;
                return false;
            }

            // 获取下一个要尝试的消解对
            int SLIPairCounts = 0;
            std::vector<std::shared_ptr<SLINode>> new_nodes;
            while (!strategy.isEmpty() && new_nodes.empty())
            {
                auto next_pair = strategy.getNextSLI();
                auto resolvent_nodes = tree.add_node(
                    next_pair.kb_clause,
                    next_pair.resolving_literal,
                    true,
                    next_pair.tree_node);
                new_nodes.insert(new_nodes.end(), resolvent_nodes.begin(), resolvent_nodes.end());
            }

            if (new_nodes.empty())
            {
                std::cout << "Failed to generate new nodes from resolution" << std::endl;
                //continue;
            }

            std::cout << "After add nodes in SLI Resolution" << std::endl;
            tree.print_tree(kb);
            // 3. 应用t-factoring
            // 在 SLIResolution::prove 中的 t-factoring 部分
            auto factoring_pairs = findPotentialFactoringPairs(new_nodes, tree.getDepthMap(), kb);
            for (const auto &[upper_node, lower_node] : factoring_pairs)
            {
                if (tree.t_factoring(upper_node, lower_node))
                {
                    std::cout << "Applied t-factoring successfully between nodes:\n";
                    std::cout << "Upper node: " << upper_node->literal.toString(kb) << "\n";
                    std::cout << "Lower node: " << lower_node->literal.toString(kb) << "\n";
                }
            }

            // 4. 应用t-ancestry
            auto ancestry_pairs = findPotentialAncestryPairs(new_nodes, kb);
            for (const auto &[ancestor, descendant] : ancestry_pairs)
            {
                if (tree.t_ancestry(ancestor, descendant))
                {
                    std::cout << "Applied t-ancestry successfully" << std::endl;
                }
            }

            // 5. 应用t-truncate
            // fix: 这里逻辑错误，应该是针对没有孩子的A-lit
            for (const auto &node : new_nodes)
            {
                if (node->is_active)
                { // 只对仍然活跃的节点应用truncate
                    tree.truncate(node);
                }
            }

            // 6. 检查是否得到空子句
            if (checkEmptyClause(new_nodes))
            {
                std::cout << "Empty clause found - proof completed" << std::endl;
                return true;
            }

            // 更新启发式信息
            strategy.updateHeuristic(new_nodes);

            // 7. 回溯检查
            if (strategy.shouldBacktrack())
            {
                std::cout << "Performing backtrack" << std::endl;
                tree.rollback();
                // 重新收集活跃节点
                active_nodes.clear();
                for (size_t depth = 0; depth < tree.getDepthMap().size(); ++depth)
                {
                    for (const auto &node : tree.get_active_nodes_at_depth(depth))
                    {
                        if (node->is_active)
                        {
                            active_nodes.push_back(node);
                        }
                    }
                }
            }
            else
            {
                // 更新活跃节点列表为新生成的节点
                active_nodes = new_nodes;
            }

            // 检查资源限制
            if (strategy.getSearchedStates() >= 1000000)
            { // 示例限制
                std::cout << "Search limit reached" << std::endl;
                return false;
            }
        }

        return false;
    }

    double SLIResolution::calculateHeuristic(const Clause &kb_clause,
                                             const std::shared_ptr<SLINode> &tree_node,
                                             const Literal &resolving_literal)
    {
        // 基础分数
        double score = 1.0;

        // 考虑子句长度 - 更短的子句更优先
        score -= 0.1 * kb_clause.getLiterals().size();

        // 考虑深度 - 较浅的节点更优先
        score -= 0.05 * tree_node->depth;

        // 考虑变量数量 - 更少的变量更优先
        int var_count = 0;
        for (const auto &arg : resolving_literal.getArgumentIds())
        {
            if (arg.type == SymbolType::VARIABLE)
            {
                var_count++;
            }
        }
        score -= 0.1 * var_count;

        return score;
    }

    // findPotentialFactoringPairs 和 findPotentialAncestryPairs 实现保持不变

    bool SLIResolution::checkEmptyClause(const std::vector<std::shared_ptr<SLINode>> &nodes)
    {
        for (const auto &node : nodes)
        {
            if (node->is_active && node->literal.isEmpty())
            {
                return true;
            }
        }
        return false;
    }

    std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
    SLIResolution::findPotentialFactoringPairs(
        const std::vector<std::shared_ptr<SLINode>> &new_nodes,
        const std::vector<std::vector<std::shared_ptr<SLINode>>> &depth_map,
        const KnowledgeBase &kb)
    {
        std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>> factoring_pairs;

        std::cout << "Searching for potential factoring pairs..." << std::endl;

        for (const auto &new_node : new_nodes)
        {
            std::cout << "Checking new node: " << new_node->literal.toString(kb)
                      << " at depth " << new_node->depth << std::endl;

            for (size_t d = 0; d <= new_node->depth; ++d)
            {
                for (const auto &existing_node : depth_map[d])
                {
                    if (existing_node->depth > new_node->depth)
                        continue;
                    if (!existing_node->is_active)
                        continue;

                    std::cout << "Comparing with existing node: " << existing_node->literal.toString(kb)
                              << " at depth " << existing_node->depth << std::endl;

                    if (existing_node != new_node &&
                        existing_node->literal.getPredicateId() == new_node->literal.getPredicateId() &&
                        existing_node->literal.isNegated() == new_node->literal.isNegated() &&
                        existing_node->literal.getArgumentIds().size() == new_node->literal.getArgumentIds().size())
                    {
                        std::cout << "Found potential match, attempting MGU..." << std::endl;
                        auto mgu = Unifier::findMGU(existing_node->literal, new_node->literal, kb);
                        if (mgu)
                        {
                            std::cout << "MGU found! Adding to factoring pairs" << std::endl;
                            factoring_pairs.emplace_back(existing_node, new_node);

                            // 打印MGU信息
                            std::cout << "MGU details:" << std::endl;
                            for (const auto &[var, term] : *mgu)
                            {
                                std::cout << "  " << kb.getSymbolName(var)
                                          << " -> " << kb.getSymbolName(term) << std::endl;
                            }
                        }
                        else
                        {
                            std::cout << "No MGU found" << std::endl;
                        }
                    }
                }
            }
        }

        std::cout << "Found " << factoring_pairs.size() << " potential factoring pairs" << std::endl;
        return factoring_pairs;
    }

    std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
    SLIResolution::findPotentialAncestryPairs(const std::vector<std::shared_ptr<SLINode>> &new_nodes, const KnowledgeBase &kb)
    {
        std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>> pairs;

        // 对于每个新节点
        for (const auto &new_node : new_nodes)
        {
            // 获取当前节点的所有祖先
            std::shared_ptr<SLINode> current = new_node->parent.lock();
            while (current)
            {
                // 检查ancestry的基本条件
                if (current->is_active &&
                    current->literal.getPredicateId() == new_node->literal.getPredicateId() &&
                    current->literal.isNegated() != new_node->literal.isNegated() &&
                    current->literal.getArgumentIds().size() == new_node->literal.getArgumentIds().size())
                {
                    // 检查是否存在MGU
                    auto mgu = Unifier::findMGU(current->literal, new_node->literal, kb);
                    if (mgu)
                    {
                        pairs.emplace_back(current, new_node);
                    }
                }
                current = current->parent.lock();
            }
        }

        return pairs;
    }
} // namespace LogicSystem