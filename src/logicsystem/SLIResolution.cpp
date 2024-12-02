// SLIResolution.cpp
#include "SLIResolution.h"
#include <iostream>

namespace LogicSystem
{

    bool SLIResolution::prove(KnowledgeBase &kb, const Clause &goal, SearchStrategy &strategy)
    {
        // 添加visited集合
        std::unordered_set<size_t> visited_states;
        // 创建初始树
        std::unique_ptr<SLITree> currentTree = std::make_unique<SLITree>(kb);

        // 记录初始状态
        visited_states.insert(currentTree->computeStateHash());

        auto initial_nodes = currentTree->add_node(goal, Literal(), false, currentTree->getRoot());

        if (initial_nodes.empty())
        {
            return false;
        }

        // 队列中存储<消解对, 树状态>对
        struct SearchState
        {
            SLIResolutionPair pair;
            std::unique_ptr<SLITree> tree;
            SearchState(const SLIResolutionPair &p, std::unique_ptr<SLITree> t)
                : pair(p), tree(std::move(t)) {}
        };
        std::queue<SearchState> stateQueue;

        // 初始化第一批消解对
        for (const auto &node : initial_nodes)
        {
            for (const auto &kb_clause : kb.getClauses())
            {
                for (const auto &lit : kb_clause.getLiterals())
                {
                    if (Resolution::isComplementary(node->literal, lit))
                    {
                        double score = calculateHeuristic(kb_clause, node, lit);
                        if (strategy.shouldTryResolution(score))
                        {
                            auto newTree = std::make_unique<SLITree>(*currentTree, node);

                            // 计算新状态的哈希值并检查是否访问过
                            size_t new_state_hash = newTree->computeStateHash();
                            if (visited_states.find(new_state_hash) != visited_states.end())
                            {
                                std::cout << "Initial state already visited, skipping..." << std::endl;
                                continue;
                            }

                            // 记录新状态
                            visited_states.insert(new_state_hash);

                            stateQueue.emplace(SLIResolutionPair(node, kb_clause, lit, score),
                                               std::move(newTree));

                            std::cout << "Successfully emplaced initial resolution pair" << std::endl;
                        }
                    }
                }
            }
        }

        int count = 0;
        while (!stateQueue.empty())
        {
            count++;
            std::cout << "round " << count << std::endl;
            if (count >= 5)
                return false;

            // 获取下一个状态
            auto current_state = std::move(stateQueue.front());
            stateQueue.pop();

            // // 计算当前状态的哈希值
            // size_t current_hash = current_state.tree->computeStateHash();

            // // 检查是否已访问过该状态
            // if (visited_states.find(current_hash) != visited_states.end())
            // {
            //     continue;
            // }

            // // 记录当前状态
            // visited_states.insert(current_hash);

            // 在新树中找到对应的节点
            auto corresponding_node = current_state.tree->findNodeById(current_state.pair.node_id);
            if (!corresponding_node)
            {
                std::cout << "Cannot find corresponding node, skipping..." << std::endl;
                continue;
            }

            std::cout << "Tree After before nodes " << std::endl;
            current_state.tree->print_tree(kb);

            std::cout << "Parent Node before add" << std::endl;
            corresponding_node->print(kb);

            // 在当前树状态上执行消解
            auto resolvent_nodes = current_state.tree->add_node(
                current_state.pair.kb_clause,
                current_state.pair.resolving_literal,
                true,
                corresponding_node);

            // 使用hasSelfLoop()来判断
            if (current_state.tree->hasSelfLoop())
            {
                std::cout << "Skipping state due to self-loop detection" << std::endl;
                continue;
            }

            // 在此处检查状态是否已访问过
            size_t current_state_hash = current_state.tree->computeStateHash();
            if (visited_states.find(current_state_hash) != visited_states.end())
            {
                std::cout << "State already visited, skipping..." << std::endl;
                continue;
            }

            // 记录新状态
            visited_states.insert(current_state_hash);

            std::cout << "Parent Node after add" << std::endl;
            corresponding_node->print(kb);

            std::cout << "Tree After add nodes " << std::endl;
            current_state.tree->print_tree(kb);

            // 检查add_node后是否需要truncate
            std::cout << "resolvent_nodes.size " << resolvent_nodes.size()
                      << " corresponding_node->children.empty "
                      << corresponding_node->children.empty() << std::endl;
            if (resolvent_nodes.empty() || corresponding_node->children.empty())
            {
                checkAndTruncateNode(corresponding_node, *current_state.tree);
            }

            std::cout << "Tree After add nodes and truncate " << std::endl;
            current_state.tree->print_tree(kb);

            // 应用归约规则
            auto factoring_pairs = findPotentialFactoringPairs(resolvent_nodes,
                                                               current_state.tree->getDepthMap(),
                                                               kb);
            for (const auto &[upper_node, lower_node] : factoring_pairs)
            {
                if (current_state.tree->t_factoring(upper_node, lower_node))
                {
                    std::cout << "Applied t-factoring successfully between nodes:\n";
                    std::cout << "Upper node: " << upper_node->literal.toString(kb) << "\n";
                    std::cout << "Lower node: " << lower_node->literal.toString(kb) << "\n";

                    if (auto parent = lower_node->parent.lock())
                    {
                        checkAndTruncateNode(parent, *current_state.tree);
                    }
                }
            }

            std::cout << "Tree After factoring and truncate " << std::endl;
            current_state.tree->print_tree(kb);

            auto ancestry_pairs = findPotentialAncestryPairs(resolvent_nodes, kb);
            for (const auto &[ancestor, descendant] : ancestry_pairs)
            {
                if (current_state.tree->t_ancestry(ancestor, descendant))
                {
                    std::cout << "Applied t-ancestry successfully" << std::endl;

                    if (auto parent = descendant->parent.lock())
                    {
                        checkAndTruncateNode(parent, *current_state.tree);
                    }
                }
            }

            std::cout << "Tree After ancestry and truncate " << std::endl;
            current_state.tree->print_tree(kb);
            // 检查是否找到证明
            if (checkEmptyClause(*current_state.tree))
            {
                return true;
            }
            // 为新节点生成新的状态
            for (const auto &node : resolvent_nodes)
            {
                if (!node->is_A_literal)
                {
                    for (const auto &kb_clause : kb.getClauses())
                    {
                        for (const auto &lit : kb_clause.getLiterals())
                        {
                            if (Resolution::isComplementary(node->literal, lit))
                            {
                                std::cout << "Found Complementary Literal " << std::endl;
                                std::cout << "Lit1 " << node->literal.toString(kb)
                                          << " Lit2 " << lit.toString(kb) << std::endl;
                                double score = calculateHeuristic(kb_clause, node, lit);
                                if (strategy.shouldTryResolution(score))
                                {
                                    auto newTree = std::make_unique<SLITree>(*current_state.tree, node);

                                    /*// 计算新状态的哈希值并检查是否访问过
                                    size_t new_state_hash = newTree->computeStateHash();
                                    if (visited_states.find(new_state_hash) != visited_states.end())
                                    {
                                        std::cout << "State already visited, skipping..." << std::endl;
                                        continue;
                                    }

                                    // 记录新状态
                                    visited_states.insert(new_state_hash);*/

                                    std::cout << "Old Tree after construction: " << std::endl;
                                    current_state.tree->print_tree(kb);
                                    std::cout << "New Tree after construction:" << std::endl;
                                    newTree->print_tree(kb);
                                    std::cout << "KB Clause: " << kb_clause.toString(kb) << std::endl;

                                    stateQueue.emplace(SLIResolutionPair(node, kb_clause, lit, score),
                                                       std::move(newTree));
                                    std::cout << "Successfully emplaced new resolution pair" << std::endl;
                                }
                            }
                        }
                    }
                }
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

    bool SLIResolution::checkEmptyClause(const SLITree &tree)
    {
        // 获取整个树的深度映射
        auto &depth_map = tree.getDepthMap();

        // 统计所有深度的active节点
        int active_count = 0;

        for (size_t depth = 0; depth < depth_map.size(); ++depth)
        {
            for (const auto &node : depth_map[depth])
            {
                if (node->is_active)
                {
                    active_count++;
                    // 如果在非根节点层发现active节点，直接返回false
                    if (depth > 0)
                    {
                        return false;
                    }
                }
            }
        }

        // 只有根节点是active时返回true
        return active_count == 1;
    }

    std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
    SLIResolution::findPotentialFactoringPairs(
        const std::vector<std::shared_ptr<SLINode>> &new_nodes,
        const std::vector<std::vector<std::shared_ptr<SLINode>>> &depth_map,
        KnowledgeBase &kb)
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
                        std::find(new_nodes.begin(), new_nodes.end(), existing_node) == new_nodes.end() &&
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
    SLIResolution::findPotentialAncestryPairs(const std::vector<std::shared_ptr<SLINode>> &new_nodes, KnowledgeBase &kb)
    {
        std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>> pairs;
        std::cout << "Searching for potential ancestry pairs..." << std::endl;
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
        std::cout << "Found " << pairs.size() << " potential ancestry pairs" << std::endl;
        return pairs;
    }

    void SLIResolution::checkAndTruncateNode(const std::shared_ptr<SLINode> &node, SLITree &tree)
    {
        if (node && node->is_active && node->is_A_literal)
        { // 是A-lit
            if (node->children.empty())
            { // 没有孩子节点
                tree.truncate(node);
            }
        }
    }
} // namespace LogicSystem