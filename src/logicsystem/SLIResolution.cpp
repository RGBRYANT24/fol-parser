// SLIResolution.cpp
#include "SLIResolution.h"
#include <iostream>

namespace LogicSystem
{
    int ProofState::next_id = 0;

    bool SLIResolution::prove(KnowledgeBase &kb, const Clause &goal, SearchStrategy &strategy)
    {
        StateManager state_manager;

        // 创建初始状态
        std::unique_ptr<SLITree> initialTree = std::make_unique<SLITree>(kb);
        auto initial_nodes = initialTree->add_node(goal, Literal(), false, initialTree->getRoot());

        if (initial_nodes.empty())
        {
            return false;
        }

        auto initial_state = state_manager.createState(std::move(initialTree));

        // 记录访问过的状态
        // std::unordered_set<size_t> visited_states;

        // 搜索栈
        std::stack<StateManager::StatePtr> state_stack;
        state_stack.push(initial_state);

        StateManager::StatePtr successful_state = nullptr;

        while (!state_stack.empty())
        {
            auto current_state = state_stack.top();
            state_stack.pop();

            // 状态去重检查
            // size_t state_hash = current_state->tree->computeStateHash();
            // if (visited_states.find(state_hash) != visited_states.end()) {
            //     continue;
            // }
            // visited_states.insert(state_hash);

            // 基本条件检查
            bool AC_result = current_state->tree->check_all_nodes_AC();
            bool MC_result = current_state->tree->check_all_nodes_MC();

            // 检查空子句
            if (checkEmptyClause(*current_state->tree))
            {
                successful_state = current_state;
                return true;
            }

            auto active_nodes = current_state->tree->get_all_active_nodes();

            if (AC_result && MC_result)
            {
                // t-extension

                // t-factoring

                // t-ancestry

                // t-truncate
            }

            else if (AC_result)
            {
                // t-factoring

                // t-ancestry
            }

            else if (MC_result)
            {
                // t-truncate
            }
            else
            {
                continue;
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

        return tree.get_all_active_nodes().size() == 0 ? true : false;

        // // 统计所有深度的active节点
        // int active_count = 0;

        // for (size_t depth = 0; depth < depth_map.size(); ++depth)
        // {
        //     for (const auto &node : depth_map[depth])
        //     {
        //         if (node->is_active)
        //         {
        //             active_count++;
        //             // 如果在非根节点层发现active节点，直接返回false
        //             if (depth > 0)
        //             {
        //                 return false;
        //             }
        //         }
        //     }
        // }

        // // 只有根节点是active时返回true
        // return active_count == 1;
    }

    std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
    SLIResolution::findPotentialFactoringPairs(const std::shared_ptr<SLITree> &tree)
    {
        std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>> factoring_pairs;

        // 获取所有的B文字
        auto b_lit_nodes = tree->get_all_B_literals();

        for (const auto &node : b_lit_nodes)
        {
            // 获取当前node的gamma_L集合
            auto gamma_nodes = tree->get_gamma_L(node);
            //std::cout << "node " << node->node_id << " gamma_nodes size " << gamma_nodes.size() << std::endl;
            // 遍历gamma_node 检查是否能进行factoring
            for (const auto &node_m : gamma_nodes)
            {
                // 只在第一个节点的地址大于第二个节点的地址时才添加配对 因为第一个节点node是下层的节点 后添加的 地址会大
                if (node != node_m &&
                    node->node_id > node_m->node_id &&
                    // (node->depth == node_m->depth && node.get() > node_m.get()) &&
                    node->literal.getPredicateId() == node_m->literal.getPredicateId() &&
                    node->literal.isNegated() == node_m->literal.isNegated() &&
                    node->literal.getArgumentIds().size() == node_m->literal.getArgumentIds().size())
                {
                    factoring_pairs.emplace_back(node, node_m);
                }
            }
        }
        return factoring_pairs;
    }

    std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
    SLIResolution::findPotentialAncestryPairs(const std::shared_ptr<SLITree> &tree)
    {
        std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>> ancestry_pairs;

        // 获取所有的B文字
        auto b_lit_nodes = tree->get_all_B_literals();
        // std::cout << "Searching for potential ancestry pairs..." << std::endl;
        // 对于每个新节点
        for (const auto &node : b_lit_nodes)
        {
            // 获取当前节点的所有祖先
            std::shared_ptr<SLINode> current = node->parent.lock();
            while (current)
            {
                // 检查ancestry的基本条件
                if (node->parent.lock() != current &&
                    current->literal.getPredicateId() == node->literal.getPredicateId() &&
                    current->literal.isNegated() != node->literal.isNegated() &&
                    current->literal.getArgumentIds().size() == node->literal.getArgumentIds().size())
                {
                    ancestry_pairs.emplace_back(current, node);
                }
                current = current->parent.lock();
            }
        }
        // std::cout << "Found " << pairs.size() << " potential ancestry pairs" << std::endl;
        return ancestry_pairs;
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

    // 辅助函数：生成t-extension状态
    void SLIResolution::generateExtensionStates(
        KnowledgeBase &kb,
        const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack)
    {
        for (const auto &node : b_lit_nodes)
        {
            if (!node->is_active || node->is_A_literal)
                continue;

            for (const auto &kb_clause : kb.getClauses())
            {
                for (const auto &lit : kb_clause.getLiterals())
                {
                    if (Resolution::isComplementary(node->literal, lit))
                    {
                        // 直接使用当前树的指针
                        auto new_state = std::make_shared<SLIOperation::OperationState>(
                            current_state->sli_tree, // 直接使用现有树
                            SLIActionType::EXTENSION,
                            node, // 直接使用原始节点
                            SecondOperand(lit),
                            kb_clause,
                            current_state);
                        state_stack.push(new_state);
                    }
                }
            }
        }
        return;
    }

    // 辅助函数：生成t-factoring状态
    void SLIResolution::generateFactoringStates(
        const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack)
    {
        auto factoring_pairs = findPotentialFactoringPairs(current_state->sli_tree);
        for (const auto &[upper_node, lower_node] : factoring_pairs)
        {
            auto new_state = SLIOperation::createFactoringState(
                current_state->sli_tree,
                upper_node,
                lower_node,
                current_state);
            state_stack.push(new_state);
        }
    }

    // 辅助函数：生成t-ancestry状态
    void SLIResolution::generateAncestryStates(
        const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack)
    {
        auto ancestry_pairs = findPotentialAncestryPairs(current_state->sli_tree);
        for (const auto &[upper_node, lower_node] : ancestry_pairs)
        {
            auto new_state = SLIOperation::createAncestryState(
                current_state->sli_tree,
                upper_node,
                lower_node,
                current_state);
            state_stack.push(new_state);
        }
    }

    void SLIResolution::printProofPath(std::shared_ptr<ProofState> state, KnowledgeBase &kb)
    {
        // std::vector<std::shared_ptr<ProofState>> path;
        // auto current = state;

        // while (current)
        // {
        //     path.push_back(current);
        //     current = current->parent;
        // }

        // std::reverse(path.begin(), path.end());

        // std::cout << "\n====== Proof Path ======\n";
        // for (size_t i = 0; i < path.size(); ++i)
        // {
        //     std::cout << "\nStep " << i << " (State ID: " << path[i]->state_id << "):\n";
        //     if (i > 0)
        //     {
        //         std::cout << "Applied resolution:\n";
        //         std::cout << "- Node ID: " << path[i]->resolution_pair.node_id << "\n";
        //         std::cout << "- KB Clause: " << path[i]->resolution_pair.kb_clause.toString(kb) << "\n";
        //         std::cout << "- Resolving literal: " << path[i]->resolution_pair.resolving_literal.toString(kb) << "\n";
        //     }
        //     std::cout << "\nResulting Tree:\n";
        //     path[i]->tree->print_tree(kb);
        //     std::cout << "\n----------------------\n";
        // }
        // std::cout << "====== End of Proof ======\n";
    }
} // namespace LogicSystem