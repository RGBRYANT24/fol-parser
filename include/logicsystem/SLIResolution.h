// SLIResolution.h
#ifndef LOGIC_SYSTEM_SLI_RESOLUTION_H
#define LOGIC_SYSTEM_SLI_RESOLUTION_H

#include "SLITree.h"
#include "KnowledgeBase.h"
#include "SearchStrategy.h"
#include "StateManager.h"
#include "SLIOperation.h"
#include <optional>
#include <vector>

namespace LogicSystem
{
    class SLIResolution
    {
    public:
        static bool prove(KnowledgeBase &kb, const Clause &goal, SearchStrategy &strategy);
        static bool prove(KnowledgeBase &kb, const Clause &goal);

        // 辅助函数：生成t-extension状态
        static void generateExtensionStates(
            KnowledgeBase &kb,
            const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
            const std::shared_ptr<SLIOperation::OperationState> &current_state,
            std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack);

        // 辅助函数：生成t-factoring状态
        static void generateFactoringStates(
            const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
            const std::shared_ptr<SLIOperation::OperationState> &current_state,
            std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack);

        // 辅助函数：生成t-ancestry状态
        static void generateAncestryStates(
            const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
            const std::shared_ptr<SLIOperation::OperationState> &current_state,
            std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack);

        // 辅助函数：生成t-truncate状态
        static void generateTruncateStates(
            const std::vector<std::shared_ptr<SLINode>> &active_nodes,
            const std::shared_ptr<SLIOperation::OperationState> &current_state,
            std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack);
            
        static std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
        findPotentialAncestryPairs(const std::shared_ptr<SLITree> &tree);

        static std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
        findPotentialFactoringPairs(const std::shared_ptr<SLITree> &tree);

        static std::vector<std::shared_ptr<SLINode>> findPotentialTruncateNodes(
            const std::shared_ptr<SLITree> &tree);

    private:
        static bool checkEmptyClause(const SLITree &tree);

        static double calculateHeuristic(const Clause &kb_clause,
                                         const std::shared_ptr<SLINode> &tree_node,
                                         const Literal &resolving_literal);
        static void checkAndTruncateNode(const std::shared_ptr<SLINode> &node, SLITree &tree);

        // 用于存储已访问状态的哈希值
        static std::unordered_set<size_t> visited_states;

        static void printProofPath(std::shared_ptr<ProofState> state, KnowledgeBase &kb);
    };
}

#endif // LOGIC_SYSTEM_SLI_RESOLUTION_H