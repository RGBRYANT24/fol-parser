// SLIResolution.h
#ifndef LOGIC_SYSTEM_SLI_RESOLUTION_H
#define LOGIC_SYSTEM_SLI_RESOLUTION_H

#include "SLITree.h"
#include "KnowledgeBase.h"
#include "SearchStrategy.h"
#include <optional>
#include <vector>

namespace LogicSystem
{
    struct ProofState
    {
        int state_id;
        std::shared_ptr<ProofState> parent;
        SLIResolutionPair resolution_pair;
        std::unique_ptr<SLITree> tree;

        ProofState(const SLIResolutionPair &pair,
                   std::unique_ptr<SLITree> t,
                   std::shared_ptr<ProofState> p = nullptr)
            : resolution_pair(pair), tree(std::move(t)), parent(p)
        {
            static int next_id = 0;
            state_id = next_id++;
        }
    };

    class SLIResolution
    {
    public:
        static bool prove(KnowledgeBase &kb, const Clause &goal, SearchStrategy &strategy);

    private:
        static std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>> findPotentialFactoringPairs(
            const std::vector<std::shared_ptr<SLINode>> &new_nodes,
            const std::vector<std::vector<std::shared_ptr<SLINode>>> &depth_map,
            KnowledgeBase &kb);

        static std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
        findPotentialAncestryPairs(const std::vector<std::shared_ptr<SLINode>> &new_nodes,
                                   KnowledgeBase &kb);

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