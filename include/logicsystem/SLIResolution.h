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
    class SLIResolution
    {
    public:
        static bool prove(const KnowledgeBase &kb, const Clause &goal, SearchStrategy &strategy);

    private:
        static std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>> findPotentialFactoringPairs(
            const std::vector<std::shared_ptr<SLINode>> &new_nodes,
            const std::vector<std::vector<std::shared_ptr<SLINode>>> &depth_map,
            const KnowledgeBase &kb);

        static std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
        findPotentialAncestryPairs(const std::vector<std::shared_ptr<SLINode>> &new_nodes,
                                   const KnowledgeBase &kb);

        static bool checkEmptyClause(const SLITree &tree);

        static double calculateHeuristic(const Clause &kb_clause,
                                         const std::shared_ptr<SLINode> &tree_node,
                                         const Literal &resolving_literal);
        static void checkAndTruncateNode(const std::shared_ptr<SLINode> &node, SLITree &tree);
        
        // 用于存储已访问状态的哈希值
        static std::unordered_set<size_t> visited_states;
    };
}

#endif // LOGIC_SYSTEM_SLI_RESOLUTION_H