// Resolution.h
#ifndef LOGIC_SYSTEM_RESOLUTION_H
#define LOGIC_SYSTEM_RESOLUTION_H

#include "KnowledgeBase.h"
#include "ResolutionPair.h"
#include "SearchStrategy.h"
#include <queue>
#include <optional>
#include <unordered_set>
#include <stack>

namespace LogicSystem {
    class Resolution {
    public:
        static bool prove(const KnowledgeBase &kb, const Clause &goal, SearchStrategy& strategy, int addClauseSize = 1); //启发式方法
        static std::optional<Clause> testResolve(const Clause& c1, const Clause& c2, int l1, int l2, const KnowledgeBase& kb) {return resolve(c1, c2,l1,l2,kb);}
    private:
        static double calculateHeuristic(const Clause& c1, const Clause& c2, int l1, int l2);
        static std::optional<Clause> resolve(const Clause& c1, const Clause& c2, int l1, int l2, const KnowledgeBase& kb);
        static bool isComplementary(const Literal& lit1, const Literal& lit2);
        static bool dfsHelper(KnowledgeBase &kb, const Clause &goal, std::unordered_set<std::string> &seenClauses);
    };
}

#endif // LOGIC_SYSTEM_RESOLUTION_H