// Resolution.h
#ifndef LOGIC_SYSTEM_RESOLUTION_H
#define LOGIC_SYSTEM_RESOLUTION_H

#include "KnowledgeBase.h"
#include "ResolutionPair.h"
#include <queue>
#include <optional>

namespace LogicSystem {
    class Resolution {
    public:
        static bool prove(const KnowledgeBase& kb, const Clause& goal);

    private:
        static double calculateHeuristic(const Clause& c1, const Clause& c2, int l1, int l2);
        static std::optional<Clause> resolve(const Clause& c1, const Clause& c2, int l1, int l2, const KnowledgeBase& kb);
        static bool isComplementary(const Literal& lit1, const Literal& lit2);
    };
}

#endif // LOGIC_SYSTEM_RESOLUTION_H