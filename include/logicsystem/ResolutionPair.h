// ResolutionPair.h
#ifndef LOGIC_SYSTEM_RESOLUTION_PAIR_H
#define LOGIC_SYSTEM_RESOLUTION_PAIR_H

#include "Clause.h"

namespace LogicSystem {
    struct ResolutionPair {
        const Clause* clause1;
        const Clause* clause2;
        int literal1Index;
        int literal2Index;
        double heuristicScore;

        ResolutionPair(const Clause* c1, const Clause* c2, int l1, int l2, double score)
            : clause1(c1), clause2(c2), literal1Index(l1), literal2Index(l2), heuristicScore(score) {}

        bool operator<(const ResolutionPair& other) const {
            return heuristicScore > other.heuristicScore;
        }
    };
}

#endif // LOGIC_SYSTEM_RESOLUTION_PAIR_H