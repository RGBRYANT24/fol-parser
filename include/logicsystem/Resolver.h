#ifndef LOGIC_SYSTEM_RESOLVER_H
#define LOGIC_SYSTEM_RESOLVER_H

#include "KnowledgeBase.h"
#include <unordered_set>

namespace LogicSystem {
    class Resolver {
    public:
        bool isSatisfiable(const KnowledgeBase& kb);

    private:
        Clause* resolve(const Clause* c1, const Clause* c2);
        bool canResolve(const CNF* lit1, const CNF* lit2);
        void addResolvents(std::unordered_set<Clause*>& clauses, const Clause* c1, const Clause* c2);
        std::string clauseToString(const Clause* clause);
    };
}

#endif // LOGIC_SYSTEM_RESOLVER_H