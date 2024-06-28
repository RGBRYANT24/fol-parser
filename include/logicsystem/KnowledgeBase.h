#ifndef LOGIC_SYSTEM_KNOWLEDGE_BASE_H
#define LOGIC_SYSTEM_KNOWLEDGE_BASE_H

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include "Clause.h"

namespace LogicSystem {
    class KnowledgeBase {
    public:
        void addClause(Clause* clause);
        void removeClause(Clause* clause);
        std::vector<Clause*> getClauses() const;
        std::vector<Clause*> getClausesWithPredicate(const std::string& predicateName) const;
        void print() const;
        size_t size() const;
        ~KnowledgeBase();

    private:
        std::unordered_map<std::string, std::unordered_set<Clause*>> predicateIndex;
        std::vector<Clause*> clauseOrder;  // 仅用于保持插入顺序

        void indexClause(Clause* clause);
        void deindexClause(Clause* clause);
    };
}

#endif // LOGIC_SYSTEM_KNOWLEDGE_BASE_H