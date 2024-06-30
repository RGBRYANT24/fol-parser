#include "Resolver.h"
#include <queue>
#include <algorithm>

namespace LogicSystem {

    bool Resolver::isSatisfiable(const KnowledgeBase& kb) {
        std::unordered_set<Clause*> clauses;
        std::unordered_set<std::string> seenClauses;
        std::queue<Clause*> newClauses;

        // Initialize with clauses from the knowledge base
        for (const auto& clause : kb.getClauses()) {
            clauses.insert(new Clause(*clause));
            newClauses.push(new Clause(*clause));
        }

        while (!newClauses.empty()) {
            Clause* clause = newClauses.front();
            newClauses.pop();

            // Check for empty clause
            if (clause->size() == 0) {
                // Clean up
                for (auto c : clauses) {
                    delete c;
                }
                return false;  // Unsatisfiable
            }

            for (const auto& literal : clause->getAllLiterals()) {
                std::vector<Clause*> resolvableClauses = kb.getClausesWithPredicate(literal->getPredicateName());
                for (const auto& otherClause : resolvableClauses) {
                    if (otherClause != clause) {
                        std::unordered_set<Clause*> resolvents;
                        addResolvents(resolvents, clause, otherClause);
                        for (auto resolvent : resolvents) {
                            std::string resolventStr = clauseToString(resolvent);
                            if (seenClauses.find(resolventStr) == seenClauses.end()) {
                                seenClauses.insert(resolventStr);
                                clauses.insert(resolvent);
                                newClauses.push(resolvent);
                            } else {
                                delete resolvent;
                            }
                        }
                    }
                }
            }
        }

        // Clean up
        for (auto c : clauses) {
            delete c;
        }

        return true;  // Satisfiable
    }

    Clause* Resolver::resolve(const Clause* c1, const Clause* c2) {
        Clause* result = new Clause();
        
        // Add all literals from c1 and c2 to result, except the ones that resolve
        for (const auto& lit : c1->getAllLiterals()) {
            if (!canResolve(lit, c2->getLiteral(lit->getPredicateName()))) {
                result->addLiteral(new CNF(*lit));
            }
        }
        for (const auto& lit : c2->getAllLiterals()) {
            if (!canResolve(lit, c1->getLiteral(lit->getPredicateName()))) {
                result->addLiteral(new CNF(*lit));
            }
        }

        return result;
    }

    bool Resolver::canResolve(const CNF* lit1, const CNF* lit2) {
        return lit1 && lit2 && lit1->getPredicateName() == lit2->getPredicateName() && lit1->isNegated() != lit2->isNegated();
    }

    void Resolver::addResolvents(std::unordered_set<Clause*>& clauses, const Clause* c1, const Clause* c2) {
        for (const auto& lit1 : c1->getAllLiterals()) {
            const CNF* lit2 = c2->getLiteral(lit1->getPredicateName());
            if (canResolve(lit1, lit2)) {
                Clause* resolvent = resolve(c1, c2);
                if (resolvent->size() > 0) {
                    clauses.insert(resolvent);
                } else {
                    clauses.insert(resolvent);  // Empty clause
                    return;
                }
            }
        }
    }

    std::string Resolver::clauseToString(const Clause* clause) {
        std::vector<std::string> literalStrings;
        for (const auto& literal : clause->getAllLiterals()) {
            //literalStrings.push_back(literal->toString());
            literalStrings.push_back(literal->getPredicateName());
        }
        std::sort(literalStrings.begin(), literalStrings.end());
        std::string result;
        for (const auto& str : literalStrings) {
            result += str + " ";
        }
        return result;
    }
}