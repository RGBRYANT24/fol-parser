#include "Clause.h"
#include "KnowledgeBase.h"

namespace LogicSystem
{
    void Clause::addLiteral(const Literal& lit) {
        literals.push_back(lit);
    }

    const std::vector<Literal>& Clause::getLiterals() const {
        return literals;
    }

    bool Clause::isEmpty() const {
        return literals.empty();
    }

    std::string Clause::toString(const KnowledgeBase& kb) const {
        std::string result;
        for (size_t i = 0; i < literals.size(); ++i) {
            result += literals[i].toString(kb);
            if (i < literals.size() - 1) {
                result += " ∨ ";
            }
        }
        return result;
    }
}