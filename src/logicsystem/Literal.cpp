#include "Literal.h"
#include "KnowledgeBase.h"

namespace LogicSystem
{
    Literal::Literal(int predId, const std::vector<SymbolId>& argIds, bool negated)
        : predicateId(predId), argumentIds(argIds), negated(negated) {}

    std::string Literal::toString(const KnowledgeBase& kb) const {
        std::string result = negated ? "¬" : "";
        result += kb.getPredicateName(predicateId) + "(";
        for (size_t i = 0; i < argumentIds.size(); ++i) {
            const SymbolId& symbolId = argumentIds[i];
            result += kb.getSymbolName(symbolId);
            if (i < argumentIds.size() - 1) {
                result += ", ";
            }
        }
        result += ")";
        return result;
    }

    int Literal::getPredicateId() const {
        return predicateId;
    }

    const std::vector<SymbolId>& Literal::getArgumentIds() const {
        return argumentIds;
    }

    bool Literal::isNegated() const {
        return negated;
    }
}