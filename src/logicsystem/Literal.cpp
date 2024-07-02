#include "Literal.h"
#include "KnowledgeBase.h"

namespace LogicSystem
{
    Literal::Literal(int predId, const std::vector<int>& argIds, bool negated)
        : predicateId(predId), argumentIds(argIds), negated(negated) {}

    std::string Literal::toString(const KnowledgeBase& kb) const {
        std::string result = negated ? "¬" : "";
        result += kb.getPredicateName(predicateId) + "(";
        for (size_t i = 0; i < argumentIds.size(); ++i) {
            int argId = argumentIds[i];
            if (kb.isVariable(argId)) {
                result += kb.getVariableName(argId);
            } else {
                result += kb.getConstantName(argId);
            }
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

    const std::vector<int>& Literal::getArgumentIds() const {
        return argumentIds;
    }

    bool Literal::isNegated() const {
        return negated;
    }
}