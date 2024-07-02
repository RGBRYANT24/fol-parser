#include "Fact.h"
#include "KnowledgeBase.h"

namespace LogicSystem
{
    Fact::Fact(int predId, const std::vector<int>& argIds)
        : predicateId(predId), argumentIds(argIds) {}

    int Fact::getPredicateId() const {
        return predicateId;
    }

    const std::vector<int>& Fact::getArgumentIds() const {
        return argumentIds;
    }

    bool Fact::operator==(const Fact& other) const {
        return predicateId == other.predicateId &&
               argumentIds == other.argumentIds;
    }

    std::string Fact::toString(const KnowledgeBase& kb) const {
        std::string result = kb.getPredicateName(predicateId) + "(";
        for (size_t i = 0; i < argumentIds.size(); ++i) {
            result += kb.getConstantName(argumentIds[i]);
            if (i < argumentIds.size() - 1) {
                result += ", ";
            }
        }
        result += ")";
        return result;
    }
}