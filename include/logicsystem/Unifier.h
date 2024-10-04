#ifndef LOGIC_SYSTEM_UNIFIER_H
#define LOGIC_SYSTEM_UNIFIER_H

#include <optional>
#include <unordered_map>
#include <vector>
#include "KnowledgeBase.h"
#include "Literal.h"

namespace LogicSystem
{
    class Unifier
    {
    public:
        using Substitution = std::unordered_map<SymbolId, SymbolId>;

        static std::optional<Substitution> findMGU(const Literal& lit1, const Literal& lit2, const KnowledgeBase& kb);
        static Literal applySubstitutionToLiteral(const Literal& lit, const Substitution& substitution, const KnowledgeBase& kb);

    private:
        static bool unify(const std::vector<SymbolId>& args1, const std::vector<SymbolId>& args2, Substitution& subst, const KnowledgeBase& kb);
        static SymbolId  applySubstitution(const SymbolId& termId, const Substitution& subst);
        static bool occursCheck(const SymbolId& varId, const SymbolId& termId, const Substitution& subst); //进行出现检查
    };
}

#endif // LOGIC_SYSTEM_UNIFIER_H