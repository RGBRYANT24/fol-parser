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
        using Substitution = std::unordered_map<int, int>;

        static std::optional<Substitution> findMGU(const Literal& lit1, const Literal& lit2, const KnowledgeBase& kb);
        static Literal applySubstitutionToLiteral(const Literal& lit, const Substitution& substitution, const KnowledgeBase& kb);

    private:
        static bool unify(const std::vector<int>& args1, const std::vector<int>& args2, Substitution& subst, const KnowledgeBase& kb);
        static int applySubstitution(int termId, const Substitution& subst);
        static bool occursCheck(int varId, int termId, const Substitution& subst); //进行出现检查
    };
}

#endif // LOGIC_SYSTEM_UNIFIER_H