#include "Unifier.h"

namespace LogicSystem
{

    Literal Unifier::applySubstitutionToLiteral(const Literal &lit, const Substitution &substitution, const KnowledgeBase &kb)
    {
        std::vector<int> newArgs;
        for (int argId : lit.getArgumentIds())
        {
            newArgs.push_back(applySubstitution(argId, substitution));
        }
        return Literal(lit.getPredicateId(), newArgs, lit.isNegated());
    }
    std::optional<Unifier::Substitution> Unifier::findMGU(const Literal &lit1, const Literal &lit2, const KnowledgeBase &kb)
    {
        if (lit1.getPredicateId() != lit2.getPredicateId())
            return std::nullopt;

        Substitution subst;
        if (unify(lit1.getArgumentIds(), lit2.getArgumentIds(), subst, kb))
            return subst;
        return std::nullopt;
    }

    bool Unifier::unify(const std::vector<int> &args1, const std::vector<int> &args2, Substitution &subst, const KnowledgeBase &kb)
    {
        if (args1.size() != args2.size())
            return false;

        for (size_t i = 0; i < args1.size(); ++i)
        {
            int term1 = applySubstitution(args1[i], subst);
            int term2 = applySubstitution(args2[i], subst);

            if (term1 != term2)
            {
                if (kb.isVariable(term1))
                {
                    if (!occursCheck(term1, term2, subst))
                        subst[term1] = term2;
                    else
                        return false;
                }
                else if (kb.isVariable(term2))
                {
                    if (!occursCheck(term2, term1, subst))
                        subst[term2] = term1;
                    else
                        return false;
                }
                else
                {
                    return false;
                }
            }
        }
        return true;
    }

    int Unifier::applySubstitution(int termId, const Substitution &subst)
    {
        auto it = subst.find(termId);
        while (it != subst.end())
        {
            termId = it->second;
            it = subst.find(termId);
        }
        return termId;
    }

    bool Unifier::occursCheck(int varId, int termId, const Substitution &subst)
    {
        if (varId == termId)
            return true;

        auto it = subst.find(termId);
        if (it != subst.end())
            return occursCheck(varId, it->second, subst);

        return false;
    }
}