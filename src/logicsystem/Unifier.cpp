#include "Unifier.h"
#include <iostream>
#include <set>

namespace LogicSystem
{
    std::optional<Unifier::Substitution> Unifier::findMGU(const Literal &lit1, const Literal &lit2, KnowledgeBase &kb)
    {
        // 检查谓词是否相同
        if (lit1.getPredicateId() != lit2.getPredicateId())
            return std::nullopt;

        Substitution subst;
        // 直接对原始文字进行统一化
        if (!unify(lit1.getArgumentIds(), lit2.getArgumentIds(), subst, kb))
            return std::nullopt;

        return subst;
    }

    Literal Unifier::applySubstitutionToLiteral(const Literal &lit, const Substitution &substitution, const KnowledgeBase &kb)
    {
        std::vector<SymbolId> newArgs;
        // std::cout << "Try apply Subsitution to " << lit.toString(kb) << std::endl;
        // if (substitution.empty())
        // {
        //     std::cout << " applySubstitutionToLiteral Found subsitution empty " << std::endl;
        // }
        // else
        // {
        //     std::cout << "Subsitution size " << substitution.size() << std::endl;
        //     std::cout << "Subsitution:" << std::endl;
        //     printSubstitution(substitution, kb);
        // }
        for (const SymbolId &argId : lit.getArgumentIds())
        {
            auto it = substitution.find(argId);
            // if (it != substitution.end())
            // {
            //     std::cout << "apply Subsitution " << kb.getSymbolName(argId) << " subsitution: "
            //               << kb.getSymbolName(it->second) << std::endl;
            // }
            // else
            // {
            //     std::cout << "Found No Subsitution of " << kb.getSymbolName(argId) << std::endl;
            // }
            newArgs.push_back(applySubstitution(argId, substitution));
            // std::cout << newArgs.back().id << " ";
        }
        //std::cout << std::endl;
        return Literal(lit.getPredicateId(), newArgs, lit.isNegated());
    }

    bool Unifier::unify(const std::vector<SymbolId> &args1, const std::vector<SymbolId> &args2,
                        Substitution &subst, const KnowledgeBase &kb)
    {
        if (args1.size() != args2.size())
            return false;

        for (size_t i = 0; i < args1.size(); ++i)
        {
            SymbolId term1 = applySubstitution(args1[i], subst);
            SymbolId term2 = applySubstitution(args2[i], subst);

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

    SymbolId Unifier::applySubstitution(const SymbolId &termId, const Substitution &subst)
    {
        SymbolId currentId = termId;
        // std::cout << "applySubstitution termId " << termId.id << std::endl;
        auto it = subst.find(currentId);
        while (it != subst.end())
        {
            currentId = it->second;
            it = subst.find(currentId);
            // std::cout << "current id " << currentId.id << " it " << it->second.id << std::endl;
        }
        return currentId;
    }

    bool Unifier::occursCheck(const SymbolId &varId, const SymbolId &termId, const Substitution &subst)
    {
        if (varId == termId)
            return true;

        auto it = subst.find(termId);
        if (it != subst.end())
            return occursCheck(varId, it->second, subst);

        return false;
    }

    void Unifier::printSubstitution(const Substitution &subst, const KnowledgeBase &kb)
    {
        if (subst.empty())
        {
            std::cout << "Empty substitution (identity mapping)" << std::endl;
            return;
        }

        std::cout << "Substitution:" << std::endl;
        for (const auto &[var, term] : subst)
        {
            std::cout << kb.getSymbolName(var) << " -> " << kb.getSymbolName(term) << std::endl;
        }
    }
}