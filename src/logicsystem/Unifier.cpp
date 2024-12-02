#include "Unifier.h"
#include <iostream>

namespace LogicSystem
{
    std::optional<Unifier::Substitution> Unifier::findMGU(const Literal &lit1, const Literal &lit2, KnowledgeBase &kb)
    {
        if (lit1.getPredicateId() != lit2.getPredicateId())
            return std::nullopt;

        // 对两个文字进行变量标准化，使用不同的后缀
        Literal standardizedLit1 = standardizeVariables(lit1, kb, 1);
        Literal standardizedLit2 = standardizeVariables(lit2, kb, 2);

        Substitution subst;
        if (unify(standardizedLit1.getArgumentIds(), standardizedLit2.getArgumentIds(), subst, kb))
            return subst;
        return std::nullopt;
    }

    Literal Unifier::standardizeVariables(const Literal& lit, KnowledgeBase& kb, int suffix)
    {
        std::vector<SymbolId> newArgs;
        std::unordered_map<SymbolId, SymbolId> renamingMap;

        for (const SymbolId& argId : lit.getArgumentIds())
        {
            if (kb.isVariable(argId))
            {
                newArgs.push_back(renameVariable(argId, kb, suffix, renamingMap));
            }
            else
            {
                newArgs.push_back(argId);
            }
        }
        return Literal(lit.getPredicateId(), newArgs, lit.isNegated());
    }

    SymbolId Unifier::renameVariable(const SymbolId& varId, KnowledgeBase& kb, int suffix,
                                    std::unordered_map<SymbolId, SymbolId>& renamingMap)
    {
        auto it = renamingMap.find(varId);
        if (it != renamingMap.end())
        {
            return it->second;
        }

        std::string originalName = kb.getSymbolName(varId);
        std::string newName = originalName + "_" + std::to_string(suffix);
        
        // 检查变量是否已存在
        auto existingId = kb.getVariableId(newName);
        if (existingId)
        {
            SymbolId newVarId = {SymbolType::VARIABLE, *existingId};
            renamingMap[varId] = newVarId;
            return newVarId;
        }

        // 插入新变量
        int newId = kb.insertVariable(newName);
        SymbolId newVarId = {SymbolType::VARIABLE, newId};
        renamingMap[varId] = newVarId;
        return newVarId;
    }

    Literal Unifier::applySubstitutionToLiteral(const Literal &lit, const Substitution &substitution, const KnowledgeBase &kb)
    {
        std::vector<SymbolId> newArgs;
        for (const SymbolId& argId : lit.getArgumentIds())
        {
            newArgs.push_back(applySubstitution(argId, substitution));
        }
        return Literal(lit.getPredicateId(), newArgs, lit.isNegated());
    }

    bool Unifier::unify(const std::vector<SymbolId>& args1, const std::vector<SymbolId>& args2, 
                        Substitution& subst, const KnowledgeBase& kb)
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

    SymbolId Unifier::applySubstitution(const SymbolId& termId, const Substitution &subst)
    {
        SymbolId currentId = termId;
        auto it = subst.find(currentId);
        while (it != subst.end())
        {
            currentId = it->second;
            it = subst.find(currentId);
        }
        return currentId;
    }

    bool Unifier::occursCheck(const SymbolId& varId, const SymbolId& termId, const Substitution &subst)
    {
        if (varId == termId)
            return true;

        auto it = subst.find(termId);
        if (it != subst.end())
            return occursCheck(varId, it->second, subst);

        return false;
    }

    void Unifier::printSubstitution(const Substitution& subst, const KnowledgeBase& kb)
    {
        if (subst.empty())
        {
            std::cout << "Empty substitution (identity mapping)" << std::endl;
            return;
        }

        std::cout << "Substitution:" << std::endl;
        for (const auto& [var, term] : subst)
        {
            std::cout << kb.getSymbolName(var) << " -> " << kb.getSymbolName(term) << std::endl;
        }
    }
}