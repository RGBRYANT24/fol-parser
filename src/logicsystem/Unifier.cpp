#include "Unifier.h"
#include <iostream>

namespace LogicSystem
{
    std::optional<Unifier::Substitution> Unifier::findMGU(const Literal &lit1, const Literal &lit2, KnowledgeBase &kb)
    {
        if (lit1.getPredicateId() != lit2.getPredicateId())
            return std::nullopt;

        // 保存原始变量到重命名变量的映射
        std::unordered_map<SymbolId, SymbolId> renamingMap1, renamingMap2;

        // 对两个文字进行变量标准化，使用不同的后缀
        Literal standardizedLit1 = standardizeVariables(lit1, kb, 1, renamingMap1);
        Literal standardizedLit2 = standardizeVariables(lit2, kb, 2, renamingMap2);

        Substitution subst;
        if (!unify(standardizedLit1.getArgumentIds(), standardizedLit2.getArgumentIds(), subst, kb))
            return std::nullopt;

        // 转换substitution，将重命名的变量映射回原始变量
        Substitution finalSubst;
        for (const auto &[var, term] : subst)
        {
            // 查找var对应的原始变量
            SymbolId originalVar;
            bool found = false;

            // 检查renamingMap1
            for (const auto &[orig, renamed] : renamingMap1)
            {
                if (renamed == var)
                {
                    originalVar = orig;
                    found = true;
                    break;
                }
            }

            // 检查renamingMap2
            if (!found)
            {
                for (const auto &[orig, renamed] : renamingMap2)
                {
                    if (renamed == var)
                    {
                        originalVar = orig;
                        found = true;
                        break;
                    }
                }
            }

            // 如果找到原始变量，添加到最终替换中
            if (found)
            {
                finalSubst[originalVar] = term;
            }
            else
            {
                // 如果不是重命名的变量，直接添加
                finalSubst[var] = term;
            }
        }

        return finalSubst;
    }

    Literal Unifier::standardizeVariables(const Literal &lit, KnowledgeBase &kb, int suffix,
                                          std::unordered_map<SymbolId, SymbolId> &renamingMap)
    {
        std::vector<SymbolId> newArgs;

        for (const SymbolId &argId : lit.getArgumentIds())
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

    SymbolId Unifier::renameVariable(const SymbolId &varId, KnowledgeBase &kb, int suffix,
                                     std::unordered_map<SymbolId, SymbolId> &renamingMap)
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
        std::cout << "Try apply Subsitution to " << lit.toString(kb) << std::endl;
        if (substitution.empty())
        {
            std::cout << " applySubstitutionToLiteral Found subsitution empty " << std::endl;
        }
        else
        {
            std::cout << "Subsitution size " << substitution.size() << std::endl;
            std::cout << "Subsitution:" << std::endl;
            printSubstitution(substitution, kb);
        }
        for (const SymbolId &argId : lit.getArgumentIds())
        {
            auto it = substitution.find(argId);
            if (it != substitution.end())
            {
                std::cout << "apply Subsitution " << kb.getSymbolName(argId) << " subsitution: "
                          << kb.getSymbolName(it->second) << std::endl;
            }
            else
            {
                std::cout << "Found No Subsitution of " << kb.getSymbolName(argId) << std::endl;
            }
            newArgs.push_back(applySubstitution(argId, substitution));
            std::cout << newArgs.back().id << " ";
        }
        std::cout << std::endl;
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