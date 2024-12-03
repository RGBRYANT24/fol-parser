#include "Unifier.h"
#include <iostream>
#include <set>

namespace LogicSystem
{
    // 在cpp文件中定义静态成员
    int Unifier::globalRenamingCounter = 0;
    std::optional<Unifier::Substitution> Unifier::findMGU(const Literal &lit1, const Literal &lit2, KnowledgeBase &kb)
    {
        if (lit1.getPredicateId() != lit2.getPredicateId())
            return std::nullopt;

        // 创建变量重命名映射
        std::unordered_map<SymbolId, SymbolId> renamingMap1, renamingMap2;

        // 查找已存在的变量名，避免冲突
        std::set<std::string> existingVarNames;
        for (const auto &arg : lit1.getArgumentIds())
        {
            if (kb.isVariable(arg))
            {
                existingVarNames.insert(kb.getSymbolName(arg));
            }
        }
        for (const auto &arg : lit2.getArgumentIds())
        {
            if (kb.isVariable(arg))
            {
                existingVarNames.insert(kb.getSymbolName(arg));
            }
        }

        Literal standardizedLit1 = standardizeVariables(lit1, kb, 1, renamingMap1);
        Literal standardizedLit2 = standardizeVariables(lit2, kb, 2, renamingMap2);

        Substitution subst;
        if (!unify(standardizedLit1.getArgumentIds(), standardizedLit2.getArgumentIds(), subst, kb))
            return std::nullopt;

        // 构建最终替换
        Substitution finalSubst;
        for (const auto &[var, term] : subst)
        {
            // 查找原始变量
            SymbolId originalVar = var;
            bool found = false;

            // 在renamingMap1中查找
            for (const auto &[orig, renamed] : renamingMap1)
            {
                if (renamed == var)
                {
                    originalVar = orig;
                    found = true;
                    break;
                }
            }

            // 在renamingMap2中查找
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

            // 添加到最终替换中，避免自反替换
            if (originalVar != term)
            {
                finalSubst[originalVar] = term;
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
        std::string baseName = normalizeVariableName(originalName);

        // 使用全局计数器生成唯一的变量名
        std::string newName;
        SymbolId newVarId;
        bool nameExists;

        do
        {
            globalRenamingCounter++;
            newName = baseName + "_" + std::to_string(globalRenamingCounter);
            auto existingId = kb.getVariableId(newName);
            nameExists = existingId.has_value();

            if (!nameExists)
            {
                int newId = kb.insertVariable(newName);
                newVarId = {SymbolType::VARIABLE, newId};
            }
        } while (nameExists);

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