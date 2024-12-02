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

        // 主要接口
        static std::optional<Substitution> findMGU(const Literal& lit1, const Literal& lit2, KnowledgeBase& kb);
        static Literal applySubstitutionToLiteral(const Literal& lit, const Substitution& substitution, const KnowledgeBase& kb);
        static void printSubstitution(const Substitution& subst, const KnowledgeBase& kb);

    private:
        // 统一化辅助函数
        static bool unify(const std::vector<SymbolId>& args1, const std::vector<SymbolId>& args2, 
                         Substitution& subst, const KnowledgeBase& kb);
        static SymbolId applySubstitution(const SymbolId& termId, const Substitution& subst);
        static bool occursCheck(const SymbolId& varId, const SymbolId& termId, const Substitution& subst);

        // 变量重命名辅助函数
        static Literal standardizeVariables(const Literal& lit, KnowledgeBase& kb, int suffix);
        static SymbolId renameVariable(const SymbolId& varId, KnowledgeBase& kb, int suffix,
                                     std::unordered_map<SymbolId, SymbolId>& renamingMap);
    };
}

#endif // LOGIC_SYSTEM_UNIFIER_H