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
        static std::optional<Substitution> findMGU(const Literal &lit1, const Literal &lit2, KnowledgeBase &kb);
        static Literal applySubstitutionToLiteral(const Literal &lit, const Substitution &substitution, const KnowledgeBase &kb);
        static void printSubstitution(const Substitution &subst, const KnowledgeBase &kb);

    private:
        static int globalRenamingCounter; // 添加静态计数器

        // 添加变量名规范化函数
        static std::string normalizeVariableName(const std::string &name)
        {
            size_t pos = name.find_last_of('_');
            if (pos == std::string::npos)
                return name;

            // 检查下划线后是否都是数字
            std::string suffix = name.substr(pos + 1);
            if (std::all_of(suffix.begin(), suffix.end(), ::isdigit))
            {
                return name.substr(0, pos);
            }
            return name;
        }
        // 统一化辅助函数
        static bool unify(const std::vector<SymbolId> &args1, const std::vector<SymbolId> &args2,
                          Substitution &subst, const KnowledgeBase &kb);
        static SymbolId applySubstitution(const SymbolId &termId, const Substitution &subst);
        static bool occursCheck(const SymbolId &varId, const SymbolId &termId, const Substitution &subst);

        // 变量重命名辅助函数
        // 修改standardizeVariables的签名，添加renamingMap参数
        static Literal standardizeVariables(const Literal &lit, KnowledgeBase &kb, int suffix,
                                            std::unordered_map<SymbolId, SymbolId> &renamingMap);
        static SymbolId renameVariable(const SymbolId &varId, KnowledgeBase &kb, int suffix,
                                       std::unordered_map<SymbolId, SymbolId> &renamingMap);
    };
}

#endif // LOGIC_SYSTEM_UNIFIER_H