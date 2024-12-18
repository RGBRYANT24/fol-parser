// VariableRenamer.h
#ifndef LOGIC_SYSTEM_VARIABLE_RENAMER_H
#define LOGIC_SYSTEM_VARIABLE_RENAMER_H

#include "SLITree.h"
#include "Clause.h"
#include <set>
#include <unordered_map>

namespace LogicSystem {
class SLITree;  // 前向声明

class VariableRenamer {
public:
    static Clause renameClauseVariables(const Clause& clause, const SLITree& tree, KnowledgeBase& kb);
    static std::set<SymbolId> collectTreeVariables(const SLITree& tree, const KnowledgeBase& kb);
    static std::set<SymbolId> collectClauseVariables(const Clause& clause, const KnowledgeBase& kb);
    static std::string generateNewVariableName(const std::string& oldName, 
                                             const std::set<SymbolId>& usedVariables, 
                                             const KnowledgeBase& kb);
    
};

} // namespace LogicSystem

#endif // LOGIC_SYSTEM_VARIABLE_RENAMER_H