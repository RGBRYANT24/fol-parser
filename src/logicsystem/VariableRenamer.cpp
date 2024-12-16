// VariableRenamer.cpp
#include "VariableRenamer.h"

namespace LogicSystem
{

    Clause VariableRenamer::renameClauseVariables(const Clause &clause, const SLITree &tree, KnowledgeBase &kb)
    {
        // 收集SLITree中的所有变量
        std::set<SymbolId> treeVariables = collectTreeVariables(tree, kb);

        // 收集输入子句中的所有变量
        std::set<SymbolId> clauseVariables = collectClauseVariables(clause, kb);

        // 创建变量映射
        std::unordered_map<SymbolId, SymbolId> variableMap;

        // 为每个子句中的变量创建新名称
        for (const SymbolId &varId : clauseVariables)
        {
            if (kb.isVariable(varId))
            {
                std::string oldVarName = kb.getSymbolName(varId);
                std::string newVarName = generateNewVariableName(oldVarName, treeVariables, kb);
                SymbolId newVarId = kb.addVariable(newVarName);
                variableMap[varId] = newVarId;
                treeVariables.insert(newVarId); // 将新变量添加到已使用集合中
            }
        }

        // 创建新的子句并替换变量
        Clause newClause;
        for (const Literal &lit : clause.getLiterals())
        {
            std::vector<SymbolId> newArgs;
            for (const SymbolId &argId : lit.getArgumentIds())
            {
                if (kb.isVariable(argId) && variableMap.find(argId) != variableMap.end())
                {
                    newArgs.push_back(variableMap[argId]);
                }
                else
                {
                    newArgs.push_back(argId);
                }
            }
            newClause.addLiteral(Literal(lit.getPredicateId(), newArgs, lit.isNegated()));
        }

        return newClause;
    }

    std::set<SymbolId> VariableRenamer::collectTreeVariables(const SLITree &tree, const KnowledgeBase &kb)
    {
        std::set<SymbolId> variables;

        // 遍历树的所有深度
        for (const auto &level : tree.getDepthMap())
        {
            for (const auto &node : level)
            {
                if (node)
                {
                    // 收集节点文字中的变量
                    for (const SymbolId &argId : node->literal.getArgumentIds())
                    {
                        if (kb.isVariable(argId))
                        {
                            variables.insert(argId);
                        }
                    }
                }
            }
        }
        return variables;
    }

    std::set<SymbolId> VariableRenamer::collectClauseVariables(const Clause &clause, const KnowledgeBase &kb)
    {
        std::set<SymbolId> variables;

        for (const Literal &lit : clause.getLiterals())
        {
            for (const SymbolId &argId : lit.getArgumentIds())
            {
                if (kb.isVariable(argId))
                {
                    variables.insert(argId);
                }
            }
        }
        return variables;
    }

    std::string VariableRenamer::generateNewVariableName(const std::string &oldName,
                                                         const std::set<SymbolId> &usedVariables,
                                                         const KnowledgeBase &kb)
    {
        std::string baseName = oldName;
        std::string newName = baseName;
        int suffix = 1;

        // 当新名称已存在时，增加后缀数字
        while (true)
        {
            auto symbolId = kb.getSymbolId(newName);
            if (!symbolId || usedVariables.find(*symbolId) == usedVariables.end())
            {
                break;
            }
            newName = baseName + std::to_string(suffix++);
        }

        return newName;
    }

} // namespace LogicSystem