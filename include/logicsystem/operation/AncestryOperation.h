// AncestryOperation.h
#ifndef LOGIC_SYSTEM_ANCESTRY_OPERATION_H
#define LOGIC_SYSTEM_ANCESTRY_OPERATION_H

#include "Operation.h"
#include "SLINode.h"
#include "Unifier.h"
#include <memory>
#include <vector>

namespace LogicSystem
{
    class AncestryOperation : public Operation
    {
    public:
        AncestryOperation(
            std::shared_ptr<SLINode> upper_node,
            std::shared_ptr<SLINode> lower_node,
            const std::vector<std::pair<std::shared_ptr<SLINode>, Literal>>& previous_literals,
            const std::vector<std::pair<std::shared_ptr<SLINode>, Unifier::Substitution>>& previous_substitutions,
            const Unifier::Substitution& applied_mgu)
            : upper_node_(upper_node)
            , lower_node_(lower_node)
            , previous_literals_(previous_literals)
            , previous_substitutions_(previous_substitutions)
            , applied_mgu_(applied_mgu) {}

        void undo() override
        {
            // 恢复所有节点的原始状态
            for (const auto& [node, lit] : previous_literals_) {
                if (node) {
                    node->literal = lit;
                }
            }
            
            for (const auto& [node, subst] : previous_substitutions_) {
                if (node) {
                    node->substitution = subst;
                }
            }

            // 恢复lower_node的活动状态
            lower_node_->is_active = true;
            lower_node_->rule_applied.clear();
            
            // 清除upper_node的规则标记
            upper_node_->rule_applied.clear();
        }

    private:
        std::shared_ptr<SLINode> upper_node_;
        std::shared_ptr<SLINode> lower_node_;
        std::vector<std::pair<std::shared_ptr<SLINode>, Literal>> previous_literals_;
        std::vector<std::pair<std::shared_ptr<SLINode>, Unifier::Substitution>> previous_substitutions_;
        Unifier::Substitution applied_mgu_;
    };
}

#endif // LOGIC_SYSTEM_ANCESTRY_OPERATION_H