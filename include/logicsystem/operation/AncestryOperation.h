#ifndef LOGIC_SYSTEM_ANCESTRY_OPERATION_H
#define LOGIC_SYSTEM_ANCESTRY_OPERATION_H

#include "Operation.h"
#include "SLINode.h"
#include "Unifier.h"
#include <memory>

namespace LogicSystem
{
    class AncestryOperation : public Operation
    {
    public:
        AncestryOperation(
            std::shared_ptr<SLINode> upper_node,
            std::shared_ptr<SLINode> lower_node,
            const Literal& previous_lit,
            const Unifier::Substitution& previous_substitution,
            const Unifier::Substitution& applied_mgu)
            : upper_node_(upper_node)
            , lower_node_(lower_node)
            , previous_lit_(previous_lit)
            , previous_substitution_(previous_substitution)
            , applied_mgu_(applied_mgu) {}

        void undo() override
        {
            // 恢复upper_node的原始状态
            upper_node_->literal = previous_lit_;
            upper_node_->substitution = previous_substitution_;
            
            // 恢复lower_node的活动状态
            lower_node_->is_active = true;
            lower_node_->rule_applied.clear();
        }

    private:
        std::shared_ptr<SLINode> upper_node_;
        std::shared_ptr<SLINode> lower_node_;
        Literal previous_lit_;
        Unifier::Substitution previous_substitution_;
        Unifier::Substitution applied_mgu_;
    };
}

#endif // LOGIC_SYSTEM_ANCESTRY_OPERATION_H