// DataCollector.h
#ifndef DATA_COLLECTOR_H
#define DATA_COLLECTOR_H

#include <nlohmann/json.hpp> // 使用nlohmann/json库处理JSON
#include "SLITree.h"
#include "SLIOperation.h"
#include <fstream> // 用于 std::ofstream
// #include <json>     // 用于 json 操作

using json = nlohmann::json;

namespace LogicSystem
{

    class DataCollector
    {
    public:
        // 收集单个SLI节点的数据
        static json collectNodeData(const std::shared_ptr<SLINode> &node, const KnowledgeBase &kb)
        {
            json node_data;
            node_data["node_id"] = node->node_id;
            node_data["node_lit"] = node->literal.toString(kb);
            node_data["depth"] = node->depth;
            node_data["is_A_literal"] = node->is_A_literal;
            node_data["is_active"] = node->is_active;

            // 收集文字信息
            json literal_data;
            literal_data["predicate"] = kb.getPredicateName(node->literal.getPredicateId());
            literal_data["is_negated"] = node->literal.isNegated();

            std::vector<std::string> args;
            for (const auto &arg : node->literal.getArgumentIds())
            {
                args.push_back(kb.getSymbolName(arg));
            }
            literal_data["arguments"] = args;

            node_data["literal"] = literal_data;

            // 收集替换信息
            json subst;
            for (const auto &[var, val] : node->substitution)
            {
                subst[kb.getSymbolName(var)] = kb.getSymbolName(val);
            }
            node_data["substitution"] = subst;

            // 收集父子节点关系
            node_data["parent_id"] = node->parent.lock() ? node->parent.lock()->node_id : -1;
            std::vector<int> child_ids;
            for (const auto &child : node->children)
            {
                child_ids.push_back(child->node_id);
            }
            node_data["children_ids"] = child_ids;

            return node_data;
        }

        // 收集树状态数据
        static json collectTreeState(const SLITree &tree, const KnowledgeBase &kb)
        {
            json tree_state;

            // 收集所有节点信息
            std::vector<json> nodes;
            for (const auto &level : tree.getDepthMap())
            {
                for (const auto &node : level)
                {
                    if (node)
                    {
                        nodes.push_back(collectNodeData(node, kb));
                    }
                }
            }
            tree_state["nodes"] = nodes;

            // 收集深度映射
            std::vector<std::vector<int>> depth_map;
            for (const auto &level : tree.getDepthMap())
            {
                std::vector<int> level_ids;
                for (const auto &node : level)
                {
                    if (node)
                    {
                        level_ids.push_back(node->node_id);
                    }
                }
                depth_map.push_back(level_ids);
            }
            tree_state["depth_map"] = depth_map;

            return tree_state;
        }

        // 收集操作数据
        static json collectOperationData(const SLIOperation::OperationState &op_state, const KnowledgeBase &kb)
        {
            json op_data;
            op_data["state_id"] = op_state.state_id;
            op_data["action_type"] = SLI_Action_to_string(op_state.action);
            // op_data["node1_id"] = op_state.lit1_node ? op_state.lit1_node->node_id : -1;
            op_data["node1_lit"] = op_state.lit1_node ? op_state.lit1_node->literal.toString(kb) : "null";

            if (SLIOperation::isNode(op_state.second_op))
            {
                auto node = SLIOperation::getNode(op_state.second_op);
                op_data["second_operand_type"] = "node";
                op_data["second_operand_id"] = node ? node->node_id : -1;
            }
            else
            {
                auto lit = SLIOperation::getLiteral(op_state.second_op);
                op_data["second_operand_type"] = "literal";
                op_data["second_operand"] = lit.toString(kb);
            }

            if (!op_state.kb_clause.isEmpty())
            {
                op_data["kb_clause"] = op_state.kb_clause.toString(kb);
            }

            return op_data;
        }

        // 收集完整的训练样本
        static json collectTrainingSample(
            const SLIOperation::OperationState &state,
            const std::vector<std::shared_ptr<SLIOperation::OperationState>> &available_ops,
            const std::shared_ptr<SLIOperation::OperationState> &selected_op, // 新增参数
            double reward,
            const KnowledgeBase &kb)
        {
            json sample;
            sample["state_id"] = state.state_id;
            sample["current_tree_state"] = collectTreeState(*state.sli_tree, kb);

            // 收集可用操作
            json available_operations;
            for (const auto &op : available_ops)
            {
                available_operations.push_back(collectOperationData(*op, kb));
            }
            sample["available_operations"] = available_operations;

            // 记录实际选择的操作
            if (selected_op)
            {
                sample["selected_operation"] = collectOperationData(*selected_op, kb);
            }
            else
            {
                sample["selected_operation"] = nullptr;
            }

            sample["reward"] = reward;
            return sample;
        }

        // 保存数据到文件
        static void saveToFile(const std::vector<json> &samples, const std::string &filename)
        {
            json output;
            output["samples"] = samples;
            std::cout << "Save to file" << std::endl;
            std::ofstream file(filename);
            file << output.dump(4);
        }
    };

} // namespace LogicSystem

#endif // DATA_COLLECTOR_H