#ifndef DATA_COLLECTOR_H
#define DATA_COLLECTOR_H

#include <nlohmann/json.hpp>
#include "SLITree.h"
#include "SLIOperation.h"
#include "SymbolType.h"
#include "TreeNodeT.h"
#include "SLIMCTSState.h"
#include <fstream>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <iostream>

using json = nlohmann::json;

namespace LogicSystem
{

    class DataCollector
    {
    public:
        // 变量/常量统一编码上下文（每个样本独立）
        struct NormalizationContext
        {
            std::unordered_map<SymbolId, std::string> var_map;
            std::unordered_map<SymbolId, std::string> const_map;
            int var_counter = 0;
            int const_counter = 0;

            std::string normalizeSymbol(const SymbolId &id)
            {
                if (id.type == SymbolType::VARIABLE)
                {
                    if (!var_map.count(id))
                    {
                        var_map[id] = "VAR" + std::to_string(var_counter++);
                    }
                    return var_map[id];
                }
                else
                {
                    if (!const_map.count(id))
                    {
                        const_map[id] = "CONST" + std::to_string(const_counter++);
                    }
                    return const_map[id];
                }
            }
        };

        ////////////////////////////
        // 序列化单个节点（SLITree 内部节点）
        static json serializeNode(const std::shared_ptr<SLINode> &node,
                                  KnowledgeBase &kb,
                                  NormalizationContext &ctx)
        {
            json node_json;
            // 基础信息
            node_json["id"] = node->node_id;
            node_json["active"] = node->is_active;
            node_json["depth"] = node->depth;
            node_json["type"] = node->is_A_literal ? "A" : "B";

            // 文字信息
            json lit_json;
            lit_json["predicate"] = kb.getPredicateName(node->literal.getPredicateId());
            lit_json["negated"] = node->literal.isNegated();

            // 参数标准化：使用同一个 ctx 保证统一
            std::vector<std::string> args;
            for (auto arg : node->literal.getArgumentIds())
            {
                args.push_back(ctx.normalizeSymbol(arg));
            }
            lit_json["arguments"] = args;
            node_json["literal"] = lit_json;

            // 替换关系（如果有的话）
            json subst_json;
            for (auto &[var, val] : node->substitution)
            {
                subst_json[ctx.normalizeSymbol(var)] = ctx.normalizeSymbol(val);
            }
            node_json["substitution"] = subst_json;

            // 子节点ID列表
            std::vector<int> children_ids;
            for (auto &child : node->children)
            {
                children_ids.push_back(child->node_id);
            }
            node_json["children"] = children_ids;

            return node_json;
        }

        ////////////////////////////
        // 序列化整棵SLITree
        static json serializeTree(const SLITree &tree,
                                  KnowledgeBase &kb,
                                  NormalizationContext &ctx)
        {
            json tree_json;
            std::queue<std::shared_ptr<SLINode>> q;
            q.push(tree.getRoot());
            while (!q.empty())
            {
                auto node = q.front();
                q.pop();
                tree_json["nodes"].push_back(serializeNode(node, kb, ctx));
                for (auto &child : node->children)
                {
                    q.push(child);
                }
            }
            tree_json["global"] = {
                {"depth", tree.getDepthMap().size()},
                {"active_nodes", tree.get_all_active_nodes().size()},
                {"has_self_loop", tree.hasSelfLoop()}};
            return tree_json;
        }

        ////////////////////////////
        // 序列化操作（普通数据收集）
        static json serializeOperation(const SLIOperation::OperationState &op,
                                       KnowledgeBase &kb,
                                       NormalizationContext &ctx)
        {
            json op_json;
            op_json["state_id"] = op.state_id;
            op_json["action"] = SLIOperation::getActionString(op.action);
            op_json["depth"] = op.depth;
            if (op.lit1_node)
            {
                op_json["node1"] = serializeNode(op.lit1_node, kb, ctx);
            }
            if (SLIOperation::isNode(op.second_op))
            {
                auto node = SLIOperation::getNode(op.second_op);
                op_json["operand2"]["type"] = "node";
                op_json["operand2"]["id"] = node ? node->node_id : -1;
            }
            else
            {
                auto lit = SLIOperation::getLiteral(op.second_op);
                json lit_json;
                lit_json["predicate"] = kb.getPredicateName(lit.getPredicateId());
                std::vector<std::string> args;
                for (auto arg : lit.getArgumentIds())
                {
                    args.push_back(ctx.normalizeSymbol(arg));
                }
                lit_json["arguments"] = args;
                op_json["operand2"] = {
                    {"type", "literal"},
                    {"literal", lit_json}};
            }
            return op_json;
        }

        ////////////////////////////
        // 收集训练样本（普通版本）
        static json collectTrainingSample(
            const SLIOperation::OperationState &state,
            const std::vector<std::shared_ptr<SLIOperation::OperationState>> &available_ops,
            const std::shared_ptr<SLIOperation::OperationState> &selected_op,
            double reward,
            KnowledgeBase &kb)
        {
            json sample;
            NormalizationContext ctx; // 每个样本独立上下文

            // 当前状态
            sample["state"] = {
                {"id", state.state_id},
                {"tree", serializeTree(*state.sli_tree, kb, ctx)},
                {"depth", state.depth}};
            json ops_json;
            for (auto &op : available_ops)
            {
                ops_json.push_back(serializeOperation(*op, kb, ctx));
            }
            sample["available_ops"] = ops_json;
            if (selected_op)
            {
                sample["selected_op"] = serializeOperation(*selected_op, kb, ctx);
            }
            else
            {
                sample["selected_op"] = nullptr;
            }
            sample["reward"] = reward;
            return sample;
        }

        ////////////////////////////
        // 计算候选动作奖励信息（略，此处代码保持不变）
        template <class State, typename Action>
        static nlohmann::json computeExpectedOpRewards(
            const std::shared_ptr<msa::mcts::TreeNodeT<State, Action>> &node)
        {
            std::unordered_map<std::string, std::pair<double, double>> reward_map;
            auto children = node->get_children();
            for (const auto &child : children)
            {
                std::string op_type = SLI_Action_to_string(child->get_action().action);
                if (op_type == "TRUNCATE")
                    continue;
                double normalized_reward = child->get_value() / (child->get_depth() + 1.0);
                double visits = child->get_num_visits();
                reward_map[op_type].first += visits * normalized_reward;
                reward_map[op_type].second += visits;
            }
            std::vector<std::string> required_ops = {"EXTENSION", "FACTORING", "ANCESTRY"};
            nlohmann::json expected_by_type;
            for (const auto &op : required_ops)
            {
                double exp_reward = 0.0;
                if (reward_map.find(op) != reward_map.end() && reward_map[op].second > 0)
                {
                    exp_reward = reward_map[op].first / reward_map[op].second;
                }
                expected_by_type[op] = exp_reward;
            }
            auto available_actions = node->get_actions();
            nlohmann::json action_rewards = nlohmann::json::array();
            for (size_t i = 0; i < available_actions.size(); ++i)
            {
                double reward = 0.0;
                if (i < children.size())
                {
                    auto child = children[i];
                    reward = child->get_value() / (child->get_depth() + 1.0);
                }
                nlohmann::json action_reward_info;
                action_reward_info["index"] = i;
                action_reward_info["op_type"] = SLI_Action_to_string(available_actions[i].action);
                action_reward_info["reward"] = reward;
                action_rewards.push_back(action_reward_info);
            }
            nlohmann::json result;
            result["expected_by_type"] = expected_by_type;
            result["action_rewards"] = action_rewards;
            return result;
        }

        ////////////////////////////
        // 针对 MCTS 的 Data Collect，修改为接受外部传入的 ctx
        static nlohmann::json collectTrainingSampleMCTS(
            const std::shared_ptr<msa::mcts::TreeNodeT<SLIMCTSState, SLIMCTSAction>> &tree_node,
            KnowledgeBase &kb,
            NormalizationContext &ctx) // 使用外部传入的 ctx（保证图和 SLITree 公用同一份映射）
        {
            nlohmann::json sample;
            auto state = tree_node->get_state();

            // 先序列化图，借此预注册所有图中出现的常量（确保顺序一致）  
            // 若你的图已经单独保存，这里可以选择先调用 serializeGraph()（见后续函数）  
            // 否则也可以认为在构造 KB 时图中所有边都已被 normalizeSymbol 注册。

            sample["state"] = {
                {"tree", serializeTree(*state.sli_tree, kb, ctx)},
                {"depth", state.getDepth()}};

            nlohmann::json reward_info = computeExpectedOpRewards<SLIMCTSState, SLIMCTSAction>(tree_node);
            nlohmann::json action_rewards = reward_info["action_rewards"];

            nlohmann::json ops_json = nlohmann::json::array();
            const auto &actions = tree_node->get_actions();
            for (size_t i = 0; i < actions.size(); ++i)
            {
                nlohmann::json op_json = serializeOperationMCTS(actions[i], state, kb, ctx);
                if (i < action_rewards.size())
                {
                    op_json["reward"] = action_rewards[i]["reward"];
                }
                else
                {
                    op_json["reward"] = 0.0;
                }
                ops_json.push_back(op_json);
            }
            sample["available_ops"] = ops_json;
            sample["global_reward"] = {{"expected_by_type", reward_info["expected_by_type"]}};
            sample["selected_op"] = nullptr;
            return sample;
        }

        ////////////////////////////
        // 针对 MCTS 的操作序列化，采用共享的 ctx
        static json serializeOperationMCTS(const SLIMCTSAction &op,
                                           const SLIMCTSState &state,
                                           KnowledgeBase &kb,
                                           NormalizationContext &ctx)
        {
            json op_json;
            op_json["action"] = SLI_Action_to_string(op.action);
            op_json["depth"] = state.getDepth() + 1;
            auto node = state.sli_tree->findNodeById(op.lit1_node_id);
            if (node)
                op_json["node1"] = serializeNode(node, kb, ctx);
            else
                op_json["node1"] = nullptr;

            json operand2;
            if (std::holds_alternative<Literal>(op.second_op))
            {
                Literal lit = std::get<Literal>(op.second_op);
                json lit_json;
                lit_json["predicate"] = kb.getPredicateName(lit.getPredicateId());
                std::vector<std::string> args;
                for (auto arg : lit.getArgumentIds())
                {
                    args.push_back(ctx.normalizeSymbol(arg));
                }
                lit_json["arguments"] = args;
                lit_json["negated"] = lit.isNegated();
                operand2["type"] = "literal";
                operand2["literal"] = lit_json;
            }
            else if (std::holds_alternative<int>(op.second_op))
            {
                int node_id = std::get<int>(op.second_op);
                operand2["type"] = "node";
                operand2["id"] = node_id;
            }
            op_json["operand2"] = operand2;

            json clause_literals = json::array();
            for (const Literal &lit : op.kb_clause.getLiterals())
            {
                json lit_json;
                lit_json["predicate"] = kb.getPredicateName(lit.getPredicateId());
                std::vector<std::string> args;
                for (auto arg : lit.getArgumentIds())
                {
                    args.push_back(ctx.normalizeSymbol(arg));
                }
                lit_json["arguments"] = args;
                lit_json["negated"] = lit.isNegated();
                clause_literals.push_back(lit_json);
            }
            op_json["kb_clause"] = clause_literals;

            return op_json;
        }

        ////////////////////////////
        // 序列化图结构：遍历 KB 中所有满足条件的 Clause（例如：长度为1且谓词为 "E" 且参数均为常量）
        static nlohmann::json serializeGraph(const KnowledgeBase &kb,
                                             NormalizationContext &ctx)
        {
            nlohmann::json graph_json;
            // 用于记录节点（按照第一次遇到顺序）
            std::vector<SymbolId> unique_nodes;
            std::unordered_set<SymbolId> encountered; // 要求 SymbolId 定义了适当的 hash

            nlohmann::json edges_json = nlohmann::json::array();

            // 遍历 KB 中所有 Clause
            for (const Clause &clause : kb.getClauses())
            {
                const auto &lits = clause.getLiterals();
                if (lits.size() == 1)
                {
                    const Literal &lit = lits[0];
                    if (kb.getPredicateName(lit.getPredicateId()) == "E")
                    {
                        const auto &args = lit.getArgumentIds();
                        if (args.size() == 2)
                        {
                            if (!kb.isVariable(args[0]) && !kb.isVariable(args[1]))
                            {
                                if (encountered.find(args[0]) == encountered.end())
                                {
                                    encountered.insert(args[0]);
                                    unique_nodes.push_back(args[0]);
                                }
                                if (encountered.find(args[1]) == encountered.end())
                                {
                                    encountered.insert(args[1]);
                                    unique_nodes.push_back(args[1]);
                                }
                                nlohmann::json lit_json;
                                lit_json["predicate"] = kb.getPredicateName(lit.getPredicateId());
                                std::vector<std::string> arg_strs;
                                arg_strs.push_back(ctx.normalizeSymbol(args[0]));
                                arg_strs.push_back(ctx.normalizeSymbol(args[1]));
                                lit_json["arguments"] = arg_strs;
                                nlohmann::json edge_entry;
                                edge_entry["literals"] = nlohmann::json::array({lit_json});
                                edges_json.push_back(edge_entry);
                            }
                        }
                    }
                }
            }

            // 为了确保后续使用中该图常量映射不变，预先调用 normalizeSymbol 注册所有节点
            for (const SymbolId &node_id : unique_nodes)
            {
                ctx.normalizeSymbol(node_id);
            }

            nlohmann::json nodes_json = nlohmann::json::array();
            for (const SymbolId &node_id : unique_nodes)
            {
                nlohmann::json node_entry;
                std::string normName = ctx.normalizeSymbol(node_id);
                node_entry["id"] = normName;
                nodes_json.push_back(node_entry);
            }

            graph_json["nodes"] = nodes_json;
            graph_json["edges"] = edges_json;
            return graph_json;
        }

        ////////////////////////////
        // 修改后的保存函数：将最终 JSON 写入文件
        static void saveToFile(const nlohmann::json &sample, const std::string &filename)
        {
            std::ofstream file(filename);
            if (!file)
            {
                std::cerr << "无法打开文件 " << filename << " 进行写入。" << std::endl;
                return;
            }
            file << sample.dump(2);
        }
    };

} // namespace LogicSystem

#endif // DATA_COLLECTOR_H