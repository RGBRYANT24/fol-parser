// DataCollector.h
#ifndef DATA_COLLECTOR_H
#define DATA_COLLECTOR_H

#include <nlohmann/json.hpp>
#include "SLITree.h"
#include "SLIOperation.h"
#include "SymbolType.h"
#include "TreeNodeT.h"
#include "SLIMCTSState.h"
// #include "SLIResolution.h"
#include <fstream>
#include <unordered_map>
#include <queue>

using json = nlohmann::json;

namespace LogicSystem
{

    class DataCollector
    {
    public:
        // 变量标准化上下文（每个样本独立）
        struct NormalizationContext
        {
            std::unordered_map<SymbolId, std::string> var_map;
            std::unordered_map<SymbolId, std::string> const_map;
            int var_counter = 0;
            int const_counter = 0;

            std::string normalizeSymbol(SymbolId id)
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

        // 序列化单个节点
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

            // 参数标准化
            std::vector<std::string> args;
            for (auto arg : node->literal.getArgumentIds())
            {
                args.push_back(ctx.normalizeSymbol(arg));
            }
            lit_json["arguments"] = args;
            node_json["literal"] = lit_json;

            // 替换关系
            json subst_json;
            for (auto &[var, val] : node->substitution)
            {
                subst_json[ctx.normalizeSymbol(var)] = ctx.normalizeSymbol(val);
            }
            node_json["substitution"] = subst_json;

            // 子节点ID
            std::vector<int> children_ids;
            for (auto &child : node->children)
            {
                children_ids.push_back(child->node_id);
            }
            node_json["children"] = children_ids;

            return node_json;
        }

        // 序列化整棵树
        static json serializeTree(const SLITree &tree,
                                  KnowledgeBase &kb,
                                  NormalizationContext &ctx)
        {
            json tree_json;

            // 广度优先遍历
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

            // 全局特征
            tree_json["global"] = {
                {"depth", tree.getDepthMap().size()},
                {"active_nodes", tree.get_all_active_nodes().size()},
                {"has_self_loop", tree.hasSelfLoop()}};

            return tree_json;
        }

        // 序列化操作（普通数据收集）
        static json serializeOperation(const SLIOperation::OperationState &op,
                                       KnowledgeBase &kb,
                                       NormalizationContext &ctx)
        {
            json op_json;

            op_json["state_id"] = op.state_id;
            op_json["action"] = SLIOperation::getActionString(op.action);
            op_json["depth"] = op.depth;

            // 节点1信息
            if (op.lit1_node)
            {
                op_json["node1"] = serializeNode(op.lit1_node, kb, ctx);
            }

            // 操作数2
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

                // 标准化参数
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

            // 可用操作
            json ops_json;
            for (auto &op : available_ops)
            {
                ops_json.push_back(serializeOperation(*op, kb, ctx));
            }
            sample["available_ops"] = ops_json;

            // 选择的操作
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

        // 计算当前节点下各个操作参数的奖励信息，包含两部分：
        // 1. 按操作类型聚合的期望奖励（采用访问次数加权平均）
        // 2. 每个候选动作的奖励值（按扩展时的下标对应关系）
        template <class State, typename Action>
        static nlohmann::json computeExpectedOpRewards(
            const std::shared_ptr<msa::mcts::TreeNodeT<State, Action>> &node)
        {
            // 1. 按操作类型聚合的期望奖励
            std::unordered_map<std::string, std::pair<double, double>> reward_map;
            auto children = node->get_children(); // 已扩展的子节点

            // 遍历所有已扩展的子节点
            for (const auto &child : children)
            {
                // 获取扩展该子节点时使用的动作的操作类型字符串（例如 "EXTENSION", "FACTORING", "ANCESTRY"）
                std::string op_type = SLI_Action_to_string(child->get_action().action);
                // 如果操作类型是 "TRUNCATE"，则不需要处理，直接跳过
                if (op_type == "TRUNCATE")
                    continue;

                // 归一化奖励：例如用子节点累计评价除以 (深度+1)
                double normalized_reward = child->get_value() / (child->get_depth() + 1.0);
                double visits = child->get_num_visits();
                reward_map[op_type].first += visits * normalized_reward; // 累加加权奖励
                reward_map[op_type].second += visits;                    // 累加访问次数
            }

            // 定义需要输出的操作类型（只包含 3 种）
            std::vector<std::string> required_ops = {"EXTENSION", "FACTORING", "ANCESTRY"};
            nlohmann::json expected_by_type;
            for (const auto &op : required_ops)
            {
                double exp_reward = 0.0;
                // 如果该操作类型有累积记录，则计算加权平均奖励；否则保持为 0.0
                if (reward_map.find(op) != reward_map.end() && reward_map[op].second > 0)
                {
                    exp_reward = reward_map[op].first / reward_map[op].second;
                }
                expected_by_type[op] = exp_reward;
            }

            // 2. 每个候选动作的奖励信息
            // 获取当前节点中的所有候选动作，这个顺序在扩展时已经经过随机打乱，但后续扩展时始终按该顺序进行对应
            auto available_actions = node->get_actions();
            nlohmann::json action_rewards = nlohmann::json::array();
            // 遍历所有候选动作
            for (size_t i = 0; i < available_actions.size(); ++i)
            {
                double reward = 0.0;
                // 如果该候选动作已经扩展（即 i < children.size()），则使用对应子节点的奖励信息
                if (i < children.size())
                {
                    auto child = children[i];
                    reward = child->get_value() / (child->get_depth() + 1.0);
                }
                // 否则该候选动作还没有被扩展，默认奖励设为 0
                nlohmann::json action_reward_info;
                action_reward_info["index"] = i;
                action_reward_info["op_type"] = SLI_Action_to_string(available_actions[i].action);
                action_reward_info["reward"] = reward;
                action_rewards.push_back(action_reward_info);
            }

            // 组合返回结果：包含按操作类型聚合的奖励和逐动作的奖励信息
            nlohmann::json result;
            result["expected_by_type"] = expected_by_type;
            result["action_rewards"] = action_rewards;

            return result;
        }

        // 针对 MCTS 进行 Data Collect
        static json serializeOperationMCTS(const SLIMCTSAction &op,
                                           const SLIMCTSState &state,
                                           KnowledgeBase &kb,
                                           NormalizationContext &ctx)
        {
            json op_json;
            // 操作类型转换为字符串，保持和普通操作一致
            op_json["action"] = SLI_Action_to_string(op.action);
            // 采用父状态深度+1作为当前操作对应的深度
            op_json["depth"] = state.getDepth() + 1;
            // 模拟生成下一个状态的 id，这里设为父状态 id+1
            // op_json["state_id"] = state.getStateId() + 1;

            // 序列化 node1，由 op.lit1_node_id 查找
            auto node = state.sli_tree->findNodeById(op.lit1_node_id);
            if (node)
                op_json["node1"] = serializeNode(node, kb, ctx);
            else
                op_json["node1"] = nullptr;

            // 序列化 operand2
            json operand2;
            if (std::holds_alternative<Literal>(op.second_op))
            {
                Literal lit = std::get<Literal>(op.second_op);
                json lit_json;
                lit_json["predicate"] = kb.getPredicateName(lit.getPredicateId());
                // 标准化参数（与普通 serializeOperation 一致）
                std::vector<std::string> args;
                for (auto arg : lit.getArgumentIds())
                {
                    args.push_back(ctx.normalizeSymbol(arg));
                }
                lit_json["arguments"] = args;
                // 添加 literal 的否定信息
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

            // 新增：序列化 kb_clause，将其中每个 literal 转换为 JSON 对象
            json clause_literals = json::array();
            // 假设 op.kb_clause.getLiterals() 返回 Literal 的 vector
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

        // 针对 MCTS 进行 Data Collect：利用 TreeNode 信息采集训练样本
        // 为了使生成的 JSON 数据格式与之前保持一致，本函数构造的样本包含：
        // "state" (包含 id, tree, depth)、"available_ops"、"selected_op"（此处未选择则设为 null）及 "reward"（默认 0.0）
        // 针对MCTS进行Data Collect：利用 TreeNode 信息采集训练样本
        // 针对MCTS进行Data Collect：利用 TreeNode 信息采集训练样本
        // 修改后的版本：available_ops中的每个操作对象都附加了对应的reward字段
        static nlohmann::json collectTrainingSampleMCTS(
            const std::shared_ptr<msa::mcts::TreeNodeT<SLIMCTSState, SLIMCTSAction>> &tree_node,
            KnowledgeBase &kb)
        {
            nlohmann::json sample;
            DataCollector::NormalizationContext ctx; // 每个样本独立上下文

            auto state = tree_node->get_state();

            // 状态信息：包含 tree 和 depth（如需要可添加 id 字段）
            sample["state"] = {
                {"tree", DataCollector::serializeTree(*state.sli_tree, kb, ctx)},
                {"depth", state.getDepth()}};

            // 首先调用 computeExpectedOpRewards，计算奖励信息
            nlohmann::json reward_info = computeExpectedOpRewards<SLIMCTSState, SLIMCTSAction>(tree_node);
            // 从中提取每个候选动作的奖励信息，要求顺序与 tree_node->get_actions() 一致
            nlohmann::json action_rewards = reward_info["action_rewards"];

            // 构造可用操作列表，并利用 action_rewards 信息为每个操作添加 reward 字段
            nlohmann::json ops_json = nlohmann::json::array();
            const auto &actions = tree_node->get_actions();
            for (size_t i = 0; i < actions.size(); ++i)
            {
                // 序列化当前操作
                nlohmann::json op_json = DataCollector::serializeOperationMCTS(actions[i], state, kb, ctx);

                // 查找对应下标的奖励值，假定 action_rewards 数组的顺序与 actions 保持一致
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

            // 同时将全局汇聚奖励 expected_by_type 保留到 reward 字段下作为全局信息
            sample["global_reward"] = {{"expected_by_type", reward_info["expected_by_type"]}};

            // MCTS中暂未使用选中操作，设为 null
            sample["selected_op"] = nullptr;

            return sample;
        }

        // 保存到文件
        static void saveToFile(const std::vector<json> &samples, const std::string &filename)
        {
            json output;
            output["samples"] = samples;

            std::ofstream file(filename);
            file << output.dump(2); // 缩进2个空格
        }
    };

} // namespace LogicSystem
#endif // DATA_COLLECTOR_H