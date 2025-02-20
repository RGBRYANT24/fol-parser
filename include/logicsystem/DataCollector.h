// DataCollector.h
#ifndef DATA_COLLECTOR_H
#define DATA_COLLECTOR_H

#include <nlohmann/json.hpp>
#include "SLITree.h"
#include "SLIOperation.h"
#include "SymbolType.h"
#include <fstream>
#include <unordered_map>

using json = nlohmann::json;

namespace LogicSystem {

class DataCollector {
public:
    // 变量标准化上下文（每个样本独立）
    struct NormalizationContext {
        std::unordered_map<SymbolId, std::string> var_map;
        std::unordered_map<SymbolId, std::string> const_map;
        int var_counter = 0;
        int const_counter = 0;

        std::string normalizeSymbol(SymbolId id) {
            if(id.type == SymbolType::VARIABLE) {
                if(!var_map.count(id)) {
                    var_map[id] = "VAR" + std::to_string(var_counter++);
                }
                return var_map[id];
            } else {
                if(!const_map.count(id)) {
                    const_map[id] = "CONST" + std::to_string(const_counter++);
                }
                return const_map[id];
            }
        }
    };

    // 序列化单个节点
    static json serializeNode(const std::shared_ptr<SLINode>& node, 
                             KnowledgeBase& kb,
                             NormalizationContext& ctx) {
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
        for(auto arg : node->literal.getArgumentIds()) {
            args.push_back(ctx.normalizeSymbol(arg));
        }
        lit_json["arguments"] = args;
        node_json["literal"] = lit_json;

        // 替换关系
        json subst_json;
        for(auto& [var, val] : node->substitution) {
            subst_json[ctx.normalizeSymbol(var)] = ctx.normalizeSymbol(val);
        }
        node_json["substitution"] = subst_json;

        // 子节点ID
        std::vector<int> children_ids;
        for(auto& child : node->children) {
            children_ids.push_back(child->node_id);
        }
        node_json["children"] = children_ids;

        return node_json;
    }

    // 序列化整棵树
    static json serializeTree(const SLITree& tree, 
                             KnowledgeBase& kb,
                             NormalizationContext& ctx) {
        json tree_json;
        
        // 广度优先遍历
        std::queue<std::shared_ptr<SLINode>> q;
        q.push(tree.getRoot());
        
        while(!q.empty()) {
            auto node = q.front();
            q.pop();
            
            tree_json["nodes"].push_back(serializeNode(node, kb, ctx));
            
            for(auto& child : node->children) {
                q.push(child);
            }
        }

        // 全局特征
        tree_json["global"] = {
            {"depth", tree.getDepthMap().size()},
            {"active_nodes", tree.get_all_active_nodes().size()},
            {"has_self_loop", tree.hasSelfLoop()}
        };

        return tree_json;
    }

    // 序列化操作
    static json serializeOperation(const SLIOperation::OperationState& op,
                                  KnowledgeBase& kb,
                                  NormalizationContext& ctx) {
        json op_json;
        
        op_json["state_id"] = op.state_id;
        op_json["action"] = SLIOperation::getActionString(op.action);
        op_json["depth"] = op.depth;

        // 节点1信息
        if(op.lit1_node) {
            op_json["node1"] = serializeNode(op.lit1_node, kb, ctx);
        }

        // 操作数2
        if(SLIOperation::isNode(op.second_op)) {
            auto node = SLIOperation::getNode(op.second_op);
            op_json["operand2"]["type"] = "node";
            op_json["operand2"]["id"] = node ? node->node_id : -1;
        } else {
            auto lit = SLIOperation::getLiteral(op.second_op);
            json lit_json;
            lit_json["predicate"] = kb.getPredicateName(lit.getPredicateId());
            
            // 标准化参数
            std::vector<std::string> args;
            for(auto arg : lit.getArgumentIds()) {
                args.push_back(ctx.normalizeSymbol(arg));
            }
            lit_json["arguments"] = args;
            
            op_json["operand2"] = {
                {"type", "literal"},
                {"literal", lit_json}
            };
        }

        return op_json;
    }

    // 收集训练样本
    static json collectTrainingSample(
        const SLIOperation::OperationState& state,
        const std::vector<std::shared_ptr<SLIOperation::OperationState>>& available_ops,
        const std::shared_ptr<SLIOperation::OperationState>& selected_op,
        double reward,
        KnowledgeBase& kb) 
    {
        json sample;
        NormalizationContext ctx; // 每个样本独立上下文

        // 当前状态
        sample["state"] = {
            {"id", state.state_id},
            {"tree", serializeTree(*state.sli_tree, kb, ctx)},
            {"depth", state.depth}
        };

        // 可用操作
        json ops_json;
        for(auto& op : available_ops) {
            ops_json.push_back(serializeOperation(*op, kb, ctx));
        }
        sample["available_ops"] = ops_json;

        // 选择的操作
        if(selected_op) {
            sample["selected_op"] = serializeOperation(*selected_op, kb, ctx);
        }

        sample["reward"] = reward;

        return sample;
    }

    // 保存到文件
    static void saveToFile(const std::vector<json>& samples, const std::string& filename) {
        json output;
        output["samples"] = samples;
        
        std::ofstream file(filename);
        file << output.dump(2); // 缩进2个空格
    }
};

} // namespace LogicSystem
#endif // DATA_COLLECTOR_H