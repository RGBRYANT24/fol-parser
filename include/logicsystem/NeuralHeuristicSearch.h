// NeuralHeuristicSearch.h
#ifndef NEURAL_HEURISTIC_SEARCH_H
#define NEURAL_HEURISTIC_SEARCH_H

#include <string>
#include <memory>
#include <vector>
#include <stack>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include "ProcessManager.h"
#include "DataCollector.h"
#include "SLIOperation.h"
#include "KnowledgeBase.h"

using json = nlohmann::json;

namespace LogicSystem
{
    class NeuralHeuristicSearch
    {
    private:
        // 分别为两个阶段创建进程管理器
        std::unique_ptr<ProcessManager> phase1ProcessManager;
        std::unique_ptr<ProcessManager> phase2ProcessManager;

        // 使用哈希集合跟踪已访问状态以避免循环
        std::unordered_set<size_t> visited_states;

        // 实验设置
        enum ExperimentMode
        {
            PHASE1_ONLY, // 只使用第一阶段（操作类型评分）
            PHASE2_ONLY, // 只使用第二阶段（参数评分）
            BOTH_PHASES  // 两个阶段都使用
        };

        ExperimentMode experiment_mode = BOTH_PHASES;

        // 获取第一阶段神经网络对操作类型的评分
        json getActionScores(const std::shared_ptr<SLITree> &tree, KnowledgeBase &kb)
        {
            if (!phase1ProcessManager->isActive())
            {
                return {{"status", "error"}, {"error_message", "第一阶段神经网络服务未初始化"}};
            }

            // 针对实验模式，如果是只使用第二阶段，返回统一的默认评分
            if (experiment_mode == PHASE2_ONLY)
            {
                return {
                    {"status", "success"},
                    {"action_scores", {0.25, 0.97, 0.98, 0.99}} // 均匀分布的评分
                };
            }

            // 准备状态数据
            DataCollector::NormalizationContext ctx;
            json state_json = DataCollector::serializeTree(*tree, kb, ctx);
            json graph_json = DataCollector::serializeGraph(kb, ctx);

            // 构建用于神经网络的请求
            json request = {
                {"state", {{"tree", state_json}}},
                {"graph", graph_json},
                {"request_type", "action_scores"}};

            // 发送请求并获取响应
            json response = phase1ProcessManager->sendRequest(request);

            if (response.contains("status") && response["status"] == "error")
            {
                std::cerr << "第一阶段神经网络服务返回错误: " << response["error_message"] << std::endl;
                if (response.contains("error_details"))
                {
                    std::cerr << "错误详情: " << response["error_details"] << std::endl;
                }
                // 出错时返回默认评分
                return {
                    {"status", "success"},
                    {"action_scores", {0.25, 0.97, 0.98, 0.99}} // Extension, Factoring, Ancestry, Truncate
                };
            }

            return response;
        }

        // 第二阶段：获取神经网络对特定操作参数的评分
        json getParameterScores(
            SLIActionType actionType,
            const std::vector<std::shared_ptr<SLIOperation::OperationState>> &states,
            KnowledgeBase &kb)
        {
            if (!phase2ProcessManager->isActive() || states.empty())
            {
                return {{"status", "error"}, {"error_message", "第二阶段神经网络服务未初始化或无可用操作"}};
            }

            // 针对实验模式，如果是只使用第一阶段，返回统一的默认评分
            if (experiment_mode == PHASE1_ONLY)
            {
                json scores = json::array();
                // 为每个参数组合生成相同分数
                for (size_t i = 0; i < states.size(); i++)
                {
                    scores.push_back(1.0);
                }
                return {
                    {"status", "success"},
                    {"parameter_scores", scores}};
            }

            // 获取参考状态（第一个状态）的树作为基准
            auto &reference_state = states[0];
            auto &tree = reference_state->sli_tree;

            // 创建统一的上下文，确保变量/常量编码一致
            DataCollector::NormalizationContext ctx;

            // 序列化图结构（仅需一次）
            json graph_json = DataCollector::serializeGraph(kb, ctx);

            // 序列化树结构（仅需一次）
            json tree_json = DataCollector::serializeTree(*tree, kb, ctx);

            // 序列化所有可用操作
            json operations_json = json::array();
            for (const auto &state : states)
            {
                // 使用serializeOperation函数序列化操作
                json op_json = serializeOperation(*state, kb, ctx);

                // 对于EXTENSION操作，添加KB子句信息
                if (state->action == SLIActionType::EXTENSION)
                {
                    json clause_literals = json::array();
                    for (const Literal &clause_lit : state->kb_clause.getLiterals())
                    {
                        json clause_lit_json;
                        clause_lit_json["predicate"] = kb.getPredicateName(clause_lit.getPredicateId());
                        clause_lit_json["negated"] = clause_lit.isNegated();

                        std::vector<std::string> clause_args;
                        for (auto arg : clause_lit.getArgumentIds())
                        {
                            clause_args.push_back(ctx.normalizeSymbol(arg));
                        }
                        clause_lit_json["arguments"] = clause_args;
                        clause_literals.push_back(clause_lit_json);
                    }
                    op_json["kb_clause"] = clause_literals;
                }

                operations_json.push_back(op_json);
            }

            // 构建最终请求
            json request = {
                {"state", {{"tree", tree_json}}},
                {"graph", graph_json},
                {"action_type", SLI_Action_to_string(actionType)},
                {"operations", operations_json},
                {"request_type", "parameter_scores"}};

            // 发送请求并获取响应
            json response = phase2ProcessManager->sendRequest(request);
            // std::cout << "NeuralHeuristicSearch::getParameterScores request " << request <<std::endl;

            if (response.contains("status") && response["status"] == "error")
            {
                std::cerr << "获取参数评分失败: " << response["error_message"] << std::endl;
                if (response.contains("error_details"))
                {
                    std::cerr << "错误详情: " << response["error_details"] << std::endl;
                }

                // 返回默认评分
                json scores = json::array();
                for (size_t i = 0; i < states.size(); i++)
                {
                    // 简单地按顺序递减评分，给第一个最高分
                    scores.push_back(1.0 - (static_cast<double>(i) / states.size()));
                }
                return {
                    {"status", "success"},
                    {"parameter_scores", scores}};
            }

            return response;
        }

        // 辅助函数：序列化操作状态
        static json serializeOperation(const SLIOperation::OperationState &op,
                                       KnowledgeBase &kb,
                                       DataCollector::NormalizationContext &ctx)
        {
            json op_json;
            op_json["state_id"] = op.state_id;
            op_json["action"] = SLI_Action_to_string(op.action);
            op_json["depth"] = op.depth;

            if (op.lit1_node)
            {
                op_json["node1"] = DataCollector::serializeNode(op.lit1_node, kb, ctx);
            }

            // 处理第二个操作数
            if (std::holds_alternative<std::shared_ptr<SLINode>>(op.second_op))
            {
                auto node = std::get<std::shared_ptr<SLINode>>(op.second_op);
                if (node)
                {
                    op_json["operand2"] = {
                        {"type", "node"},
                        {"id", node->node_id},
                        {"node", DataCollector::serializeNode(node, kb, ctx)}};
                }
                else
                {
                    op_json["operand2"] = {{"type", "null"}};
                }
            }
            else if (std::holds_alternative<Literal>(op.second_op))
            {
                auto lit = std::get<Literal>(op.second_op);
                json lit_json;
                lit_json["predicate"] = kb.getPredicateName(lit.getPredicateId());
                lit_json["negated"] = lit.isNegated();

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
            else
            {
                // 处理可能的空操作数，例如TRUNCATE操作
                op_json["operand2"] = {{"type", "null"}};
            }

            return op_json;
        }

        // 检查是否到达空子句
        bool checkEmptyClause(const std::shared_ptr<SLITree> &tree)
        {
            return tree->get_all_active_nodes().size() == 0;
        }

        // 生成所有可能的扩展操作状态
        std::vector<std::shared_ptr<SLIOperation::OperationState>> generateExtensionStates(
            KnowledgeBase &kb,
            const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
            const std::shared_ptr<SLIOperation::OperationState> &parent_state)
        {

            std::vector<std::shared_ptr<SLIOperation::OperationState>> states;

            for (const auto &node : b_lit_nodes)
            {
                if (!node->is_active || node->is_A_literal)
                    continue;

                for (const auto &kb_clause : kb.getClauses())
                {
                    for (const auto &lit : kb_clause.getLiterals())
                    {
                        if (Resolution::isComplementary(node->literal, lit) &&
                            Unifier::findMGU(node->literal, lit, kb))
                        {
                            auto new_state = SLIOperation::createExtensionState(
                                parent_state->sli_tree,
                                node,
                                lit,
                                kb_clause,
                                parent_state);

                            states.push_back(new_state);
                        }
                    }
                }
            }

            return states;
        }

        // 生成所有可能的Factoring操作状态
        std::vector<std::shared_ptr<SLIOperation::OperationState>> generateFactoringStates(
            const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
            const std::shared_ptr<SLIOperation::OperationState> &parent_state)
        {
            std::vector<std::shared_ptr<SLIOperation::OperationState>> states;
            auto factoring_pairs = SLIResolution::findPotentialFactoringPairs(parent_state->sli_tree);

            for (const auto &[upper_node, lower_node] : factoring_pairs)
            {
                auto new_state = SLIOperation::createFactoringState(
                    parent_state->sli_tree,
                    upper_node,
                    lower_node,
                    parent_state);
                states.push_back(new_state);
            }

            return states;
        }

        // 生成所有可能的Ancestry操作状态
        std::vector<std::shared_ptr<SLIOperation::OperationState>> generateAncestryStates(
            const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
            const std::shared_ptr<SLIOperation::OperationState> &parent_state)
        {
            std::vector<std::shared_ptr<SLIOperation::OperationState>> states;
            auto ancestry_pairs = SLIResolution::findPotentialAncestryPairs(parent_state->sli_tree);

            for (const auto &[upper_node, lower_node] : ancestry_pairs)
            {
                auto new_state = SLIOperation::createAncestryState(
                    parent_state->sli_tree,
                    upper_node,
                    lower_node,
                    parent_state);
                states.push_back(new_state);
            }

            return states;
        }

        // 生成所有可能的Truncate操作状态
        std::vector<std::shared_ptr<SLIOperation::OperationState>> generateTruncateStates(
            const std::vector<std::shared_ptr<SLINode>> &active_nodes,
            const std::shared_ptr<SLIOperation::OperationState> &parent_state)
        {
            std::vector<std::shared_ptr<SLIOperation::OperationState>> states;
            auto truncate_nodes = SLIResolution::findPotentialTruncateNodes(parent_state->sli_tree);

            for (const auto &node : truncate_nodes)
            {
                auto new_state = SLIOperation::createTruncateState(
                    parent_state->sli_tree,
                    node,
                    parent_state);
                states.push_back(new_state);
            }

            return states;
        }

        // 执行操作并检查是否成功
        bool applyOperation(std::shared_ptr<SLIOperation::OperationState> &state)
        {
            try
            {
                SLIOperation::performAction(state);
                return true;
            }
            catch (const std::exception &e)
            {
                std::cerr << "操作执行失败: " << e.what() << std::endl;
                return false;
            }
        }

    public:
        NeuralHeuristicSearch() : phase1ProcessManager(std::make_unique<ProcessManager>()),
                                  phase2ProcessManager(std::make_unique<ProcessManager>()) {}

        ~NeuralHeuristicSearch()
        {
            if (phase1ProcessManager->isActive())
            {
                phase1ProcessManager->stopProcess();
            }
            if (phase2ProcessManager->isActive())
            {
                phase2ProcessManager->stopProcess();
            }
        }

        // 设置实验模式
        void setExperimentMode(const std::string &mode)
        {
            if (mode == "phase1_only")
            {
                experiment_mode = PHASE1_ONLY;
                std::cout << "实验模式: 仅使用第一阶段（操作类型评分）" << std::endl;
            }
            else if (mode == "phase2_only")
            {
                experiment_mode = PHASE2_ONLY;
                std::cout << "实验模式: 仅使用第二阶段（参数评分）" << std::endl;
            }
            else
            {
                experiment_mode = BOTH_PHASES;
                std::cout << "实验模式: 同时使用两个阶段" << std::endl;
            }
        }

        // 获取当前实验模式
        std::string getExperimentMode() const
        {
            switch (experiment_mode)
            {
            case PHASE1_ONLY:
                return "phase1_only";
            case PHASE2_ONLY:
                return "phase2_only";
            case BOTH_PHASES:
                return "both_phases";
            default:
                return "unknown";
            }
        }

        // 新增方法获取访问状态数
        int getVisitedStatesCount() const
        {
            return visited_states.size();
        }

        // 修改initialize方法，使其使用neural_server.py和neural_server2.py两个不同的脚本
        bool initialize(
            const std::string &pythonPath,
            const std::string &phase1ScriptPath, // 第一阶段脚本路径 (neural_server.py)
            const std::string &phase1ModelPath,  // 第一阶段模型路径
            const std::string &phase1TokenizerPath,
            const std::string &phase2ScriptPath, // 第二阶段脚本路径 (neural_server2.py)
            const std::string &phase2ModelPath,  // 第二阶段模型路径
            const std::string &phase2TokenizerPath)
        {
            std::cout << "NeuralHeuristicSearch::initialize Init phase1 model: " << phase1ModelPath
                      << " script: " << phase1ScriptPath << std::endl;

            // 初始化第一阶段进程，使用neural_server.py
            bool phase1Init = phase1ProcessManager->initialize(
                pythonPath, phase1ScriptPath, phase1ModelPath, phase1TokenizerPath);

            if (!phase1Init)
            {
                std::cerr << "第一阶段神经网络初始化失败" << std::endl;
                return false;
            }
            std::cout << "NeuralHeuristicSearch::initialize phase1 initialize ok" << std::endl;

            // 初始化第二阶段进程，使用neural_server2.py
            std::cout << "NeuralHeuristicSearch::initialize Init phase2 model: " << phase2ModelPath
                      << " script: " << phase2ScriptPath << std::endl;

            bool phase2Init = phase2ProcessManager->initialize(
                pythonPath, phase2ScriptPath, phase2ModelPath, phase2TokenizerPath);

            if (!phase2Init)
            {
                std::cerr << "第二阶段神经网络初始化失败" << std::endl;
                phase1ProcessManager->stopProcess(); // 关闭第一阶段进程
                return false;
            }

            std::cout << "NeuralHeuristicSearch::initialize phase2 initialize ok" << std::endl;
            return true;
        }

        // 保留向后兼容的简化版初始化方法，但需要进行调整以支持双阶段
        bool initialize(const std::string &pythonPath, const std::string &scriptPath,
                        const std::string &modelPath, const std::string &tokenizerPath)
        {
            // 只初始化第一阶段
            bool phase1Init = phase1ProcessManager->initialize(
                pythonPath, scriptPath, modelPath, tokenizerPath);

            // 尝试自动推导第二阶段模型路径并初始化第二阶段
            if (phase1Init)
            {
                std::string phase2ScriptPath = scriptPath;

                // 检查是否包含目录分隔符
                size_t lastSlash = scriptPath.find_last_of("/\\");
                if (lastSlash != std::string::npos)
                {
                    // 替换文件名部分为neural_server2.py
                    phase2ScriptPath = scriptPath.substr(0, lastSlash + 1) + "neural_server2.py";
                }
                else
                {
                    // 没有目录分隔符，直接用neural_server2.py
                    phase2ScriptPath = "neural_server2.py";
                }

                // 尝试找到第二阶段模型路径
                std::string phase2ModelPath = modelPath;
                if (modelPath.find("first_stage") != std::string::npos)
                {
                    phase2ModelPath = modelPath;
                    phase2ModelPath.replace(phase2ModelPath.find("first_stage"), 11, "second_stage");
                }

                std::cout << "尝试自动初始化第二阶段，脚本: " << phase2ScriptPath
                          << "，模型: " << phase2ModelPath << std::endl;

                bool phase2Init = phase2ProcessManager->initialize(
                    pythonPath, phase2ScriptPath, phase2ModelPath, tokenizerPath);

                if (!phase2Init)
                {
                    std::cout << "警告：无法自动初始化第二阶段神经网络，第二阶段将使用默认评分" << std::endl;
                }
            }

            return phase1Init;
        }

        // 使用神经网络启发式的DFS搜索算法
        bool search(KnowledgeBase &kb, const Clause &goal, int max_iterations = 10000)
        {
            // 初始化
            auto initialTree = std::make_shared<SLITree>(kb);

            // 创建初始操作状态
            auto initial_state = SLIOperation::createExtensionState(
                initialTree,
                initialTree->getRoot(),
                Literal(),
                goal);

            // 创建栈用于DFS搜索
            std::stack<std::shared_ptr<SLIOperation::OperationState>> state_stack;
            state_stack.push(initial_state);

            visited_states.clear();
            visited_states.insert(initialTree->computeStateHash());

            int iteration = 0;
            std::shared_ptr<SLIOperation::OperationState> last_state = initial_state;

            std::cout << "开始搜索，实验模式: " << getExperimentMode() << std::endl;

            while (!state_stack.empty() && iteration < max_iterations)
            {
                iteration++;
                if (iteration % 100 == 0)
                {
                    std::cout << "搜索迭代 " << iteration << "，栈大小: " << state_stack.size() << std::endl;
                }

                // 取出栈顶状态
                auto current_state = state_stack.top();
                state_stack.pop();

                // 深拷贝当前状态以避免影响栈中的其他状态
                auto copied_state = SLIOperation::deepCopyOperationState(current_state);
                last_state = copied_state;

                // 执行当前操作
                bool valid_operation = applyOperation(copied_state);
                if (!valid_operation)
                {
                    continue; // 跳过无效操作
                }

                // 检查是否找到解
                if (checkEmptyClause(copied_state->sli_tree))
                {
                    std::cout << "成功找到解决方案，迭代次数: " << iteration << std::endl;
                    // SLIOperation::printOperationPath(copied_state, kb);
                    return true;
                }

                // 验证状态合法性
                if (!copied_state->sli_tree->validateAllNodes())
                {
                    continue;
                }

                // 计算状态哈希避免重复访问
                size_t state_hash = copied_state->sli_tree->computeStateHash();
                if (visited_states.find(state_hash) != visited_states.end())
                {
                    continue;
                }
                visited_states.insert(state_hash);

                // 获取当前状态下的节点
                auto b_lit_nodes = copied_state->sli_tree->get_all_B_literals();
                auto active_nodes = copied_state->sli_tree->get_all_active_nodes();
                bool AC_result = copied_state->sli_tree->check_all_nodes_AC();
                bool MC_result = copied_state->sli_tree->check_all_nodes_MC();

                // 根据条件决定可执行的操作类型
                std::vector<SLIActionType> available_actions;

                if (AC_result && MC_result)
                {
                    // 可以执行所有操作
                    available_actions = {
                        SLIActionType::EXTENSION,
                        SLIActionType::FACTORING,
                        SLIActionType::ANCESTRY,
                        SLIActionType::TRUNCATE};
                }
                else if (MC_result)
                {
                    // 只能执行Factoring和Ancestry
                    available_actions = {
                        SLIActionType::FACTORING,
                        SLIActionType::ANCESTRY};
                }
                else if (AC_result)
                {
                    // 只能执行Truncate
                    available_actions = {
                        SLIActionType::TRUNCATE};
                }
                else
                {
                    // 没有可执行的操作
                    continue;
                }

                // 获取神经网络对不同操作类型的评分
                json action_scores_json = getActionScores(copied_state->sli_tree, kb);

                if (action_scores_json.contains("status") && action_scores_json["status"] != "success")
                {
                    std::cerr << "获取操作评分失败，使用默认评分" << std::endl;
                    // 默认评分
                    action_scores_json = {
                        {"status", "success"},
                        {"action_scores", {0.6, 0.7, 0.8, 0.5}} // Extension, Factoring, Ancestry, Truncate
                    };
                }

                std::vector<float> action_scores = action_scores_json["action_scores"];
                if (action_scores.size() < 4)
                {
                    // 确保有足够的评分
                    while (action_scores.size() < 4)
                    {
                        action_scores.push_back(0.5);
                    }
                }

                // 将每种操作类型和对应分数配对
                std::vector<std::pair<SLIActionType, float>> scored_actions;
                for (const auto &action : available_actions)
                {
                    int action_idx = 0;
                    switch (action)
                    {
                    case SLIActionType::EXTENSION:
                        action_idx = 0;
                        break;
                    case SLIActionType::FACTORING:
                        action_idx = 1;
                        break;
                    case SLIActionType::ANCESTRY:
                        action_idx = 2;
                        break;
                    case SLIActionType::TRUNCATE:
                        action_idx = 3;
                        break;
                    }

                    if (action_idx < static_cast<int>(action_scores.size()))
                    {
                        scored_actions.push_back({action, action_scores[action_idx]});
                    }
                }

                // 按分数降序排序操作
                std::sort(scored_actions.begin(), scored_actions.end(),
                          [](const auto &a, const auto &b)
                          { return a.second > b.second; });

                // 为每种操作类型生成并评分所有可能的状态
                for (const auto &[action, score] : scored_actions)
                {
                    std::vector<std::shared_ptr<SLIOperation::OperationState>> action_states;

                    // 根据操作类型生成对应的状态集合
                    switch (action)
                    {
                    case SLIActionType::EXTENSION:
                        action_states = generateExtensionStates(kb, b_lit_nodes, copied_state);
                        break;
                    case SLIActionType::FACTORING:
                        action_states = generateFactoringStates(b_lit_nodes, copied_state);
                        break;
                    case SLIActionType::ANCESTRY:
                        action_states = generateAncestryStates(b_lit_nodes, copied_state);
                        break;
                    case SLIActionType::TRUNCATE:
                        action_states = generateTruncateStates(active_nodes, copied_state);
                        break;
                    }

                    // 如果没有可能的状态，继续下一个操作类型
                    if (action_states.empty())
                    {
                        continue;
                    }

                    // 第二阶段：使用神经网络对参数进行评分
                    std::vector<double> parameter_scores;
                    if (action_states.size() > 1) // 只有多个参数才需要评分
                    {
                        json parameter_scores_json = getParameterScores(action, action_states, kb);

                        if (parameter_scores_json.contains("status") && parameter_scores_json["status"] == "success")
                        {
                            parameter_scores = parameter_scores_json["parameter_scores"].get<std::vector<double>>();

                            // 确保评分数量匹配
                            if (parameter_scores.size() != action_states.size())
                            {
                                std::cerr << "参数评分数量不匹配，期望 " << action_states.size()
                                          << " 但收到 " << parameter_scores.size() << std::endl;

                                // 重置为默认评分
                                parameter_scores.clear();
                                for (size_t i = 0; i < action_states.size(); i++)
                                {
                                    parameter_scores.push_back(1.0 - (static_cast<double>(i) / action_states.size()));
                                }
                            }
                            else{
                                // std::cout << "NeuralHeuristicSearch::search parameter_scores ok parameter_scores.size() " << parameter_scores.size() <<std::endl;
                            }
                        }
                        else
                        {
                            // 默认评分
                            for (size_t i = 0; i < action_states.size(); i++)
                            {
                                parameter_scores.push_back(1.0 - (static_cast<double>(i) / action_states.size()));
                            }
                        }

                        // 创建状态和评分的配对
                        std::vector<std::pair<std::shared_ptr<SLIOperation::OperationState>, double>> state_scores;
                        for (size_t i = 0; i < action_states.size(); i++)
                        {
                            // 根据实验模式组合评分
                            double combined_score = 0.0;

                            switch (experiment_mode)
                            {
                            case PHASE1_ONLY:
                                combined_score = score; // 只使用第一阶段评分
                                break;
                            case PHASE2_ONLY:
                                combined_score = parameter_scores[i]; // 只使用第二阶段评分
                                break;
                            case BOTH_PHASES:
                                // 使用两个阶段的加权组合评分
                                combined_score = 0.5 * score + 0.5 * parameter_scores[i];
                                break;
                            }

                            action_states[i]->heuristic_score = combined_score;
                            state_scores.push_back({action_states[i], combined_score});
                        }

                        // 按评分降序排序（高评分优先）
                        std::sort(state_scores.begin(), state_scores.end(),
                                  [](const auto &a, const auto &b)
                                  {
                                      return a.second > b.second;
                                  });

                        // 清空并重新设置已排序的动作状态
                        action_states.clear();
                        for (const auto &[state, _] : state_scores)
                        {
                            action_states.push_back(state);
                        }
                    }
                    else if (action_states.size() == 1)
                    {
                        // 只有一个参数，直接使用操作类型评分
                        action_states[0]->heuristic_score = score;
                    }

                    // 反向入栈（确保高分的在栈顶）
                    for (int i = action_states.size() - 1; i >= 0; i--)
                    {
                        state_stack.push(action_states[i]);
                    }
                }
            }

            // 未找到解决方案
            std::cout << "未找到解决方案，达到最大迭代次数或搜索空间已耗尽" << std::endl;
            if (iteration >= max_iterations)
            {
                std::cout << "达到最大迭代次数: " << max_iterations << std::endl;
            }
            if (last_state)
            {
                std::cout << "最后的状态: " << std::endl;
                last_state->sli_tree->print_tree(kb);
                bool AC_result = last_state->sli_tree->check_all_nodes_AC();
                bool MC_result = last_state->sli_tree->check_all_nodes_MC();
                std::cout << "AC: " << AC_result << ", MC: " << MC_result << std::endl;
            }

            return false;
        }
    };
}

#endif // NEURAL_HEURISTIC_SEARCH_H