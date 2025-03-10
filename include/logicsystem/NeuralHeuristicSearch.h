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
        std::unique_ptr<ProcessManager> processManager;

        // 使用哈希集合跟踪已访问状态以避免循环
        std::unordered_set<size_t> visited_states;

        // 获取第一阶段神经网络对操作类型的评分
        json getActionScores(const std::shared_ptr<SLITree> &tree, KnowledgeBase &kb)
        {
            if (!processManager->isActive())
            {
                return {{"status", "error"}, {"error_message", "神经网络服务未初始化"}};
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
            json response = processManager->sendRequest(request);

            if (response.contains("status") && response["status"] == "error")
            {
                std::cerr << "神经网络服务返回错误: " << response["error_message"] << std::endl;
                if (response.contains("error_details"))
                {
                    std::cerr << "错误详情: " << response["error_details"] << std::endl;
                }
            }

            return response;
        }

        // 第二阶段：获取神经网络对特定操作的参数评分（预留接口）
        json getParameterScores(SLIActionType actionType,
                                const std::shared_ptr<SLINode> &node,
                                const std::shared_ptr<SLITree> &tree,
                                KnowledgeBase &kb)
        {
            // 目前返回默认评分，未来将由神经网络提供
            return {{"status", "success"}, {"parameter_scores", json::array()}};
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
                            std::cout << "NeuralHeuristicSearch::generateExtensionStates  new states" << std::endl;
                            SLIOperation::printCurrentState(new_state, kb);
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
        NeuralHeuristicSearch() : processManager(std::make_unique<ProcessManager>()) {}

        ~NeuralHeuristicSearch()
        {
            if (processManager->isActive())
            {
                processManager->stopProcess();
            }
        }

        // 初始化神经网络服务
        bool initialize(const std::string &pythonPath, const std::string &scriptPath,
                        const std::string &modelPath, const std::string &tokenizerPath)
        {
            return processManager->initialize(pythonPath, scriptPath, modelPath, tokenizerPath);
        }

        // 使用神经网络启发式的DFS搜索算法
        bool search(KnowledgeBase &kb, const Clause &goal, int max_iterations = 10000)
        {
            // 初始化
            auto initialTree = std::make_shared<SLITree>(kb);
            // initialTree->add_node(goal, Literal(), false, initialTree->getRoot());

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

                // std::cout << "current state round " << iteration << std::endl;
                // SLIOperation::printCurrentState(copied_state, kb);

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
                    SLIOperation::printOperationPath(copied_state, kb);
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
                // std::cout << "NeuralHeuristicSeach::search AC " << AC_result << " MC " << MC_result << std::endl;

                // 根据条件决定可执行的操作类型
                std::vector<SLIActionType> available_actions;

                if (AC_result && MC_result)
                {
                    // 可以执行所有操作
                    // std::cout << "NeuralHeuristicSeach::search AC&&MC is true " << std::endl;
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
                        {"action_scores", {0.7, 0.5, 0.3, 0.1}} // Extension, Factoring, Ancestry, Truncate
                    };
                }

                std::vector<float> action_scores = action_scores_json["action_scores"];
                std::cout << "第一阶段神经网络操作评分: ";
                for (auto score : action_scores)
                {
                    std::cout << score << " ";
                }
                std::cout << std::endl;

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

                // 为每种操作类型生成可能的状态
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

                    // 为这种操作类型下的每个参数选择添加评分
                    for (auto &state : action_states)
                    {
                        state->heuristic_score = score;
                    }

                    // 第二阶段：对参数进行排序（预留给神经网络）
                    // 目前使用简单的降序入栈（高分先入栈）

                    // 反向入栈（确保高分的在栈顶）
                    for (int i = action_states.size() - 1; i >= 0; i--)
                    {
                        state_stack.push(action_states[i]);
                        // std::cout << "potential action states " << std::endl;
                        // SLIOperation::printCurrentState(action_states[i], kb);
                    }
                }
            }

            // 未找到解决方案
            std::cout << "未找到解决方案，达到最大迭代次数或搜索空间已耗尽" << std::endl;
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