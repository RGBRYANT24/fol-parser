// NeuralHeuristicSearch.h
#ifndef NEURAL_HEURISTIC_SEARCH_H
#define NEURAL_HEURISTIC_SEARCH_H

#include <string>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include "ProcessManager.h"
#include "DataCollector.h"
#include "SLIMCTSState.h"
#include "SLIMCTSAction.h"
#include "KnowledgeBase.h"

using json = nlohmann::json;

namespace LogicSystem
{
    class NeuralHeuristicSearch
    {
    private:
        std::unique_ptr<ProcessManager> processManager;
        
    public:
        NeuralHeuristicSearch() : processManager(std::make_unique<ProcessManager>()) {}
        
        ~NeuralHeuristicSearch() {
            if (processManager->isActive()) {
                processManager->stopProcess();
            }
        }
        
        // 初始化神经网络服务
        bool initialize(const std::string& pythonPath, const std::string& scriptPath,
                       const std::string& modelPath, const std::string& tokenizerPath) {
            return processManager->initialize(pythonPath, scriptPath, modelPath, tokenizerPath);
        }
        
        // 获取神经网络提供的启发式评分
        json getHeuristic(const SLIMCTSState& state, const KnowledgeBase& kb) {
            if (!processManager->isActive()) {
                return {{"status", "error"}, {"error_message", "神经网络服务未初始化"}};
            }
            
            // 使用DataCollector直接将状态序列化为JSON
            DataCollector::NormalizationContext ctx;
            json state_json = DataCollector::serializeTree(*state.sli_tree, kb, ctx);
            json graph_json = DataCollector::serializeGraph(kb, ctx);
            
            // 构建用于神经网络的请求
            json request = {
                {"state", {{"tree", state_json}}},
                {"graph", graph_json}
            };
            
            // 发送请求并获取响应
            json response = processManager->sendRequest(request);
            
            if (response.contains("status") && response["status"] == "error") {
                std::cerr << "神经网络服务返回错误: " << response["error_message"] << std::endl;
                if (response.contains("error_details")) {
                    std::cerr << "错误详情: " << response["error_details"] << std::endl;
                }
            }
            
            return response;
        }
        
        // 启发式搜索算法：找到最佳动作
        SLIMCTSAction findBestAction(SLIMCTSState& current_state, const KnowledgeBase& kb) {
            std::vector<SLIMCTSAction> actions;
            current_state.get_actions(actions);
            
            if (actions.empty()) {
                return SLIMCTSAction(); // 返回一个无效动作表示没有可用动作
            }
            
            // 获取神经网络的启发式评分
            json heuristic_data = getHeuristic(current_state, kb);
            
            if (heuristic_data.contains("status") && heuristic_data["status"] != "success") {
                std::cerr << "获取启发式信息错误: " << heuristic_data["error_message"] << std::endl;
                // 回退到简单策略：选择第一个动作
                std::cout << "使用回退策略：选择第一个可用动作" << std::endl;
                return actions[0];
            }
            
            std::vector<float> action_scores = heuristic_data["action_scores"];
            
            // 根据动作类型和启发式分数选择最佳动作
            SLIMCTSAction best_action;
            float best_score = -std::numeric_limits<float>::infinity();
            
            for (const auto& action : actions) {
                // 获取当前动作的类型（Extension=0, Factoring=1, Ancestry=2）
                int action_type;
                if (action.action == SLIAction::EXTENSION) {
                    action_type = 0;
                } else if (action.action == SLIAction::FACTORING) {
                    action_type = 1;
                } else if (action.action == SLIAction::ANCESTRY) {
                    action_type = 2;
                } else {
                    // 不支持的操作类型
                    continue;
                }
                
                // 确保索引有效
                if (action_type >= 0 && action_type < static_cast<int>(action_scores.size())) {
                    // 使用神经网络提供的分数作为启发式
                    float score = action_scores[action_type];
                    
                    if (score > best_score) {
                        best_score = score;
                        best_action = action;
                    }
                }
            }
            
            // 如果没有找到有效动作，默认使用第一个
            if (best_action.action == SLIAction::INVALID && !actions.empty()) {
                std::cout << "未找到最佳动作，使用第一个可用动作" << std::endl;
                best_action = actions[0];
            }
            
            return best_action;
        }
    };
}

#endif // NEURAL_HEURISTIC_SEARCH_H