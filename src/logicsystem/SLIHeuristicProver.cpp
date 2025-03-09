#include "SLIHeuristicProver.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <string>
#include <filesystem>

#include "SLIMCTSState.h"
#include "SLIMCTSAction.h"
#include "DataCollector.h"
#include "NeuralHeuristicSearch.h"

namespace LogicSystem
{
    bool checkEmptyClauseNeural(const SLITree &sli_tree)
    {
        return sli_tree.get_all_active_nodes().size() == 0;
    }

    SLIHeuristicProver::SLIHeuristicProver(KnowledgeBase &kb, const Clause &goal)
        : kb(kb), goal(goal)
    {
    }

    bool SLIHeuristicProver::prove(const std::string &save_dir)
    {
        // 1. 初始化神经网络服务
        NeuralHeuristicSearch heuristicSearch;
        if (!heuristicSearch.initialize(pythonPath, scriptPath, modelPath, tokenizerPath)) {
            std::cerr << "初始化神经网络服务失败" << std::endl;
            return false;
        }

        // 2. 构造初始状态
        auto initialTree = std::make_shared<SLITree>(kb);
        initialTree->add_node(goal, Literal(), false, initialTree->getRoot());
        LogicSystem::SLIMCTSState current_state(initialTree);
        current_state.sli_tree = initialTree;

        // 3. 初始化数据收集容器
        std::vector<json> training_samples;
        DataCollector::NormalizationContext ctx;
        nlohmann::json graph_json = DataCollector::serializeGraph(kb, ctx);

        // 4. 迭代执行启发式搜索过程
        int max_iterations = 2000;  // 设置最大迭代次数
        int iteration = 0;
        
        while (!checkEmptyClauseNeural(*(current_state.sli_tree)) && !current_state.is_terminal() && iteration < max_iterations)
        {
            // 使用神经网络启发式获取最佳动作
            LogicSystem::SLIMCTSAction best_action = heuristicSearch.findBestAction(current_state, kb);
            return false;
            
            // // 如果没有可用动作，终止搜索
            // if (best_action.action == SLIActionType::INVALID) {
            //     std::cout << "没有可用的动作" << std::endl;
            //     break;
            // }
            
            // // 收集训练样本
            // json sample = DataCollector::serializeOperationMCTS(best_action, current_state, kb, ctx);
            // training_samples.push_back(sample);

            // 应用动作并更新状态
            current_state = current_state.next_state(best_action);
            std::cout << "迭代 " << iteration << " - 更新状态: " << current_state.to_string() << std::endl;
            
            iteration++;
        }

        // 5. 检查证明结果并保存数据
        bool is_success = checkEmptyClauseNeural(*(current_state.sli_tree));
        
        if (is_success)
        {
            std::cout << "证明成功!" << std::endl;
        }
        else
        {
            std::cout << "证明失败，迭代 " << iteration << " 次后" << std::endl;
            current_state.sli_tree->print_tree(kb);
            bool AC_result = current_state.sli_tree->check_all_nodes_AC();
            bool MC_result = current_state.sli_tree->check_all_nodes_MC();
            std::cout << "AC " << AC_result << " MC " << MC_result << std::endl;
            std::vector<SLIMCTSAction> actions;
            current_state.get_actions(actions);
            std::cout << "action size " << actions.size() << std::endl;
            std::cout << "has selfloop " << !current_state.sli_tree->validateAllNodes() << std::endl;
        }
        
        // 保存数据
        if (!save_dir.empty())
        {
            static int test_counter = 0;
            std::string fileName = save_dir;
            if (!fileName.empty() && fileName.back() != '/' && fileName.back() != '\\')
            {
                fileName += "/";
            }
            fileName += "heuristic_search_data_" + std::to_string(test_counter++) + 
                       (is_success ? "_success" : "_failure") + ".json";

            std::filesystem::create_directories(save_dir);

            nlohmann::json sample;
            sample["graph"] = graph_json;
            sample["search_path"] = training_samples;
            sample["proof_result"] = is_success ? "success" : "failure";
            sample["iterations"] = iteration;
            
            DataCollector::saveToFile(sample, fileName);
            std::cout << "保存训练样本到文件: " << fileName << std::endl;
        }
        
        return is_success;
    }

    void SLIHeuristicProver::setConfig(const std::string& python, const std::string& script, 
                                     const std::string& model, const std::string& tokenizer) {
        pythonPath = python;
        scriptPath = script;
        modelPath = model;
        tokenizerPath = tokenizer;
    }
}