#include "SLIMCTSProver.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <string>     // 用于 std::string 和 std::to_string
#include <filesystem> // C++17 文件系统

#include "SLIMCTSState.h"
#include "SLIMCTSAction.h"
#include "ofxMSAmcts.h"
#include "MSALoopTimer.h"
#include "DataCollector.h"

namespace LogicSystem
{
    bool checkEmptyClause(const SLITree &sli_tree)
    {
        return sli_tree.get_all_active_nodes().size() == 0;
    }

    SLIMCTSProver::SLIMCTSProver(KnowledgeBase &kb, const Clause &goal)
        : kb(kb), goal(goal)
    {
    }

    // 修改 prove 函数接口，传入文件保存路径。如果传入空字符串，则不保存数据。
    bool SLIMCTSProver::prove(const std::string &save_dir)
    {
        // 1. 构造初始状态
        auto initialTree = std::make_shared<SLITree>(kb);
        initialTree->add_node(goal, Literal(), false, initialTree->getRoot());
        LogicSystem::SLIMCTSState current_state(initialTree);
        current_state.sli_tree = initialTree;

        // 2. 配置 MCTS 搜索
        msa::mcts::UCT<LogicSystem::SLIMCTSState, LogicSystem::SLIMCTSAction> mcts_search;
        mcts_search.max_iterations = 10000;
        mcts_search.max_millis = 10000;
        mcts_search.simulation_depth = 2000;
        mcts_search.uct_k = std::sqrt(2);

        // 3. 初始化数据收集容器
        std::vector<json> training_samples;
        // 生成图与搜索路径的 JSON 数据
        DataCollector::NormalizationContext ctx; // 新鲜的上下文：保证整个样本中常量编码一致
        nlohmann::json graph_json = DataCollector::serializeGraph(kb, ctx);

        // 4. 迭代执行 MCTS 搜索过程
        while (!checkEmptyClause(*(current_state.sli_tree)) && !current_state.is_terminal())
        {
            // 执行一次 MCTS 搜索
            auto mcts_result = mcts_search.run(current_state);
            auto node = mcts_result.root_node;

            // 通过 DataCollector 收集训练样本
            json sample = DataCollector::collectTrainingSampleMCTS(node, kb, ctx);
            training_samples.push_back(sample);

            // 获取最佳动作并更新状态
            LogicSystem::SLIMCTSAction best_action = mcts_result.best_action;
            current_state = current_state.next_state(best_action);
            std::cout << "Updated State: " << current_state.to_string() << std::endl;
        }

        // 5. 检查证明结果并保存数据（当且仅当传入的保存路径不为空）
        bool is_success = checkEmptyClause(*(current_state.sli_tree));
        
        if (is_success)
        {
            std::cout << "Proof successful!" << std::endl;
        }
        else
        {
            std::cout << "Proof failed." << std::endl;
            current_state.sli_tree->print_tree(kb);
            bool AC_result = current_state.sli_tree->check_all_nodes_AC();
            bool MC_result = current_state.sli_tree->check_all_nodes_MC();
            std::cout << "AC " << AC_result << " MC " << MC_result << std::endl;
            std::vector<SLIMCTSAction> actions;
            current_state.get_actions(actions);
            std::cout << "action size " << actions.size() << std::endl;
            std::cout << "has selfloop " << !current_state.sli_tree->validateAllNodes() << std::endl;
        }
        
        // 无论成功还是失败，都保存数据
        if (!save_dir.empty())
        {
            // 生成唯一文件名：使用静态计数器生成不同的文件名
            static int test_counter = 0;
            std::string fileName = save_dir;
            // 确保目录路径以斜杠结尾
            if (!fileName.empty() && fileName.back() != '/' && fileName.back() != '\\')
            {
                fileName += "/";
            }
            // 在文件名中标识成功或失败
            fileName += "training_data_" + std::to_string(test_counter++) + 
                       (is_success ? "_success" : "_failure") + ".json";

            // 如果需要，可以利用 <filesystem> 检查或创建目录
            std::filesystem::create_directories(save_dir);

            // 保存收集的样本数据到文件
            nlohmann::json sample;
            sample["graph"] = graph_json;
            sample["search_path"] = training_samples;
            // 添加证明结果标记
            sample["proof_result"] = is_success ? "success" : "failure";
            
            DataCollector::saveToFile(sample, fileName);
            std::cout << "Saved training samples to file: " << fileName << std::endl;
        }
        
        return is_success;
    }
}