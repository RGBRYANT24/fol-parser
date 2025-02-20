#include "SLIMCTSProver.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

// 包含相关模块的头文件（假设这些头文件均已正确定义）
#include "SLIMCTSState.h"  // 提供 LogicSystem::SLIMCTSState 定义
#include "SLIMCTSAction.h" // 提供 LogicSystem::SLIMCTSAction 定义
#include "ofxMSAmcts.h"    // MCTS 框架中的 UCT 算法实现
#include "MSALoopTimer.h"  // 定时器
#include "DataCollector.h" // 数据采集接口

namespace LogicSystem
{
    // 假定 SLITree 类型是在 SLIMCTSState.h 中定义
    // 辅助函数：判断 SLITree 是否包含空子句（证明是否成功）
    bool checkEmptyClause(const SLITree &sli_tree)
    {
        // 根据 SLITree 的实际实现编写具体空子句判断逻辑
        return sli_tree.get_all_active_nodes().size() == 0 ? true : false;
    }

    SLIMCTSProver::SLIMCTSProver(KnowledgeBase &kb, const Clause &goal)
        : kb(kb), goal(goal)
    {
    }

    bool SLIMCTSProver::prove()
    {
        // ----------------------------
        // 1. 构造初始状态
        // ----------------------------
        auto initialTree = std::make_shared<SLITree>(kb);
        initialTree->add_node(goal, Literal(), false, initialTree->getRoot());
        LogicSystem::SLIMCTSState current_state(initialTree);
        // 此处直接将初始 tree 赋值给状态（如果有必要可以省略，因为构造函数已经深拷贝）
        current_state.sli_tree = initialTree;

        // ----------------------------
        // 2. 配置 MCTS 搜索
        // ----------------------------
        msa::mcts::UCT<LogicSystem::SLIMCTSState, LogicSystem::SLIMCTSAction> mcts_search;
        mcts_search.max_iterations = 15000;  // 最大迭代次数
        mcts_search.max_millis = 3000;    // 最大搜索时间（毫秒）
        mcts_search.simulation_depth = 400; // 模拟阶段的最大深度
        mcts_search.uct_k = std::sqrt(6);  // UCT 中的探索系数

        // ----------------------------
        // 3. 逐步扩展证明过程
        // ----------------------------
        // 通过循环不断执行动作直到达到证明目的或判断为终局状态
        int count = 0;
        while (!checkEmptyClause(*(current_state.sli_tree)) && !current_state.is_terminal())
        {
            // if(count ++ > 2) return false;
            // 使用 MCTS 搜索来选择一条最佳的动作路径
            LogicSystem::SLIMCTSAction best_action = mcts_search.run(current_state);

            // std::cout << "Current State:" << std::endl;
            // current_state.sli_tree->print_tree(kb);
            // std::cout << "Best Action: " << best_action.to_string(kb) << std::endl;

            // return false;

            // 生成新状态，该状态为当前状态的深拷贝，并在其上应用动作
            current_state = current_state.next_state(best_action);

            // 可选：采集训练样本
            // collectTrainingSamples(current_state);

            std::cout << "Updated State: " << current_state.to_string() << std::endl;
        }

        // ----------------------------
        // 4. 检查证明结果
        // ----------------------------
        if (checkEmptyClause(*(current_state.sli_tree)))
        {
            std::cout << "Proof successful!" << std::endl;
            return true;
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
            std::cout << "has selfloop " << !current_state.sli_tree->validateAllNodes() <<std::endl;
            return false;
        }
    }

    void SLIMCTSProver::collectTrainingSamples(const LogicSystem::SLIMCTSState &state)
    {
        // 示例：采集训练数据，具体实现需要根据实际数据结构调整
        std::vector<json> training_samples;
        std::string state_str = state.to_string();
        std::cout << "Collecting training sample for state: " << state_str << std::endl;

        // 以下代码仅为伪代码示例，实际请根据你的项目需求修改：
        /*
        std::vector<LogicSystem::SLIMCTSAction> available_actions;
        state.get_actions(available_actions);
        double reward = 1.0;  // 证明成功时奖励可以设为1.0
        json sample = DataCollector::collectTrainingSample(state, available_actions, chosen_action, reward, kb);
        training_samples.push_back(sample);
        */

        // 将采集到的样本存储到文件中
        DataCollector::saveToFile(training_samples, "training_samples.json");
    }
}
