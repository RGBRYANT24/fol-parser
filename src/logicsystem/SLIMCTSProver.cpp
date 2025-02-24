#include "SLIMCTSProver.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

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

    bool SLIMCTSProver::prove()
    {
        // 1. Construct initial state
        auto initialTree = std::make_shared<SLITree>(kb);
        initialTree->add_node(goal, Literal(), false, initialTree->getRoot());
        LogicSystem::SLIMCTSState current_state(initialTree);
        current_state.sli_tree = initialTree;

        // 2. Configure MCTS search
        msa::mcts::UCT<LogicSystem::SLIMCTSState, LogicSystem::SLIMCTSAction> mcts_search;
        mcts_search.max_iterations = 3000;
        mcts_search.max_millis = 3000;
        mcts_search.simulation_depth = 1000;
        mcts_search.uct_k = std::sqrt(2);

        // 3. Initialize data collection
        std::vector<json> training_samples;

        // 4. Run MCTS iteratively
        while (!checkEmptyClause(*(current_state.sli_tree)) && !current_state.is_terminal())
        {
            // Perform MCTS search
            auto mcts_result = mcts_search.run(current_state);
            auto node = mcts_result.root_node;

            // Collect training sample using DataCollect
            json sample = DataCollector::collectTrainingSampleMCTS(node, kb);
            training_samples.push_back(sample);
            std::cout << "SLIMCTSProver::prove get training sample node by node " << std::endl;
            std::cout << sample << std::endl;

            // get reward
            auto reward_sample = DataCollector::computeExpectedOpRewards(node);
            std::cout << "SLIMCTSProver::prove get training sample reward " << std::endl;
            std::cout << reward_sample << std::endl;


            // // Get and apply the best action
            // LogicSystem::SLIMCTSAction best_action = mcts_search.get_best_uct_child(); // Assumes this exists
            LogicSystem::SLIMCTSAction best_action = mcts_result.best_action;
            current_state = current_state.next_state(best_action);
            std::cout << "Updated State: " << current_state.to_string() << std::endl;
        }

        // 5. Check proof result and save data if successful
        if (checkEmptyClause(*(current_state.sli_tree)))
        {
            std::cout << "Proof successful!" << std::endl;
            // Save collected samples to file
            DataCollector::saveToFile(training_samples, "training_data.json");
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
            std::cout << "has selfloop " << !current_state.sli_tree->validateAllNodes() << std::endl;
            return false;
        }
    }

    void SLIMCTSProver::collectTrainingSamples(const LogicSystem::SLIMCTSState &state)
    {
        // Placeholder implementation remains unchanged
        std::vector<json> training_samples;
        std::string state_str = state.to_string();
        std::cout << "Collecting training sample for state: " << state_str << std::endl;
        DataCollector::saveToFile(training_samples, "training_samples.json");
    }
}