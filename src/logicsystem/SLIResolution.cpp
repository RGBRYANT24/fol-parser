// SLIResolution.cpp
#include "SLIResolution.h"
#include <iostream>
#include <chrono>

namespace LogicSystem
{
    int ProofState::next_id = 0;

    SearchResult SLIResolution::prove(KnowledgeBase &kb, const Clause &goal)
    {
        // 初始化返回结构
        SearchResult result;
        result.method = "DFS"; // 设置搜索方法名称

        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();

        // 创建初始状态
        auto initialTree = std::make_shared<SLITree>(kb);

        // 创建初始操作状态
        auto initial_state = SLIOperation::createExtensionState(
            initialTree,
            initialTree->getRoot(), // 使用第一个节点作为起始节点
            Literal(),              // 空文字
            goal                    // 目标子句
        );

        // 搜索栈
        std::stack<std::shared_ptr<SLIOperation::OperationState>> state_stack;
        state_stack.push(initial_state);

        // 访问集合，用于存储已访问的状态哈希值
        std::unordered_set<size_t> visited_states;
        visited_states.insert(initialTree->computeStateHash());

        // 记录最长路径的信息
        int max_depth = 0;
        std::shared_ptr<SLIOperation::OperationState> max_depth_state = nullptr;

        std::shared_ptr<SLIOperation::OperationState> successful_state = nullptr;
        std::shared_ptr<SLIOperation::OperationState> last_state = nullptr;

        // 节点访问计数器
        int visited_states_count = 1; // 初始状态算作一个

        while (!state_stack.empty())
        {
            visited_states_count++;

            if (visited_states_count % 5000 == 0)
            {
                std::cout << "SearchResult SLIResolution::prove DFS round " << visited_states_count << std::endl;
            }

            auto current_state = state_stack.top();
            last_state = current_state;
            state_stack.pop();

            // 更新最长路径
            if (current_state->depth > max_depth)
            {
                max_depth = current_state->depth;
                max_depth_state = current_state;
            }

            std::shared_ptr<SLIOperation::OperationState> new_state;
            try
            {
                new_state = SLIOperation::deepCopyOperationState(current_state);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error during deep copy: " << e.what() << "\n";

                // 设置结果为失败
                result.success = false;
                result.visitedStates = visited_states_count;

                // 计算用时
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                result.duration = duration;

                std::cout << "证明失败，复制状态时出错。用时: " << duration
                          << " 毫秒, 访问状态数: " << visited_states_count << std::endl;

                return result;
            }

            if (visited_states_count >= 10000LL)
            {
                // SLIOperation::printOperationPath(last_state, kb);
                // SLIOperation::printOperationPathAsClause(last_state, kb);

                // 设置结果为失败（达到最大迭代次数）
                result.success = false;
                result.visitedStates = visited_states_count;

                // 计算用时
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                result.duration = duration;

                std::cout << "证明失败，达到最大迭代次数。用时: " << duration
                          << " 毫秒, 访问状态数: " << visited_states_count << std::endl;

                return result;
            }

            // 执行操作
            switch (new_state->action)
            {
            case SLIActionType::EXTENSION:
            {
                if (SLIOperation::isLiteral(new_state->second_op))
                {
                    auto kb_lit = SLIOperation::getLiteral(new_state->second_op);
                    auto new_nodes = new_state->sli_tree->add_node(
                        new_state->kb_clause,
                        kb_lit,
                        true,
                        new_state->lit1_node);
                }
                break;
            }
            case SLIActionType::FACTORING:
            {
                if (SLIOperation::isNode(new_state->second_op))
                {
                    auto second_node = SLIOperation::getNode(new_state->second_op);
                    new_state->sli_tree->t_factoring(new_state->lit1_node, second_node);
                }
                break;
            }
            case SLIActionType::ANCESTRY:
            {
                if (SLIOperation::isNode(new_state->second_op))
                {
                    auto second_node = SLIOperation::getNode(new_state->second_op);
                    new_state->sli_tree->t_ancestry(new_state->lit1_node, second_node);
                }
                break;
            }
            case SLIActionType::TRUNCATE:
            {
                new_state->sli_tree->truncate(new_state->lit1_node);
                break;
            }
            }

            // 检查空子句
            if (checkEmptyClause(*new_state->sli_tree))
            {
                // 设置结果为成功
                result.success = true;
                result.visitedStates = visited_states_count;

                // 计算用时
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                result.duration = duration;

                std::cout << "证明成功! 用时: " << duration << " 毫秒, 访问状态数: " << visited_states_count << std::endl;

                return result;
            }

            // 在执行操作后添加验证
            if (!new_state->sli_tree->validateAllNodes())
            {
                continue; // 跳过包含无效节点的状态
            }

            // 检查是否访问过
            size_t state_hash = new_state->sli_tree->computeStateHash();
            if (visited_states.find(state_hash) != visited_states.end())
            {
                continue;
            }
            else
            {
                visited_states.insert(state_hash);
            }

            // 基本条件检查
            bool AC_result = new_state->sli_tree->check_all_nodes_AC();
            bool MC_result = new_state->sli_tree->check_all_nodes_MC();

            auto b_lit_nodes = new_state->sli_tree->get_all_B_literals();

            // 收集当前可用的操作
            std::vector<std::shared_ptr<SLIOperation::OperationState>> available_ops;

            if (AC_result && MC_result)
            {
                // t-extension
                generateExtensionStates2(kb, b_lit_nodes, new_state, state_stack, available_ops);
                // t-factoring
                generateFactoringStates(b_lit_nodes, new_state, state_stack, available_ops);
                // t-ancestry
                generateAncestryStates(b_lit_nodes, new_state, state_stack, available_ops);
                // t-truncate
                generateTruncateStates(new_state->sli_tree->get_all_active_nodes(),
                                       new_state, state_stack, available_ops);
            }
            else if (MC_result)
            {
                // t-factoring
                generateFactoringStates(b_lit_nodes, new_state, state_stack, available_ops);
                // t-ancestry
                generateAncestryStates(b_lit_nodes, new_state, state_stack, available_ops);
            }
            else if (AC_result)
            {
                // t-truncate
                generateTruncateStates(new_state->sli_tree->get_all_active_nodes(),
                                       new_state, state_stack, available_ops);
            }
            else
            {
                continue;
            }

            // 记录父状态的可用操作到子状态中
            for (auto &op : available_ops)
            {
                op->parent_available_ops = available_ops;
            }
        }

        // 搜索完成但未找到解
        SLIOperation::printOperationPath(last_state, kb);
        std::cout << "last state is above" << std::endl;

        // 设置结果为失败（搜索空间已耗尽）
        result.success = false;
        result.visitedStates = visited_states_count;

        // 计算用时
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        result.duration = duration;

        std::cout << "证明失败，搜索空间已耗尽。用时: " << duration
                  << " 毫秒, 访问状态数: " << visited_states_count << std::endl;

        return result;
    }

    bool SLIResolution::prove(KnowledgeBase &kb, const Clause &goal, SearchStrategy &strategy)
    {

        // // 初始化返回结构
        // SearchResult result;
        // result.method = "DFS"; // 设置搜索方法名称
        // std::vector<json> training_samples;
        // 创建初始状态
        auto initialTree = std::make_shared<SLITree>(kb);

        // 创建初始操作状态
        auto initial_state = SLIOperation::createExtensionState(
            initialTree,
            initialTree->getRoot(), // 使用第一个节点作为起始节点
            Literal(),              // 空文字
            goal                    // 目标子句
        );

        // 搜索栈
        std::stack<std::shared_ptr<SLIOperation::OperationState>> state_stack;
        state_stack.push(initial_state);

        // 访问集合，用于存储已访问的状态哈希值
        std::unordered_set<size_t> visited_states;
        visited_states.insert(initialTree->computeStateHash());

        // 记录最长路径的信息
        int max_depth = 0;
        std::shared_ptr<SLIOperation::OperationState> max_depth_state = nullptr;

        std::shared_ptr<SLIOperation::OperationState> successful_state = nullptr;
        std::shared_ptr<SLIOperation::OperationState> last_state = nullptr;

        long long count = 0;
        while (!state_stack.empty())
        {
            count++;
            if (count % 5000 == 0)
            {
                std::cout << "round " << count << std::endl;
            }

            auto current_state = state_stack.top();
            last_state = current_state;
            state_stack.pop();
            // std::cout << current_state->state_id << " state id " << std::endl;

            // 更新最长路径
            if (current_state->depth > max_depth)
            {
                max_depth = current_state->depth;
                max_depth_state = current_state;
                // std::cout << "New maximum depth: " << max_depth << " at state id: " << current_state->state_id << std::endl;
            }

            std::shared_ptr<SLIOperation::OperationState> new_state;
            try
            {
                new_state = SLIOperation::deepCopyOperationState(current_state);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error during deep copy: " << e.what() << "\n";
                return false; // 根据需要决定是否跳过或终止
            }

            if (count >= 900000LL)
            {
                SLIOperation::printOperationPath(last_state, kb);
                // SLIOperation::printOperationPathAsClause(max_depth_state, kb);
                SLIOperation::printOperationPathAsClause(last_state, kb);
                return false;
            }
            // std::cout << "Current State " << std::endl;
            // SLIOperation::printOperationPath(new_state, kb);
            // std::cout << "Performing action " << SLIOperation::getActionString(new_state->action) << std::endl;

            // 执行操作
            switch (new_state->action)
            {
            case SLIActionType::EXTENSION:
            {
                if (SLIOperation::isLiteral(new_state->second_op))
                {
                    auto kb_lit = SLIOperation::getLiteral(new_state->second_op);
                    auto new_nodes = new_state->sli_tree->add_node(
                        new_state->kb_clause,
                        kb_lit,
                        true,
                        new_state->lit1_node);
                }
                break;
            }
            case SLIActionType::FACTORING:
            {
                if (SLIOperation::isNode(new_state->second_op))
                {
                    auto second_node = SLIOperation::getNode(new_state->second_op);
                    new_state->sli_tree->t_factoring(new_state->lit1_node, second_node);
                }
                break;
            }
            case SLIActionType::ANCESTRY:
            {
                if (SLIOperation::isNode(new_state->second_op))
                {
                    auto second_node = SLIOperation::getNode(new_state->second_op);
                    new_state->sli_tree->t_ancestry(new_state->lit1_node, second_node);
                }
                break;
            }
            case SLIActionType::TRUNCATE:
            {
                // std::cout << "Performing Truncate " << std::endl;
                new_state->sli_tree->truncate(new_state->lit1_node);
                break;
            }
            }

            // 检查空子句
            // std::cout << "Check Empty Clause " << std::endl;
            if (checkEmptyClause(*new_state->sli_tree))
            {
                // successful_state = new_state;
                // SLIOperation::printOperationPath(successful_state, kb);
                // std::vector<json> successful_samples;
                // int max_depth = successful_state->depth;

                // // 回溯路径收集样本
                // auto current_state = successful_state;
                // while (current_state && current_state->parent)
                // {
                //     // std::cout << "current state " << std::endl;
                //     // SLIOperation::printCurrentState(current_state, kb);
                //     auto parent_state = current_state->parent;

                //     // 创建新的标准化上下文
                //     LogicSystem::DataCollector::NormalizationContext ctx;

                //     // 确保指针有效
                //     if (parent_state->sli_tree == nullptr)
                //     {
                //         std::cerr << "Invalid parent state!" << std::endl;
                //         break;
                //     }

                //     double reward = 1.0 / (1.0 + max_depth - current_state->depth);

                //     // 获取父状态生成时的所有可能操作
                //     auto &available_ops = current_state->parent_available_ops;
                //     std::cout << "available ops size " << available_ops.size() << std::endl;

                //     // 确定当前操作是父状态选择的那个
                //     auto it = std::find(available_ops.begin(), available_ops.end(), current_state->copy_state);
                //     if (it != available_ops.end())
                //     {
                //         std::cout << "find op" << std::endl;
                //         // 收集样本时传入kb引用
                //         successful_samples.push_back(
                //             DataCollector::collectTrainingSample(
                //                 *parent_state,
                //                 available_ops,
                //                 *it,
                //                 reward, // reward
                //                 kb      // 传入KnowledgeBase
                //                 ));
                //     }
                //     current_state = parent_state;
                // }

                // // 将成功路径的样本添加到总数据中（逆序）
                // training_samples.insert(training_samples.end(), successful_samples.rbegin(), successful_samples.rend());
                // std::cout << "training_samples size " << training_samples.size() << std::endl;
                // DataCollector::saveToFile(training_samples, "/home/adrin/Projects/fol-parser/data/training_data.json");
                return true;
            }

            // 在执行操作后添加验证
            if (!new_state->sli_tree->validateAllNodes())
            {
                continue; // 跳过包含无效节点的状态
            }
            // 检查是否访问过
            size_t state_hash = new_state->sli_tree->computeStateHash();
            if (visited_states.find(state_hash) != visited_states.end())
            {
                // std::cout << "Skipping already visited state with hash: " << state_hash << std::endl;
                continue;
            }
            else
            {
                visited_states.insert(state_hash);
            }

            // 基本条件检查
            // std::cout << "Basic Condition Test " << std::endl;
            bool AC_result = new_state->sli_tree->check_all_nodes_AC();
            bool MC_result = new_state->sli_tree->check_all_nodes_MC();

            auto b_lit_nodes = new_state->sli_tree->get_all_B_literals();

            // 收集当前可用的操作
            // 要添加到各个generateStates函数里面去
            std::vector<std::shared_ptr<SLIOperation::OperationState>> available_ops;

            if (AC_result && MC_result)
            {
                // t-extension
                // generateExtensionStates(kb, b_lit_nodes, new_state, state_stack, available_ops);
                generateExtensionStates2(kb, b_lit_nodes, new_state, state_stack, available_ops);
                //  t-factoring
                generateFactoringStates(b_lit_nodes, new_state, state_stack, available_ops);
                // t-ancestry
                generateAncestryStates(b_lit_nodes, new_state, state_stack, available_ops);
                // t-truncate
                generateTruncateStates(new_state->sli_tree->get_all_active_nodes(),
                                       new_state, state_stack, available_ops);
            }

            else if (MC_result)
            {
                // std::cout << "Only MC Perform Factoring and Ancestry in round " << count << std::endl;
                // t-factoring
                generateFactoringStates(b_lit_nodes, new_state, state_stack, available_ops);
                // t-ancestry
                generateAncestryStates(b_lit_nodes, new_state, state_stack, available_ops);
            }

            else if (AC_result)
            {
                // std::cout << "Only AC Perform Truncate in round " << count << std::endl;
                // t-truncate
                generateTruncateStates(new_state->sli_tree->get_all_active_nodes(),
                                       new_state, state_stack, available_ops);
            }
            else
            {
                continue;
            }
            // 记录父状态的可用操作到子状态中
            for (auto &op : available_ops)
            {
                op->parent_available_ops = available_ops; // 关键修改
            }
            // std::cout << "add availalbe_ops size " << available_ops.size() << std::endl;
            // 检查是否是特定的 state id
            // if (new_state->state_id == 19)
            // {
            //     std::cout << "Processing State ID: " << new_state->state_id << std::endl;
            //     SLIOperation::printOperationPath(new_state, kb); // 打印当前状态路径
            //     auto top_state = state_stack.top();
            //     std::cout << "top action " << SLIOperation::getActionString(top_state->action) << std::endl;
            //     SLIOperation::printOperationPath(top_state, kb);
            //     return false;
            //     // 这里可以进一步检查或记录当前状态的详细信息
            // }
            // 收集训练样本
            // double reward = 1.0;
            // training_samples.push_back(
            //     DataCollector::collectTrainingSample(*new_state, available_ops, reward, kb));
        }
        // 保存训练数据
        SLIOperation::printOperationPath(last_state, kb);
        std::cout << "last state is above" << std::endl;
        // DataCollector::saveToFile(training_samples, "/path/to/failed_data.json");
        return false;
    }

    // SLIResolution.cpp

    // bool SLIResolution::proveBFS(KnowledgeBase &kb, const Clause &goal)
    // {
    //     auto defaultStrategy = BFSStrategy(2000, 60.0, 1024 * 1024 * 100);
    //     return proveBFS(kb, goal, defaultStrategy);
    // }

    bool SLIResolution::proveBFS(KnowledgeBase &kb, const Clause &goal, SearchStrategy &strategy)
    {
        auto initialTree = std::make_shared<SLITree>(kb);

        // 创建初始操作状态
        auto initial_state = SLIOperation::createExtensionState(
            initialTree,
            initialTree->getRoot(),
            Literal(),
            goal);

        std::queue<std::shared_ptr<SLIOperation::OperationState>> state_queue;
        state_queue.push(initial_state);

        std::unordered_set<size_t> visited_states;
        visited_states.insert(initialTree->computeStateHash());

        int max_depth = 0;
        std::shared_ptr<SLIOperation::OperationState> max_depth_state = nullptr;
        std::shared_ptr<SLIOperation::OperationState> successful_state = nullptr;
        std::shared_ptr<SLIOperation::OperationState> last_state = nullptr;

        long long count = 0;
        while (!state_queue.empty())
        {
            count++;
            if (count % 5000 == 0)
            {
                std::cout << "BFS round " << count << std::endl;
            }

            auto current_state = state_queue.front();
            last_state = current_state;
            state_queue.pop();

            if (current_state->depth > max_depth)
            {
                max_depth = current_state->depth;
                max_depth_state = current_state;
            }

            if (count >= 1000000000LL)
            {
                SLIOperation::printOperationPathAsClause(max_depth_state, kb);
                return false;
            }

            std::shared_ptr<SLIOperation::OperationState> new_state;
            try
            {
                new_state = SLIOperation::deepCopyOperationState(current_state);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error during deep copy: " << e.what() << "\n";
                return false;
            }

            // 执行操作
            switch (new_state->action)
            {
            case SLIActionType::EXTENSION:
            {
                if (SLIOperation::isLiteral(new_state->second_op))
                {
                    auto kb_lit = SLIOperation::getLiteral(new_state->second_op);
                    auto new_nodes = new_state->sli_tree->add_node(
                        new_state->kb_clause,
                        kb_lit,
                        true,
                        new_state->lit1_node);
                }
                break;
            }
            case SLIActionType::FACTORING:
            {
                if (SLIOperation::isNode(new_state->second_op))
                {
                    auto second_node = SLIOperation::getNode(new_state->second_op);
                    new_state->sli_tree->t_factoring(new_state->lit1_node, second_node);
                }
                break;
            }
            case SLIActionType::ANCESTRY:
            {
                if (SLIOperation::isNode(new_state->second_op))
                {
                    auto second_node = SLIOperation::getNode(new_state->second_op);
                    new_state->sli_tree->t_ancestry(new_state->lit1_node, second_node);
                }
                break;
            }
            case SLIActionType::TRUNCATE:
            {
                // std::cout << "Performing Truncate " << std::endl;
                new_state->sli_tree->truncate(new_state->lit1_node);
                break;
            }
            }

            if (checkEmptyClause(*new_state->sli_tree))
            {
                successful_state = new_state;
                SLIOperation::printOperationPath(successful_state, kb);
                return true;
            }

            if (!new_state->sli_tree->validateAllNodes())
            {
                continue;
            }

            auto b_lit_nodes = new_state->sli_tree->get_all_B_literals();
            bool AC_result = new_state->sli_tree->check_all_nodes_AC();
            bool MC_result = new_state->sli_tree->check_all_nodes_MC();

            if (AC_result && MC_result)
            {
                generateExtensionStatesBFS(kb, b_lit_nodes, new_state, state_queue);
                generateFactoringStatesBFS(b_lit_nodes, new_state, state_queue);
                generateAncestryStatesBFS(b_lit_nodes, new_state, state_queue);
                generateTruncateStatesBFS(new_state->sli_tree->get_all_active_nodes(),
                                          new_state, state_queue);
            }
            else if (MC_result)
            {
                generateFactoringStatesBFS(b_lit_nodes, new_state, state_queue);
                generateAncestryStatesBFS(b_lit_nodes, new_state, state_queue);
            }
            else if (AC_result)
            {
                generateTruncateStatesBFS(new_state->sli_tree->get_all_active_nodes(),
                                          new_state, state_queue);
            }
        }

        SLIOperation::printOperationPath(last_state, kb);
        return false;
    }

    bool SLIResolution::proveHeuristic(KnowledgeBase &kb, const Clause &goal, SearchStrategy &strategy)
    {

        // std::vector<json> training_samples;
        // 创建初始状态
        auto initialTree = std::make_shared<SLITree>(kb);

        // 创建初始操作状态
        auto initial_state = SLIOperation::createExtensionState(
            initialTree,
            initialTree->getRoot(), // 使用第一个节点作为起始节点
            Literal(),              // 空文字
            goal                    // 目标子句
        );

        // 搜索栈
        std::stack<std::shared_ptr<SLIOperation::OperationState>> state_stack;
        state_stack.push(initial_state);

        // 访问集合，用于存储已访问的状态哈希值
        std::unordered_set<size_t> visited_states;
        visited_states.insert(initialTree->computeStateHash());

        // 记录最长路径的信息
        int max_depth = 0;
        std::shared_ptr<SLIOperation::OperationState> max_depth_state = nullptr;

        std::shared_ptr<SLIOperation::OperationState> successful_state = nullptr;
        std::shared_ptr<SLIOperation::OperationState> last_state = nullptr;

        long long count = 0;
        while (!state_stack.empty())
        {
            count++;
            if (count % 5000 == 0)
            {
                std::cout << "round " << count << std::endl;
            }

            auto current_state = state_stack.top();
            last_state = current_state;
            state_stack.pop();
            std::cout << "current_state score " << current_state->heuristic_score << std::endl;
            // std::cout << current_state->state_id << " state id " << std::endl;

            // 更新最长路径
            if (current_state->depth > max_depth)
            {
                max_depth = current_state->depth;
                max_depth_state = current_state;
                // std::cout << "New maximum depth: " << max_depth << " at state id: " << current_state->state_id << std::endl;
            }

            // std::cout << "Get new state " << std::endl;
            // std::cout << "Current State before copy" << std::endl;
            // SLIOperation::printOperationPath(current_state, kb);

            std::shared_ptr<SLIOperation::OperationState> new_state;
            try
            {
                new_state = SLIOperation::deepCopyOperationState(current_state);
                // std::cout << "new state " << std::endl;
                // SLIOperation::printCurrentState(new_state, kb);
                // std::cout << "old state " << std::endl;
                // SLIOperation::printCurrentState(current_state, kb);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error during deep copy: " << e.what() << "\n";
                return false; // 根据需要决定是否跳过或终止
            }

            if (count >= 99200LL)
            {
                // SLIOperation::printOperationPath(current_state, kb);
                SLIOperation::printOperationPathAsClause(max_depth_state, kb);
                return false;
            }
            // std::cout << "Current State " << std::endl;
            // SLIOperation::printOperationPath(new_state, kb);
            // std::cout << "Performing action " << SLIOperation::getActionString(new_state->action) << std::endl;

            // 执行操作
            switch (new_state->action)
            {
            case SLIActionType::EXTENSION:
            {
                if (SLIOperation::isLiteral(new_state->second_op))
                {
                    auto kb_lit = SLIOperation::getLiteral(new_state->second_op);
                    auto new_nodes = new_state->sli_tree->add_node(
                        new_state->kb_clause,
                        kb_lit,
                        true,
                        new_state->lit1_node);
                }
                break;
            }
            case SLIActionType::FACTORING:
            {
                if (SLIOperation::isNode(new_state->second_op))
                {
                    auto second_node = SLIOperation::getNode(new_state->second_op);
                    new_state->sli_tree->t_factoring(new_state->lit1_node, second_node);
                }
                break;
            }
            case SLIActionType::ANCESTRY:
            {
                if (SLIOperation::isNode(new_state->second_op))
                {
                    auto second_node = SLIOperation::getNode(new_state->second_op);
                    new_state->sli_tree->t_ancestry(new_state->lit1_node, second_node);
                }
                break;
            }
            case SLIActionType::TRUNCATE:
            {
                // std::cout << "Performing Truncate " << std::endl;
                new_state->sli_tree->truncate(new_state->lit1_node);
                break;
            }
            }

            // 检查空子句
            // std::cout << "Check Empty Clause " << std::endl;
            if (checkEmptyClause(*new_state->sli_tree))
            {
                successful_state = new_state;
                // 打印操作路径
                SLIOperation::printOperationPath(successful_state, kb);
                return true;
            }

            // 在执行操作后添加验证
            if (!new_state->sli_tree->validateAllNodes())
            {
                continue; // 跳过包含无效节点的状态
            }
            // // 检查是否访问过
            // size_t state_hash = new_state->sli_tree->computeStateHash();
            // if (visited_states.find(state_hash) != visited_states.end())
            // {
            //     std::cout << "Skipping already visited state with hash: " << state_hash << std::endl;
            //     continue;
            // }
            // else
            // {
            //     visited_states.insert(state_hash);
            // }

            generataAllStatesHeuristic(new_state, state_stack);
            // 检查是否是特定的 state id
            // if (new_state->state_id == 19)
            // {
            //     std::cout << "Processing State ID: " << new_state->state_id << std::endl;
            //     SLIOperation::printOperationPath(new_state, kb); // 打印当前状态路径
            //     auto top_state = state_stack.top();
            //     std::cout << "top action " << SLIOperation::getActionString(top_state->action) << std::endl;
            //     SLIOperation::printOperationPath(top_state, kb);
            //     return false;
            //     // 这里可以进一步检查或记录当前状态的详细信息
            // }
        }
        SLIOperation::printOperationPath(last_state, kb);
        return false;
    }

    // 实现BFS版本的generate函数
    void SLIResolution::generateExtensionStatesBFS(
        KnowledgeBase &kb,
        const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::queue<std::shared_ptr<SLIOperation::OperationState>> &state_queue)
    {
        // 实现与原版类似，只是使用queue.push替代stack.push
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
                        auto new_state = std::make_shared<SLIOperation::OperationState>(
                            current_state->sli_tree,
                            SLIActionType::EXTENSION,
                            node,
                            SecondOperand(lit),
                            kb_clause,
                            current_state);
                        state_queue.push(new_state);
                    }
                }
            }
        }
    }

    // Factoring的BFS版本
    void SLIResolution::generateFactoringStatesBFS(
        const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::queue<std::shared_ptr<SLIOperation::OperationState>> &state_queue)
    {
        auto factoring_pairs = findPotentialFactoringPairs(current_state->sli_tree);
        for (const auto &[upper_node, lower_node] : factoring_pairs)
        {
            auto new_state = SLIOperation::createFactoringState(
                current_state->sli_tree,
                upper_node,
                lower_node,
                current_state);
            state_queue.push(new_state);
        }
    }

    // Ancestry的BFS版本
    void SLIResolution::generateAncestryStatesBFS(
        const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::queue<std::shared_ptr<SLIOperation::OperationState>> &state_queue)
    {
        auto ancestry_pairs = findPotentialAncestryPairs(current_state->sli_tree);
        for (const auto &[upper_node, lower_node] : ancestry_pairs)
        {
            auto new_state = SLIOperation::createAncestryState(
                current_state->sli_tree,
                upper_node,
                lower_node,
                current_state);
            state_queue.push(new_state);
        }
    }

    // Truncate的BFS版本
    void SLIResolution::generateTruncateStatesBFS(
        const std::vector<std::shared_ptr<SLINode>> &active_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::queue<std::shared_ptr<SLIOperation::OperationState>> &state_queue)
    {
        auto truncate_nodes = findPotentialTruncateNodes(current_state->sli_tree);
        for (const auto &node : truncate_nodes)
        {
            auto new_state = SLIOperation::createTruncateState(
                current_state->sli_tree,
                node,
                current_state);
            state_queue.push(new_state);
        }
    }

    void generateExtensionStatesHeuristc(const std::shared_ptr<SLIOperation::OperationState> &current_state,
                                         std::vector<std::shared_ptr<SLIOperation::OperationState>> &allStates)
    {
        auto b_lit_nodes = current_state->sli_tree->get_all_B_literals();
        auto kb = current_state->sli_tree->getKB();

        // ExtensionStates
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
                        auto new_state = std::make_shared<SLIOperation::OperationState>(
                            current_state->sli_tree,
                            SLIActionType::EXTENSION,
                            node,
                            SecondOperand(lit),
                            kb_clause,
                            current_state);
                        SLIOperation::calculateHeuristicScore(new_state);
                        allStates.push_back(new_state);
                    }
                }
            }
        }
    }

    void generateFactoringStatesHeuristc(const std::shared_ptr<SLIOperation::OperationState> &current_state,
                                         std::vector<std::shared_ptr<SLIOperation::OperationState>> &allStates)
    {
        // Factoring
        auto factoring_pairs = SLIResolution::findPotentialFactoringPairs(current_state->sli_tree);
        for (const auto &[upper_node, lower_node] : factoring_pairs)
        {
            auto new_state = SLIOperation::createFactoringState(
                current_state->sli_tree,
                upper_node,
                lower_node,
                current_state);
            SLIOperation::calculateHeuristicScore(new_state);
            allStates.push_back(new_state);
        }
    }

    void generateAncestryStatesHeuristc(const std::shared_ptr<SLIOperation::OperationState> &current_state,
                                        std::vector<std::shared_ptr<SLIOperation::OperationState>> &allStates)
    {
        // Ancestry
        auto ancestry_pairs = SLIResolution::findPotentialAncestryPairs(current_state->sli_tree);
        for (const auto &[upper_node, lower_node] : ancestry_pairs)
        {
            auto new_state = SLIOperation::createAncestryState(
                current_state->sli_tree,
                upper_node,
                lower_node,
                current_state);
            SLIOperation::calculateHeuristicScore(new_state);
            allStates.push_back(new_state);
        }
    }

    void generateTruncateStatesHeuristc(const std::shared_ptr<SLIOperation::OperationState> &current_state,
                                        std::vector<std::shared_ptr<SLIOperation::OperationState>> &allStates)
    {
        // Truncate
        auto truncate_nodes = SLIResolution::findPotentialTruncateNodes(current_state->sli_tree);
        for (const auto &node : truncate_nodes)
        {
            auto new_state = SLIOperation::createTruncateState(
                current_state->sli_tree,
                node,
                current_state);
            SLIOperation::calculateHeuristicScore(new_state);
            allStates.push_back(new_state);
        }
    }

    void SLIResolution::generataAllStatesHeuristic(
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack)
    {
        std::vector<std::shared_ptr<SLIOperation::OperationState>> allStates;

        auto b_lit_nodes = current_state->sli_tree->get_all_B_literals();
        auto kb = current_state->sli_tree->getKB();

        bool AC_result = current_state->sli_tree->check_all_nodes_AC();
        bool MC_result = current_state->sli_tree->check_all_nodes_MC();

        if (AC_result && MC_result)
        {
            generateExtensionStatesHeuristc(current_state, allStates);
            generateFactoringStatesHeuristc(current_state, allStates);
            generateAncestryStatesHeuristc(current_state, allStates);
            generateTruncateStatesHeuristc(current_state, allStates);
        }
        else if (MC_result)
        {
            generateFactoringStatesHeuristc(current_state, allStates);
            generateAncestryStatesHeuristc(current_state, allStates);
        }
        else if (AC_result)
        {
            generateTruncateStatesHeuristc(current_state, allStates);
        }

        // 从低到高排序 这样按照顺序入栈 栈顶就是得分最高的
        std::sort(allStates.begin(), allStates.end(),
                  [](const auto &a, const auto &b)
                  {
                      return a->heuristic_score < b->heuristic_score;
                  });

        for (const auto &state : allStates)
        {
            state_stack.push(state);
        }
    }

    double SLIResolution::calculateHeuristic(const Clause &kb_clause,
                                             const std::shared_ptr<SLINode> &tree_node,
                                             const Literal &resolving_literal)
    {
        // 基础分数
        double score = 1.0;

        // 考虑子句长度 - 更短的子句更优先
        score -= 0.1 * kb_clause.getLiterals().size();

        // 考虑深度 - 较浅的节点更优先
        score -= 0.05 * tree_node->depth;

        // 考虑变量数量 - 更少的变量更优先
        int var_count = 0;
        for (const auto &arg : resolving_literal.getArgumentIds())
        {
            if (arg.type == SymbolType::VARIABLE)
            {
                var_count++;
            }
        }
        score -= 0.1 * var_count;

        return score;
    }

    bool SLIResolution::checkEmptyClause(const SLITree &tree)
    {
        // 获取整个树的深度映射
        auto &depth_map = tree.getDepthMap();

        return tree.get_all_active_nodes().size() == 0 ? true : false;

        // // 统计所有深度的active节点
        // int active_count = 0;

        // for (size_t depth = 0; depth < depth_map.size(); ++depth)
        // {
        //     for (const auto &node : depth_map[depth])
        //     {
        //         if (node->is_active)
        //         {
        //             active_count++;
        //             // 如果在非根节点层发现active节点，直接返回false
        //             if (depth > 0)
        //             {
        //                 return false;
        //             }
        //         }
        //     }
        // }

        // // 只有根节点是active时返回true
        // return active_count == 1;
    }

    std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
    SLIResolution::findPotentialFactoringPairs(const std::shared_ptr<SLITree> &tree)
    {
        std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>> factoring_pairs;

        // 获取所有的B文字
        auto b_lit_nodes = tree->get_all_B_literals();
        KnowledgeBase kb = tree->getKB();

        for (const auto &node : b_lit_nodes)
        {
            // 获取当前node的gamma_L集合
            auto gamma_nodes = tree->get_gamma_L(node);
            // std::cout << "node " << node->node_id << " gamma_nodes size " << gamma_nodes.size() << std::endl;
            //  遍历gamma_node 检查是否能进行factoring
            for (const auto &node_m : gamma_nodes)
            {

                // 只在第一个节点的地址大于第二个节点的地址时才添加配对 因为第一个节点node是下层的节点 后添加的 地址会大
                if (node != node_m &&
                    node->node_id > node_m->node_id &&
                    // (node->depth == node_m->depth && node.get() > node_m.get()) &&
                    node->literal.getPredicateId() == node_m->literal.getPredicateId() &&
                    node->literal.isNegated() == node_m->literal.isNegated() &&
                    node->literal.getArgumentIds().size() == node_m->literal.getArgumentIds().size())
                {
                    // 如果能找到MGU 才能factoring 避免出现两个具有不同常量的 谓词符号符合条件 但是不能存在MGU
                    auto mgu = Unifier::findMGU(node->literal, node_m->literal, kb);
                    if (mgu)
                    {
                        // 注意前面是upper node 是在gammal中得到的
                        factoring_pairs.emplace_back(node_m, node);
                    }
                }
            }
        }
        return factoring_pairs;
    }

    std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>>
    SLIResolution::findPotentialAncestryPairs(const std::shared_ptr<SLITree> &tree)
    {
        std::vector<std::pair<std::shared_ptr<SLINode>, std::shared_ptr<SLINode>>> ancestry_pairs;
        KnowledgeBase kb = tree->getKB();
        // 获取所有的B文字
        auto b_lit_nodes = tree->get_all_B_literals();
        // std::cout << "Searching for potential ancestry pairs..." << std::endl;
        // 对于每个新节点
        for (const auto &node : b_lit_nodes)
        {
            // 获取当前节点的所有祖先
            std::shared_ptr<SLINode> current = node->parent.lock();
            while (current)
            {
                // 检查ancestry的基本条件
                if (node->parent.lock() != current &&
                    current->literal.getPredicateId() == node->literal.getPredicateId() &&
                    current->literal.isNegated() != node->literal.isNegated() &&
                    current->literal.getArgumentIds().size() == node->literal.getArgumentIds().size())
                {
                    // 如果能找到MGU 才能factoring 避免出现两个具有不同常量的 谓词符号符合条件 但是不能存在MGU
                    auto mgu = Unifier::findMGU(node->literal, current->literal, kb);
                    if (mgu)
                    {
                        ancestry_pairs.emplace_back(current, node);
                    }
                }
                current = current->parent.lock();
            }
        }
        // std::cout << "Found " << pairs.size() << " potential ancestry pairs" << std::endl;
        return ancestry_pairs;
    }

    std::vector<std::shared_ptr<SLINode>> SLIResolution::findPotentialTruncateNodes(
        const std::shared_ptr<SLITree> &tree)
    {
        std::vector<std::shared_ptr<SLINode>> truncate_nodes;
        // 获取所有活动节点
        auto active_nodes = tree->get_all_active_nodes();
        for (const auto &node : active_nodes)
        {
            if (node && node->is_active && node->is_A_literal)
            { // 是A-lit
                if (node->children.empty())
                { // 没有孩子节点
                    truncate_nodes.push_back(node);
                }
            }
        }
        return truncate_nodes;
    }

    void SLIResolution::checkAndTruncateNode(const std::shared_ptr<SLINode> &node, SLITree &tree)
    {
        if (node && node->is_active && node->is_A_literal)
        { // 是A-lit
            if (node->children.empty())
            { // 没有孩子节点
                tree.truncate(node);
            }
        }
    }

    void SLIResolution::generateExtensionStates2(
        KnowledgeBase &kb,
        const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack,
        std::vector<std::shared_ptr<SLIOperation::OperationState>> &available_ops)
    {
        std::vector<std::pair<double, std::shared_ptr<SLIOperation::OperationState>>> scored_states;
        auto tree = current_state->sli_tree;

        for (const auto &node : b_lit_nodes)
        {
            if (!node->is_active || node->is_A_literal)
                continue;

            // 预计算当前节点的Gamma集合和祖先集合
            auto gamma_nodes = tree->get_gamma_L(node);
            std::unordered_set<std::shared_ptr<SLINode>> ancestors;
            std::shared_ptr<SLINode> ancestor = node->parent.lock();
            while (ancestor)
            {
                ancestors.insert(ancestor);
                ancestor = ancestor->parent.lock();
            }

            for (const auto &kb_clause : kb.getClauses())
            {
                for (const auto &lit : kb_clause.getLiterals())
                {
                    if (!Resolution::isComplementary(node->literal, lit) ||
                        !Unifier::findMGU(node->literal, lit, kb))
                        continue;

                    // 创建新状态
                    auto new_state = std::make_shared<SLIOperation::OperationState>(
                        tree, SLIActionType::EXTENSION, node, SecondOperand(lit), kb_clause, current_state);

                    // 计算K值 -------------------------------------------------
                    int K = 0;
                    int N = kb_clause.getLiterals().size();
                    if (N == 0)
                        continue;

                    for (const auto &other_lit : kb_clause.getLiterals())
                    {
                        if (other_lit == lit)
                            continue;

                        // 检查该文字是否可被factoring（Gamma集合中存在可因子化节点）
                        bool can_factor = false;
                        for (const auto &gamma_node : gamma_nodes)
                        {
                            if (gamma_node->literal.getPredicateId() == other_lit.getPredicateId() &&
                                gamma_node->literal.isNegated() == other_lit.isNegated() &&
                                Unifier::findMGU(gamma_node->literal, other_lit, kb))
                            {
                                can_factor = true;
                                break;
                            }
                        }

                        // 检查该文字是否可被ancestry（祖先中存在互补节点）
                        bool can_ancestry = false;
                        for (const auto &ancestor_node : ancestors)
                        {
                            if (ancestor_node->literal.getPredicateId() == other_lit.getPredicateId() &&
                                ancestor_node->literal.isNegated() != other_lit.isNegated() &&
                                Unifier::findMGU(ancestor_node->literal, other_lit, kb))
                            {
                                can_ancestry = true;
                                break;
                            }
                        }

                        if (can_factor || can_ancestry)
                            K++;
                    }

                    // 计算得分（添加权重调节）
                    double score = (N == 1 && K == 0) ? 1.0 : (K * 1.0 / N); // 如果是单子句 N == 1 && K == 0 显然是权重最大的
                    if (score - 1.0 > 1e-5)
                        score -= (N - 1 - K) * 0.2;
                    scored_states.emplace_back(score, new_state);
                }
            }
        }

        // 按得分降序排序并压栈
        std::sort(scored_states.begin(), scored_states.end(),
                  [](const auto &a, const auto &b)
                  { return a.first < b.first; }); // 要升序排序，因为后面要顺序入栈，让分低的先入栈

        // // 在排序后添加日志
        // std::cout << "=== K/N Scores for Extension States ===" << std::endl;
        // int rank = 1;
        // for (const auto &[score, state] : scored_states)
        // {
        //     std::cout << "Rank " << rank++ << std::endl
        //               << "Score: " << score
        //               << " | Clause: " << state->kb_clause.toString(kb)
        //               << " | Resolving Lit: " << state->lit1_node->literal.toString(kb)
        //               << std::endl
        //               << "SLITree "
        //               << std::endl;
        //     current_state->sli_tree->print_tree(kb);
        // }

        for (const auto &[score, state] : scored_states)
        {
            state_stack.push(state);
            available_ops.push_back(state);
        }
    }

    // 辅助函数：生成t-extension状态
    void SLIResolution::generateExtensionStates(
        KnowledgeBase &kb,
        const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack,
        std::vector<std::shared_ptr<SLIOperation::OperationState>> &available_ops)
    {
        for (const auto &node : b_lit_nodes)
        {
            if (!node->is_active || node->is_A_literal)
                continue;

            for (const auto &kb_clause : kb.getClauses())
            {
                for (const auto &lit : kb_clause.getLiterals())
                {
                    if (Resolution::isComplementary(node->literal, lit) && Unifier::findMGU(node->literal, lit, kb))
                    {

                        // 直接使用当前树的指针
                        auto new_state = std::make_shared<SLIOperation::OperationState>(
                            current_state->sli_tree, // 还是原来的树
                            SLIActionType::EXTENSION,
                            node, // 直接使用原始节点
                            SecondOperand(lit),
                            kb_clause,
                            current_state);
                        state_stack.push(new_state);
                        available_ops.push_back(new_state);
                    }
                }
            }
        }
        return;
    }

    // 辅助函数：生成t-factoring状态
    void SLIResolution::generateFactoringStates(
        const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack,
        std::vector<std::shared_ptr<SLIOperation::OperationState>> &available_ops)
    {
        auto factoring_pairs = findPotentialFactoringPairs(current_state->sli_tree);
        for (const auto &[upper_node, lower_node] : factoring_pairs)
        {
            auto new_state = SLIOperation::createFactoringState(
                current_state->sli_tree,
                upper_node,
                lower_node,
                current_state);
            state_stack.push(new_state);
            available_ops.push_back(new_state);
        }
    }

    // 辅助函数：生成t-ancestry状态
    void SLIResolution::generateAncestryStates(
        const std::vector<std::shared_ptr<SLINode>> &b_lit_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack,
        std::vector<std::shared_ptr<SLIOperation::OperationState>> &available_ops)
    {
        auto ancestry_pairs = findPotentialAncestryPairs(current_state->sli_tree);
        for (const auto &[upper_node, lower_node] : ancestry_pairs)
        {
            auto new_state = SLIOperation::createAncestryState(
                current_state->sli_tree,
                upper_node,
                lower_node,
                current_state);
            state_stack.push(new_state);
            available_ops.push_back(new_state);
        }
    }

    // 辅助函数：生成t-truncate状态
    // 这里面传入的是所有active的节点，而不是所有b-lit
    void SLIResolution::generateTruncateStates(
        const std::vector<std::shared_ptr<SLINode>> &active_nodes,
        const std::shared_ptr<SLIOperation::OperationState> &current_state,
        std::stack<std::shared_ptr<SLIOperation::OperationState>> &state_stack,
        std::vector<std::shared_ptr<SLIOperation::OperationState>> &available_ops)
    {
        auto truncate_nodes = findPotentialTruncateNodes(current_state->sli_tree);
        for (const auto &node : truncate_nodes)
        {
            auto new_state = SLIOperation::createTruncateState(
                current_state->sli_tree,
                node,
                current_state);
            state_stack.push(new_state);
            available_ops.push_back(new_state);
        }
    }

    void SLIResolution::printProofPath(std::shared_ptr<ProofState> state, KnowledgeBase &kb)
    {
        // std::vector<std::shared_ptr<ProofState>> path;
        // auto current = state;

        // while (current)
        // {
        //     path.push_back(current);
        //     current = current->parent;
        // }

        // std::reverse(path.begin(), path.end());

        // std::cout << "\n====== Proof Path ======\n";
        // for (size_t i = 0; i < path.size(); ++i)
        // {
        //     std::cout << "\nStep " << i << " (State ID: " << path[i]->state_id << "):\n";
        //     if (i > 0)
        //     {
        //         std::cout << "Applied resolution:\n";
        //         std::cout << "- Node ID: " << path[i]->resolution_pair.node_id << "\n";
        //         std::cout << "- KB Clause: " << path[i]->resolution_pair.kb_clause.toString(kb) << "\n";
        //         std::cout << "- Resolving literal: " << path[i]->resolution_pair.resolving_literal.toString(kb) << "\n";
        //     }
        //     std::cout << "\nResulting Tree:\n";
        //     path[i]->tree->print_tree(kb);
        //     std::cout << "\n----------------------\n";
        // }
        // std::cout << "====== End of Proof ======\n";
    }

} // namespace LogicSystem