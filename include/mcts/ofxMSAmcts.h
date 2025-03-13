#pragma once

#include "TreeNodeT.h"
#include "MSALoopTimer.h"
#include <cfloat>
#include <vector>
#include <limits>
#include <cmath>
#include <iostream>

namespace msa
{
    namespace mcts
    {
        // 定义返回结果结构体：包含最佳动作以及整个搜索树根节点的智能指针
        template <class State, typename Action>
        struct MCTSResult
        {
            Action best_action;
            std::shared_ptr<TreeNodeT<State, Action>> root_node;
        };

        // State 必须满足 IState 接口要求，且其拷贝构造函数需实现深拷贝
        template <class State, typename Action>
        class UCT
        {
            using TreeNode = TreeNodeT<State, Action>;
            using Ptr = typename TreeNode::Ptr;

        private:
            LoopTimer timer;
            int iterations;
            int visited_states; // 跟踪一次搜索中访问的状态数量

        public:
            float uct_k;                   // UCT 中的探索系数，默认值 sqrt(2)
            unsigned int max_iterations;   // 最大迭代次数（0 表示无限制）
            unsigned int max_millis;       // 最大运行时间（毫秒，0 表示无限制）
            unsigned int simulation_depth; // 模拟深度

            // 添加获取访问状态数的方法
            const int get_visited_states() const
            {
                return visited_states;
            }

            //--------------------------------------------------------------
            UCT() : iterations(0),
                    visited_states(0),
                    uct_k(std::sqrt(2)),
                    max_iterations(100),
                    max_millis(0),
                    simulation_depth(10)
            {
            }

            //--------------------------------------------------------------
            const LoopTimer &get_timer() const
            {
                return timer;
            }

            const int get_iterations() const
            {
                return iterations;
            }

            //--------------------------------------------------------------
            // 基于 UCT 得分选取当前节点下最佳子节点
            Ptr get_best_uct_child(Ptr node, float uct_k) const
            {
                if (!node->is_fully_expanded())
                    return nullptr;

                float best_uct_score = -std::numeric_limits<float>::max();
                Ptr best_node = nullptr;
                int num_children = node->get_num_children();
                for (int i = 0; i < num_children; i++)
                {
                    Ptr child = node->get_child(i);
                    float uct_exploitation = child->get_value() / (child->get_num_visits() + FLT_EPSILON);
                    float uct_exploration = std::sqrt(std::log(node->get_num_visits() + 1) / (child->get_num_visits() + FLT_EPSILON));
                    float uct_score = uct_exploitation + uct_k * uct_exploration;

                    if (uct_score > best_uct_score)
                    {
                        best_uct_score = uct_score;
                        best_node = child;
                    }
                }

                return best_node;
            }

            //--------------------------------------------------------------
            // 返回访问次数最多的子节点
            Ptr get_most_visited_child(Ptr node) const
            {
                int most_visits = -1;
                Ptr best_node = nullptr;
                int num_children = node->get_num_children();
                for (int i = 0; i < num_children; i++)
                {
                    Ptr child = node->get_child(i);
                    if (child->get_num_visits() > most_visits)
                    {
                        most_visits = child->get_num_visits();
                        best_node = child;
                    }
                }
                return best_node;
            }

            //--------------------------------------------------------------
            // 修改后的 MCTS 入口函数，返回一个 MCTSResult 结构体，
            // 其中包含最佳动作以及搜索树的根节点指针
            MCTSResult<State, Action> run(const State &current_state,
                                          unsigned int seed = 1,
                                          std::vector<State> *explored_states = nullptr)
            {
                auto KB = current_state.sli_tree->getKB();
                timer.init();
                visited_states = 0; // 重置状态计数器
                Ptr root_node = std::make_shared<TreeNodeT<State, Action>>(current_state, nullptr);
                visited_states++; // 计算根节点状态
                Ptr best_node = nullptr;

                iterations = 0;
                while (true)
                {
                    timer.loop_start();

                    // 1. SELECT
                    Ptr node = root_node;
                    while (!node->is_terminal() && node->is_fully_expanded())
                    {
                        node = get_best_uct_child(node, uct_k);
                        // 不计入选择阶段的状态，因为这些节点在之前已被计数
                    }

                    // 2. EXPAND
                    if (!node->is_fully_expanded() && !node->is_terminal())
                    {
                        node = node->expand();
                        visited_states++; // 计算新扩展节点的状态
                    }

                    // 获取扩展节点后的状态副本
                    State state = node->get_state();

                    // 3. SIMULATE
                    if (!node->is_terminal())
                    {
                        Action action;
                        for (unsigned int t = 0; t < simulation_depth; t++)
                        {
                            if (state.is_terminal())
                                break;
                            if (state.get_random_action(action))
                            {
                                state.apply_action(action);
                                visited_states++; // 计算模拟中的每个新状态
                            }
                            else
                                break;
                        }
                    }

                    // 获取模拟后的奖励
                    const std::vector<float> rewards = state.evaluate();

                    // 如果需要，保存探索过的状态
                    if (explored_states)
                    {
                        explored_states->push_back(state);
                    }

                    // 4. BACK PROPAGATION
                    while (node)
                    {
                        node->update(rewards);
                        node = node->get_parent();
                    }

                    best_node = get_most_visited_child(root_node);
                    timer.loop_end();

                    if (max_millis > 0 && timer.check_duration(max_millis))
                        break;
                    if (max_iterations > 0 && iterations >= max_iterations)
                        break;
                    iterations++;
                }

                // 以下内容仅用于调试：遍历 root_node 的子节点，打印奖励和动作信息
                auto children = root_node->get_children();
                auto actions = root_node->get_actions();

                // std::cout << "ofxMSAmcts::run TreeNode" << std::endl;
                // std::cout << "children.size() " << children.size() << " actions size() " << actions.size() << std::endl;
                // for (const auto &node : children)
                // {
                //     std::cout << node->get_value() << " ";
                // }
                // std::cout << std::endl;
                // if (children.size() == actions.size())
                // {
                //     for (int i = 0; i < children.size(); i++)
                //     {
                //         auto node = children[i];
                //         auto sli_tree = node->get_state().sli_tree;
                //         sli_tree->print_tree(KB);
                //         auto action = actions[i];
                //         std::cout << action.to_string(KB) << std::endl;
                //     }
                // }

                MCTSResult<State, Action> result;
                if (best_node)
                    result.best_action = best_node->get_action();
                result.root_node = root_node;
                return result;
            }
        };
    }
}