#pragma once

#include "TreeNodeT.h"
#include "MSALoopTimer.h"
#include <cfloat>
#include <vector>
#include <limits>
#include <cmath>

namespace msa
{
    namespace mcts
    {

        // State 必须满足 IState 接口要求，且其拷贝构造函数需实现深拷贝
        template <class State, typename Action>
        class UCT
        {
            using TreeNode = TreeNodeT<State, Action>;
            using Ptr = typename TreeNode::Ptr;

        private:
            LoopTimer timer;
            int iterations;

        public:
            float uct_k;                   // UCT 中的探索系数，默认值 sqrt(2)
            unsigned int max_iterations;   // 最大迭代次数（0 表示无限制）
            unsigned int max_millis;       // 最大运行时间（毫秒，0 表示无限制）
            unsigned int simulation_depth; // 模拟深度

            //--------------------------------------------------------------
            UCT() : iterations(0),
                    uct_k(sqrt(2)),
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
                    float uct_exploration = sqrt(log(node->get_num_visits() + 1) / (child->get_num_visits() + FLT_EPSILON));
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
            // MCTS 的入口函数，返回根节点下最佳动作
            // 注意：构造根节点时，会利用 State 的拷贝构造函数（深拷贝）生成根状态
            Action run(const State &current_state, unsigned int seed = 1, std::vector<State> *explored_states = nullptr)
            {
                auto KB = current_state.sli_tree->getKB();
                timer.init();
                Ptr root_node = std::make_shared<TreeNodeT<State, Action>>(current_state, nullptr);
                Ptr best_node = nullptr;

                iterations = 0;
                while (true)
                {
                    // if (iterations >= 5)
                    //     break;
                    timer.loop_start();

                    // 1. SELECT：从根开始沿着最佳路径前进，直至遇到非完全扩展或终端节点
                    Ptr node = root_node;
                    while (!node->is_terminal() && node->is_fully_expanded())
                    {
                        node = get_best_uct_child(node, uct_k);
                    }

                    // 2. EXPAND：如果当前节点未完全扩展且非终端状态，则扩展一个子节点
                    if (!node->is_fully_expanded() && !node->is_terminal())
                    {
                        node = node->expand();
                        // std::cout << "Expand " << iterations << std::endl;
                        // State state = node->get_state();
                        // state.sli_tree->print_tree(state.sli_tree->getKB());
                    }

                    // 获取扩展节点后的状态副本（调用 State 的拷贝构造函数保证深拷贝）
                    // std::cout << "get_stae" <<  std::endl;
                    State state = node->get_state();
                    // state.sli_tree->print_tree(state.sli_tree->getKB());

                    // 3. SIMULATE：从扩展节点开始进行模拟（非终端节点）
                    if (!node->is_terminal())
                    {
                        // std::cout << "simuluate iterations " << iterations << " simulation_depth " << simulation_depth << std::endl;
                        Action action;
                        for (unsigned int t = 0; t < simulation_depth; t++)
                        {
                            if (state.is_terminal())
                                break;
                            if (state.get_random_action(action))
                            {
                                // std::cout << "simution deepth " << t << std::endl;
                                // std::cout << "random action " << action.to_string(KB) << std::endl;
                                // std::cout << "before apply action" << std::endl;
                                // state.sli_tree->print_tree(KB);
                                state.apply_action(action);
                                // std::cout << "after apply action" << std::endl;
                                // state.sli_tree->print_tree(KB);
                            }
                            else
                                break;
                        }
                    }

                    // 获取模拟后的奖励
                    const std::vector<float> rewards = state.evaluate();
                    // std::cout << "rewards " << rewards[0] << " iterations " << iterations << std::endl;

                    // 可选：保存探索过的状态（保存的是模拟结束后的 state 副本）
                    if (explored_states)
                    {
                        explored_states->push_back(state);
                    }

                    // 4. BACK PROPAGATION：更新从扩展节点到根节点路径上所有节点的统计数据
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

                if (best_node)
                    return best_node->get_action();

                return Action();
            }
        };

    }
}