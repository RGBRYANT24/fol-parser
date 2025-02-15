/*
 一个简单的 C++11 模板化 MCTS 实现。
 主要修改了指针类型，全部采用智能指针进行管理。同时对随机化部分进行了替换改进。
*/

#pragma once

#include "TreeNodeT.h"
#include "MSALoopTimer.h"
#include <cfloat>
#include <vector>
#include <limits>
#include <cmath>

namespace msa {
    namespace mcts {

        // State 必须符合接口要求（参见 IState.h）
        template <class State, typename Action>
        class UCT {
            using TreeNode = TreeNodeT<State, Action>;
            using Ptr = typename TreeNode::Ptr;

        private:
            LoopTimer timer;
            int iterations;

        public:
            float uct_k;             // UCT 算法中的探索系数，默认 sqrt(2)
            unsigned int max_iterations;  // 允许的最大迭代次数（0 为不设限制）
            unsigned int max_millis;      // 最大运行时长（毫秒，0 为不设限制）
            unsigned int simulation_depth; // 模拟深度

            //--------------------------------------------------------------
            UCT() :
                iterations(0),
                uct_k(sqrt(2)),
                max_iterations(100),
                max_millis(0),
                simulation_depth(10)
            {}

            //--------------------------------------------------------------
            const LoopTimer& get_timer() const {
                return timer;
            }

            const int get_iterations() const {
                return iterations;
            }

            //--------------------------------------------------------------
            // 基于 UCT 得分获取当前节点下最佳子节点
            Ptr get_best_uct_child(Ptr node, float uct_k) const {
                if(!node->is_fully_expanded()) return nullptr;

                float best_uct_score = -std::numeric_limits<float>::max();
                Ptr best_node = nullptr;
                int num_children = node->get_num_children();
                for(int i = 0; i < num_children; i++) {
                    Ptr child = node->get_child(i);
                    float uct_exploitation = child->get_value() / (child->get_num_visits() + FLT_EPSILON);
                    float uct_exploration = sqrt(log(node->get_num_visits() + 1) / (child->get_num_visits() + FLT_EPSILON));
                    float uct_score = uct_exploitation + uct_k * uct_exploration;

                    if(uct_score > best_uct_score) {
                        best_uct_score = uct_score;
                        best_node = child;
                    }
                }

                return best_node;
            }

            //--------------------------------------------------------------
            // 获取访问次数最多的子节点
            Ptr get_most_visited_child(Ptr node) const {
                int most_visits = -1;
                Ptr best_node = nullptr;
                int num_children = node->get_num_children();
                for(int i = 0; i < num_children; i++) {
                    Ptr child = node->get_child(i);
                    if(child->get_num_visits() > most_visits) {
                        most_visits = child->get_num_visits();
                        best_node = child;
                    }
                }

                return best_node;
            }

            //--------------------------------------------------------------
            // MCTS 入口函数，返回根节点下最佳动作
            Action run(const State& current_state, unsigned int seed = 1, std::vector<State>* explored_states = nullptr) {
                // 初始化计时器
                timer.init();

                // 构造根节点，注意父指针传递 nullptr
                Ptr root_node = std::make_shared<TreeNode>(current_state, nullptr);
                Ptr best_node = nullptr;

                iterations = 0;
                while(true) {
                    timer.loop_start();

                    // 1. SELECT：从根节点开始沿最优路径走直到遇到非完全扩展或终端节点
                    Ptr node = root_node;
                    while(!node->is_terminal() && node->is_fully_expanded()) {
                        node = get_best_uct_child(node, uct_k);
                    }

                    // 2. EXPAND：如果节点还未完全扩展且不是终端状态，则扩展一个子节点
                    if(!node->is_fully_expanded() && !node->is_terminal()){
                        node = node->expand();
                    }

                    State state = node->get_state();

                    // 3. SIMULATE：从扩展后的节点进行模拟（若非终端状态）
                    if(!node->is_terminal()) {
                        Action action;
                        for(unsigned int t = 0; t < simulation_depth; t++) {
                            if(state.is_terminal()) break;
                            if(state.get_random_action(action))
                                state.apply_action(action);
                            else
                                break;
                        }
                    }

                    // 获取模拟后的奖励
                    const std::vector<float> rewards = state.evaluate();

                    // 保存探索过的状态（如果需要）
                    if(explored_states) {
                        explored_states->push_back(state);
                    }

                    // 4. BACK PROPAGATION：从模拟节点回溯更新所有祖先节点状态
                    while(node) {
                        node->update(rewards);
                        node = node->get_parent();
                    }

                    // 选择最常访问的子节点作为当前最佳节点
                    best_node = get_most_visited_child(root_node);

                    timer.loop_end();

                    // 根据运行时长或迭代次数决定是否退出循环
                    if(max_millis > 0 && timer.check_duration(max_millis)) break;
                    if(max_iterations > 0 && iterations >= max_iterations) break;
                    iterations++;
                }

                // 返回找到的最佳动作
                if(best_node)
                    return best_node->get_action();

                return Action();
            }
        };

    }
}