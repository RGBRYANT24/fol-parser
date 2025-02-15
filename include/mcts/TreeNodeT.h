/*
 A TreeNode in the decision tree.
 改为使用智能指针，父节点采用 weak_ptr 避免循环引用，
 同时继承 std::enable_shared_from_this 以便在内部获取自身的 shared_ptr。
*/

#pragma once

#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

namespace msa {
    namespace mcts {

        template <class State, typename Action>
        class TreeNodeT : public std::enable_shared_from_this<TreeNodeT<State, Action>> {
        public:
            using Ptr = std::shared_ptr<TreeNodeT<State, Action>>;

            //--------------------------------------------------------------
            // 构造时传入状态和父节点（默认为 nullptr）
            TreeNodeT(const State& state, Ptr parent = nullptr)
                : state(state),
                  parent(parent),
                  action(),
                  agent_id(state.agent_id()),
                  num_visits(0),
                  value(0),
                  depth(parent ? parent->depth + 1 : 0)
            {
            }

            //--------------------------------------------------------------
            // 如果节点未完全扩展，则扩展一个子节点
            Ptr expand() {
                // 已全部扩展则返回 nullptr
                if(is_fully_expanded()) return nullptr;

                // 如果第一次扩展且动作列表为空，从状态中获取所有可能动作
                if(actions.empty()){
                    state.get_actions(actions);
                    // 使用 std::shuffle 替换已弃用的 std::random_shuffle
                    std::random_device rd;
                    std::mt19937 g(rd());
                    std::shuffle(actions.begin(), actions.end(), g);
                }

                return add_child_with_action(actions[children.size()]);
            }

            //--------------------------------------------------------------
            // 更新节点统计量
            void update(const std::vector<float>& rewards) {
                value += rewards[agent_id];
                num_visits++;
            }

            //--------------------------------------------------------------
            // GETTERS
            const State& get_state() const { return state; }

            // 导致该节点状态发生变化的动作
            const Action& get_action() const { return action; }

            // 当所有动作均已扩展时返回 true
            bool is_fully_expanded() const {
                return (!children.empty() && children.size() == actions.size());
            }

            // 当前节点是否为终端状态
            bool is_terminal() const { return state.is_terminal(); }

            // 访问次数
            int get_num_visits() const { return num_visits; }

            // 节点累计评价值（例如胜率）
            float get_value() const { return value; }

            // 节点在树中的深度
            int get_depth() const { return depth; }

            // 子节点数目
            int get_num_children() const { return children.size(); }

            // 返回第 i 个子节点
            Ptr get_child(int i) const { return children[i]; }

            // 返回父节点（如果存在）
            Ptr get_parent() const { return parent.lock(); }

        private:
            State state;           // 当前状态
            Action action;         // 导致此状态的动作
            std::weak_ptr<TreeNodeT<State, Action>> parent;  // 父节点（使用 weak_ptr 避免循环引用）
            int agent_id;          // 作出决策的 agent id

            int num_visits;        // 访问次数
            float value;           // 节点累计评价
            int depth;             // 所处深度

            std::vector< Ptr > children;       // 子节点集合
            std::vector< Action > actions;     // 可用动作列表

            //--------------------------------------------------------------
            // 基于传入的动作创建子节点，并将该子节点添加到 children 中
            Ptr add_child_with_action(const Action& new_action) {
                // 使用 shared_from_this() 获取父节点的 shared_ptr
                Ptr child_node = std::make_shared<TreeNodeT>(state, this->shared_from_this());
                child_node->action = new_action;
                child_node->state.apply_action(new_action);
                children.push_back(child_node);
                return child_node;
            }
        };

    }
}