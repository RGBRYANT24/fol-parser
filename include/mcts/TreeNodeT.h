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

namespace msa
{
    namespace mcts
    {

        template <class State, typename Action>
        class TreeNodeT : public std::enable_shared_from_this<TreeNodeT<State, Action>>
        {
        public:
            using Ptr = std::shared_ptr<TreeNodeT<State, Action>>;

            //--------------------------------------------------------------
            // 构造时传入状态和父节点（默认为 nullptr）
            // 注意：这里直接调用 State 的拷贝构造函数，必须保证这个拷贝操作是深拷贝
            TreeNodeT(const State &state, Ptr parent = nullptr)
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
            Ptr expand()
            {
                // 如果所有动作已经扩展完毕，则返回 nullptr
                if (is_fully_expanded())
                    return nullptr;

                // 首次扩展时从状态中获取所有可能动作
                if (actions.empty())
                {
                    // std::cout << "actions.empty() actions size " << actions.size() << std::endl;
                    state.get_actions(actions);
                    // std::cout << "after get_actions actions size " << actions.size() << std::endl;
                    // 打乱动作顺序，防止搜索策略过于固定
                    std::random_device rd;
                    std::mt19937 g(rd());
                    std::shuffle(actions.begin(), actions.end(), g);
                }

                // 注意：add_child_with_action 内部会利用当前节点保存的 state 构造子节点，
                // 因此 State 的拷贝构造函数要确保深拷贝
                // std::cout << "actions in expand, actions size " << actions.size() << " children size " << children.size() << std::endl;
                // bool AC_result = state.sli_tree->check_all_nodes_AC();
                // bool MC_result = state.sli_tree->check_all_nodes_MC();
                // std::cout << "AC " << AC_result << " MC " << MC_result << std::endl;
                if (actions.size() == 0)
                {
                    std::cout << "actions size == 0, sli tree" << std::endl;
                    state.sli_tree->print_tree(state.sli_tree->getKB());
                    bool AC_result = state.sli_tree->check_all_nodes_AC();
                    bool MC_result = state.sli_tree->check_all_nodes_MC();
                    std::cout << "AC " << AC_result << " MC " << MC_result << std::endl;

                    // std::cout << "extension actions " << std::endl;
                    // auto kb = state.sli_tree->getKB();
                    // auto b_lit_nodes = state.sli_tree->get_all_B_literals();
                    // int count = 1;
                    // for (auto &node : b_lit_nodes)
                    // {
                    //     if (!node->is_active || node->is_A_literal)
                    //         continue;
                    //     for (const auto &kb_clause : kb.getClauses())
                    //     {
                    //         for (const auto &lit : kb_clause.getLiterals())
                    //         {
                    //             // std::cout << "sli node " << std::endl;
                    //             // node->print(kb);
                    //             // std::cout << "kb lit " << lit.toString(kb) << " clause " << kb_clause.toString(kb) << std::endl;
                    //             if (LogicSystem::Resolution::isComplementary(node->literal, lit) &&
                    //                 LogicSystem::Unifier::findMGU(node->literal, lit, kb))
                    //             {
                    //                 std::cout << "found complementary count " << count++ << std::endl;
                    //                 std::cout << "sli node " << std::endl;
                    //                 node->print(kb);
                    //                 std::cout << "kb lit " << lit.toString(kb) << " clause " << kb_clause.toString(kb) << std::endl;
                    //             }
                    //         }
                    //     }
                    // }
                }
                // for (const auto &action : actions)
                // {
                //     std::cout << action.to_string(state.sli_tree->getKB()) << std::endl;
                // }
                return add_child_with_action(actions[children.size()]);
            }

            //--------------------------------------------------------------
            // 更新节点统计量
            void update(const std::vector<float> &rewards)
            {
                value += rewards[agent_id];
                num_visits++;
            }

            //--------------------------------------------------------------
            // GETTERS
            const State &get_state() const { return state; }

            // 导致当前状态变化的动作
            const Action &get_action() const { return action; }

            // 判断是否已完全扩展：若所有动作均生成了子节点，则返回 true
            bool is_fully_expanded() const
            {
                return (!children.empty() && children.size() == actions.size());
            }

            // 当前状态是否为终端状态
            bool is_terminal() const { return state.is_terminal(); }

            // 访问次数
            int get_num_visits() const { return num_visits; }

            // 节点累计评价值，例如胜率
            float get_value() const { return value; }

            // 节点深度
            int get_depth() const { return depth; }

            // 子节点数目
            int get_num_children() const { return children.size(); }

            // 返回第 i 个子节点
            Ptr get_child(int i) const { return children[i]; }

            // 返回父节点（如果存在）
            Ptr get_parent() const { return parent.lock(); }

        private:
            // 此处的 state 需确保在拷贝时进行深拷贝
            State state;
            // 执行该状态转移的动作
            Action action;
            std::weak_ptr<TreeNodeT<State, Action>> parent; // 父节点（弱引用，防止循环引用）
            int agent_id;                                   // 当前决策 agent 的 id

            int num_visits; // 访问次数
            float value;    // 累计评价
            int depth;      // 节点在树中的深度

            std::vector<Ptr> children;   // 子节点集合
            std::vector<Action> actions; // 当前状态可用的动作列表

            //--------------------------------------------------------------
            // 基于传入的动作，创建一个子节点，并将该子节点添加到 children 内。
            // 这里调用 State 的拷贝构造函数，确保子节点状态与父节点状态分离。
            Ptr add_child_with_action(const Action &new_action)
            {
                // 生成子节点时以当前节点的 state 为基础进行深拷贝
                Ptr child_node = std::make_shared<TreeNodeT>(state, this->shared_from_this());
                // if(new_action.lit1_node == nullptr)
                // {
                //     std::cerr<<"lit1 node == nullptr" << std::endl;
                // }
                // auto lit1_node = state.sli_tree->findNodeById(new_action.lit1_node->node_id);
                // child_node->action = Action(new_action.action, lit1_node, new_action.second_op, new_action.kb_clause);
                child_node->action = new_action;
                // 在子节点上应用动作，从而得到新的（深拷贝后的）状态
                child_node->state.apply_action(new_action);
                children.push_back(child_node);
                return child_node;
            }
        };

    }
}