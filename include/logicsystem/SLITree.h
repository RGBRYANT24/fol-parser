// SLITree.h
#ifndef LOGIC_SYSTEM_SLI_TREE_H
#define LOGIC_SYSTEM_SLI_TREE_H

#include "SLINode.h"
#include "KnowledgeBase.h"
#include "Unifier.h"
#include "Resolution.h"
#include <unordered_map>
#include <stack>
#include <memory>

namespace LogicSystem
{

    class Operation
    {
    public:
        virtual ~Operation() = default;
        virtual void undo() = 0;
    };

    class TruncateOperation : public Operation
    {
    public:
        explicit TruncateOperation(std::shared_ptr<SLINode> n) : node(n) {}
        void save_state(std::shared_ptr<SLINode> n)
        {
            affected_nodes.push_back(n);
            previous_states.push_back(n->is_active);
        }
        void undo() override;

    private:
        std::shared_ptr<SLINode> node;
        std::vector<bool> previous_states;
        std::vector<std::shared_ptr<SLINode>> affected_nodes;
    };

    class AddOperation : public Operation
    {
    public:
        AddOperation(const std::vector<std::shared_ptr<SLINode>> &nodes,
                     std::unordered_map<size_t, std::shared_ptr<SLINode>> &litMap,
                     std::vector<std::vector<std::shared_ptr<SLINode>>> &depthMap)
            : added_nodes(nodes), literal_map(litMap), depth_map(depthMap) {}

        void undo() override;

    private:
        std::vector<std::shared_ptr<SLINode>> added_nodes;
        std::unordered_map<size_t, std::shared_ptr<SLINode>> &literal_map;
        std::vector<std::vector<std::shared_ptr<SLINode>>> &depth_map;
    };
    class SLITree
    {
    public:
        SLITree(KnowledgeBase &kb) : kb(kb), next_node_id(0)
        { // 创建空的根节点
            root = std::make_shared<SLINode>(Literal(), false, next_node_id++);
            root->depth = 0;
            if (depth_map.empty())
            {
                depth_map.resize(1);
            }
            depth_map[0].push_back(root);
        }
        // 添加获取根节点的函数
        std::shared_ptr<SLINode> getRoot() const
        {
            return root;
        }

        // 核心操作
        std::vector<std::shared_ptr<SLINode>> add_node(const Clause &input_clause, const Literal &resolving_literal,
                                                            bool is_A_literal, std::shared_ptr<SLINode> parent);
        bool is_ancestor(std::shared_ptr<SLINode> potential_ancestor,
                         std::shared_ptr<SLINode> potential_descendant);
        void truncate(std::shared_ptr<SLINode> node);
        void rollback();

        // 查找操作
        std::shared_ptr<SLINode> find_literal(const Literal &literal) const;
        std::vector<std::shared_ptr<SLINode>> get_active_nodes_at_depth(int depth) const;

        // SLI特定操作
        bool t_factoring(std::shared_ptr<SLINode> node1, std::shared_ptr<SLINode> node2);
        bool t_ancestry(std::shared_ptr<SLINode> node1, std::shared_ptr<SLINode> node2);

        // 返回统一后的新文字，如果无法统一则返回nullopt
        std::optional<Literal> try_unify(const Literal &lit1, const Literal &lit2)
        {
            auto mgu = Unifier::findMGU(lit1, lit2, kb);
            if (!mgu)
                return std::nullopt;
            return Unifier::applySubstitutionToLiteral(lit1, *mgu, kb);
        }

        void print_tree(const KnowledgeBase &kb) const;

    private:
        KnowledgeBase &kb;
        std::unordered_map<size_t, std::shared_ptr<SLINode>> literal_map;
        std::vector<std::vector<std::shared_ptr<SLINode>>> depth_map;
        std::stack<std::unique_ptr<Operation>> operation_stack;
        std::shared_ptr<SLINode> root;
        int next_node_id;

        void print_node(std::shared_ptr<SLINode> node, const KnowledgeBase &kb,
                        std::string prefix, bool is_last) const;

        void print_node_info(std::shared_ptr<SLINode> node, const KnowledgeBase &kb,
                             std::string prefix, bool is_last) const;

        std::string get_branch_str(bool is_last) const
        {
            return is_last ? "\\-" : "|-";
        }

        // 将 join 函数作为私有成员函数
        std::string join(const std::vector<std::string> &elements,
                         const std::string &delimiter) const
        {
            std::string result;
            for (size_t i = 0; i < elements.size(); ++i)
            {
                if (i > 0)
                    result += delimiter;
                result += elements[i];
            }
            return result;
        }
    };

}

#endif // LOGIC_SYSTEM_SLI_TREE_H