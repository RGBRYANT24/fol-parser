// SLITree.h
#ifndef LOGIC_SYSTEM_SLI_TREE_H
#define LOGIC_SYSTEM_SLI_TREE_H

#include "SLINode.h"
#include "KnowledgeBase.h"
#include "operation/Operations.h"
#include "Unifier.h"
#include "Resolution.h"
#include <unordered_map>
#include <stack>
#include <memory>

namespace LogicSystem
{
    class SLITree
    {
    public:
        SLITree(KnowledgeBase &kb) : kb(kb)
        {
            root = std::make_shared<SLINode>(Literal(), false, SLINode::next_node_id++);
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
        // 添加拷贝构造函数声明
        SLITree(const SLITree &other, std::shared_ptr<SLINode> startNode);
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
        bool t_factoring(std::shared_ptr<SLINode> upper_node, std::shared_ptr<SLINode> lower_node);
        bool t_ancestry(std::shared_ptr<SLINode> node1, std::shared_ptr<SLINode> node2);

        // 计算树状态的哈希值
        size_t computeStateHash() const;

        // 比较两个树状态是否等价
        bool isEquivalentTo(const SLITree &other) const;

        // 返回统一后的新文字，如果无法统一则返回nullopt
        std::optional<Literal> try_unify(const Literal &lit1, const Literal &lit2)
        {
            auto mgu = Unifier::findMGU(lit1, lit2, kb);
            if (!mgu)
                return std::nullopt;
            return Unifier::applySubstitutionToLiteral(lit1, *mgu, kb);
        }

        // 添加通过ID查找节点的方法
        std::shared_ptr<SLINode> findNodeById(int id) const
        {
            for (const auto &level : depth_map)
            {
                for (const auto &node : level)
                {
                    if (node && node->node_id == id)
                    {
                        return node;
                    }
                }
            }
            return nullptr;
        }

        const std::vector<std::vector<std::shared_ptr<SLINode>>> &getDepthMap() const { return this->depth_map; };
        std::unordered_map<size_t, std::shared_ptr<SLINode>> getLitMap() { return this->literal_map; };

        bool hasSelfLoop() const { return has_self_loop; }

        void print_tree(const KnowledgeBase &kb) const;

    private:
        KnowledgeBase &kb;
        std::unordered_map<size_t, std::shared_ptr<SLINode>> literal_map;
        std::vector<std::vector<std::shared_ptr<SLINode>>> depth_map;
        std::stack<std::unique_ptr<Operation>> operation_stack;
        std::shared_ptr<SLINode> root;
        bool has_self_loop = false; // 判定是否产生E(x,x)这样的自环
        // 删除静态 next_node_id，因为现在在 SLINode 中维护

        // 用于计算单个节点状态的哈希值
        size_t computeNodeHash(const std::shared_ptr<SLINode> &node) const;

        // 检查两个节点是否等价
        bool areNodesEquivalent(const std::shared_ptr<SLINode> &node1,
                                const std::shared_ptr<SLINode> &node2) const;

        void print_node(std::shared_ptr<SLINode> node, const KnowledgeBase &kb,
                        std::string prefix, bool is_last) const;

        void print_node_info(std::shared_ptr<SLINode> node, const KnowledgeBase &kb,
                             std::string prefix, bool is_last) const;

        std::string get_branch_str(bool is_last) const
        {
            return is_last ? "\\-" : "|-";
        }

        // 添加用于拷贝的辅助函数
        std::shared_ptr<SLINode> copySubtree(std::shared_ptr<SLINode> node,
                                             std::shared_ptr<SLINode> parent,
                                             std::unordered_map<std::shared_ptr<SLINode>,
                                                                std::shared_ptr<SLINode>> &nodeMap);
        // 用于树复制的辅助函数
        std::shared_ptr<SLINode> copyNode(const std::shared_ptr<SLINode> &node);

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

        bool isValidLiteral(const Literal &lit) const
        {
            // 检查是否为边关系谓词
            if (lit.getPredicateId() == kb.getPredicateId("E"))
            {
                const auto &args = lit.getArgumentIds();
                // 检查是否有两个参数且参数相同（自环）
                if (args.size() == 2 && args[0] == args[1])
                {
                    return false;
                }
            }
            return true;
        }
    };
}

#endif // LOGIC_SYSTEM_SLI_TREE_H