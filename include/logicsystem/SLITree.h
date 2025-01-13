// SLITree.h
#ifndef LOGIC_SYSTEM_SLI_TREE_H
#define LOGIC_SYSTEM_SLI_TREE_H

#include "SLINode.h"
#include "KnowledgeBase.h"
#include "operation/Operations.h"
#include "Unifier.h"
#include "VariableRenamer.h"
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
        // 深拷贝方法
        std::shared_ptr<SLITree> deepCopy() const
        {
            return std::make_shared<SLITree>(*this, root); // 使用拷贝构造函数
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

        // 获取δL集合
        std::vector<std::shared_ptr<SLINode>> get_delta_L(std::shared_ptr<SLINode> L_node) const;
        // 获取γL集合
        std::vector<std::shared_ptr<SLINode>> get_gamma_L(std::shared_ptr<SLINode> L_node) const;

        // 检查一个节点是否满足AC条件
        bool check_AC(std::shared_ptr<SLINode> L_node) const;
        // 检查一个节点是否满足MC条件
        bool check_MC(const std::shared_ptr<SLINode> &node) const;
        // 检查所有节点是否满足AC条件
        bool check_all_nodes_AC() const;
        // 检查所有节点是否满足MC条件
        bool check_all_nodes_MC() const;

        // 查找操作
        std::shared_ptr<SLINode> find_literal(const Literal &literal) const;
        std::vector<std::shared_ptr<SLINode>> get_active_nodes_at_depth(int depth) const;

        // SLI特定操作
        bool t_factoring(std::shared_ptr<SLINode> upper_node, std::shared_ptr<SLINode> lower_node);
        bool t_ancestry(std::shared_ptr<SLINode> node1, std::shared_ptr<SLINode> node2);

        // 计算树状态的哈希值
        size_t computeStateHash() const;
        // 用于计算单个节点状态的哈希值
        size_t computeNodeHash(const std::shared_ptr<SLINode> &node) const;

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

        std::vector<std::shared_ptr<SLINode>> get_all_B_literals();

        const std::vector<std::vector<std::shared_ptr<SLINode>>> &getDepthMap() const { return this->depth_map; };
        // std::unordered_map<size_t, std::shared_ptr<SLINode>> getLitMap() { return this->literal_map; };

        // 获取所有活动节点
        std::vector<std::shared_ptr<SLINode>> get_all_active_nodes() const;

        bool hasSelfLoop() const { return has_self_loop; }

        void print_tree(const KnowledgeBase &kb) const;

        KnowledgeBase getKB() { return this->kb; }

        bool validateAllNodes() 
        {
            // 从根节点开始递归检查所有节点
            for (const auto &level : depth_map)
            {
                for (const auto &node : level)
                {
                    if (node && node->is_active)
                    { // 只检查活跃节点
                        const auto &lit = node->literal;
                        // 检查是否为边关系谓词
                        auto predId = kb.getPredicateId("E");
                        if (predId && lit.getPredicateId() == *predId)
                        {
                            const auto &args = lit.getArgumentIds();
                            // 检查是否有两个参数且参数相同（自环）
                            if (args.size() == 2 && args[0] == args[1])
                            {
                                this->has_self_loop = true; // 设置标志
                                return false;         // 发现无效节点，返回false
                            }
                        }
                    }
                }
            }
            return true; // 所有节点都有效
        }
        std::string printBLiteralsAsClause() const;

    private:
        KnowledgeBase &kb;
        // std::unordered_map<size_t, std::shared_ptr<SLINode>> literal_map;
        std::vector<std::vector<std::shared_ptr<SLINode>>> depth_map;
        std::stack<std::unique_ptr<Operation>> operation_stack;
        std::shared_ptr<SLINode> root;
        bool has_self_loop = false; // 判定是否产生E(x,x)这样的自环
        // 删除静态 next_node_id，因为现在在 SLINode 中维护

        // 清理depth_map中的空层
        void cleanup_empty_depths();

        // 辅助函数 检查是否有两个原子
        bool have_same_atom(const std::shared_ptr<SLINode> &node1, const std::shared_ptr<SLINode> &node2) const;

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