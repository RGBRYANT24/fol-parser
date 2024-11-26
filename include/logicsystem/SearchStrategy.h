// SearchStrategy.h
#ifndef LOGIC_SYSTEM_SEARCHSTRATEGY_H
#define LOGIC_SYSTEM_SEARCHSTRATEGY_H

#include "KnowledgeBase.h"
#include "ResolutionPair.h"
#include "SLINode.h"
#include <queue>

namespace LogicSystem
{
    struct SLIResolutionPair {
        std::shared_ptr<SLINode> tree_node;
        Clause kb_clause;
        Literal resolving_literal;
        double score;
        
        SLIResolutionPair(std::shared_ptr<SLINode> node, 
                         const Clause& clause,
                         const Literal& lit,
                         double s) 
            : tree_node(node), kb_clause(clause), 
              resolving_literal(lit), score(s) {}
              
        bool operator<(const SLIResolutionPair& other) const {
            return score > other.score;
        }
    };

    class SearchStrategy
    {
    public:
        // 基本操作
        virtual void addPair(const ResolutionPair& pair) = 0;
        virtual void addSLIPair(const SLIResolutionPair& pair) = 0;
        virtual bool isEmpty() const = 0;
        virtual ResolutionPair getNext() = 0;
        virtual SLIResolutionPair getNextSLI() = 0;
        
        // 启发式控制
        virtual bool shouldTryResolution(double score) const = 0;
        virtual bool shouldBacktrack() const = 0;
        virtual void updateHeuristic(const std::vector<std::shared_ptr<SLINode>>& new_nodes) = 0;
        
        // 资源限制
        virtual void setMaxDepth(int depth) = 0;
        virtual void setTimeLimit(double seconds) = 0;
        virtual void setMemoryLimit(size_t bytes) = 0;
        
        // 状态查询
        virtual double getBestScore() const = 0;
        virtual size_t getSearchedStates() const = 0;
        
        virtual ~SearchStrategy() = default;
    };

    // 具体策略实现示例
    class BestFirstStrategy : public SearchStrategy {
    private:
        std::priority_queue<SLIResolutionPair> sli_queue;
        int max_depth = -1;
        double time_limit = -1;
        size_t memory_limit = -1;
        size_t searched_states = 0;
        
    public:
        void addSLIPair(const SLIResolutionPair& pair) override {
            if (max_depth != -1 && pair.tree_node->depth > max_depth) {
                return;
            }
            sli_queue.push(pair);
        }
        
        bool isEmpty() const override {
            return sli_queue.empty();
        }
        
        SLIResolutionPair getNextSLI() override {
            auto next = sli_queue.top();
            sli_queue.pop();
            searched_states++;
            return next;
        }
        
        bool shouldTryResolution(double score) const override {
            return score > 0.5; // 示例阈值
        }
        
        bool shouldBacktrack() const override {
            return searched_states % 1000 == 0; // 周期性回溯
        }
        
        // ... 实现其他接口方法
    };
}

#endif // LOGIC_SYSTEM_SEARCHSTRATEGY_H