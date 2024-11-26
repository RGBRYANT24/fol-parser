// BFSStrategy.h
#ifndef LOGIC_SYSTEM_BFS_STRATEGY_H
#define LOGIC_SYSTEM_BFS_STRATEGY_H

#include "SearchStrategy.h"
#include <queue>
#include <chrono>

namespace LogicSystem {

class BFSStrategy : public SearchStrategy {
public:
    BFSStrategy(int maxDepth = -1, double timeLimit = -1, size_t memLimit = -1)
        : max_depth(maxDepth)
        , time_limit(timeLimit)
        , memory_limit(memLimit)
        , searched_states(0)
        , start_time(std::chrono::steady_clock::now())
    {}

    void addPair(const ResolutionPair& pair) override {
        // 不使用传统的ResolutionPair
    }

    void addSLIPair(const SLIResolutionPair& pair) override {
        // 检查深度限制
        if (max_depth != -1 && pair.tree_node->depth > max_depth) {
            return;
        }

        // 检查时间限制
        if (time_limit > 0) {
            auto current_time = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(current_time - start_time).count();
            if (elapsed > time_limit) {
                return;
            }
        }

        // 检查内存限制（简化版）
        if (memory_limit > 0 && sli_queue.size() * sizeof(SLIResolutionPair) > memory_limit) {
            return;
        }

        sli_queue.push(pair);
    }

    bool isEmpty() const override {
        return sli_queue.empty();
    }

    ResolutionPair getNext() override {
        // 不使用传统的ResolutionPair
        return ResolutionPair(nullptr, nullptr, 0, 0, 0);
    }

    SLIResolutionPair getNextSLI() override {
        if (sli_queue.empty()) {
            throw std::runtime_error("No more pairs available");
        }

        auto next = sli_queue.front();
        sli_queue.pop();
        searched_states++;
        return next;
    }

    bool shouldTryResolution(double score) const override {
        // BFS策略下尝试所有可能的消解
        return true;
    }

    bool shouldBacktrack() const override {
        // 在BFS中，我们通常不需要回溯
        return false;
    }

    void updateHeuristic(const std::vector<std::shared_ptr<SLINode>>& new_nodes) override {
        // BFS不需要更新启发式信息
    }

    void setMaxDepth(int depth) override {
        max_depth = depth;
    }

    void setTimeLimit(double seconds) override {
        time_limit = seconds;
    }

    void setMemoryLimit(size_t bytes) override {
        memory_limit = bytes;
    }

    double getBestScore() const override {
        return sli_queue.empty() ? 0.0 : sli_queue.front().score;
    }

    size_t getSearchedStates() const override {
        return searched_states;
    }

    // 用于调试的辅助函数
    void printStatus() const {
        std::cout << "BFS Strategy Status:" << std::endl;
        std::cout << "Queue size: " << sli_queue.size() << std::endl;
        std::cout << "Searched states: " << searched_states << std::endl;
        if (!sli_queue.empty()) {
            std::cout << "Next pair score: " << sli_queue.front().score << std::endl;
        }
    }

private:
    std::queue<SLIResolutionPair> sli_queue;
    int max_depth;
    double time_limit;
    size_t memory_limit;
    size_t searched_states;
    std::chrono::steady_clock::time_point start_time;
};

} // namespace LogicSystem

#endif // LOGIC_SYSTEM_BFS_STRATEGY_H