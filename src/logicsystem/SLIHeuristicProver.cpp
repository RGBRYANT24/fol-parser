#include "SLIHeuristicProver.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <string>
#include <filesystem>
#include <chrono>

#include "DataCollector.h"
#include "NeuralHeuristicSearch.h"

namespace LogicSystem
{
    SLIHeuristicProver::SLIHeuristicProver(KnowledgeBase &kb, const Clause &goal)
        : kb(kb), goal(goal)
    {
    }

    SearchResult SLIHeuristicProver::prove(const std::string &save_dir)
    {
        // 初始化返回结构
        SearchResult result;
        result.method = "NeuralHeuristic"; // 设置搜索方法名称

        // 1. 初始化神经网络服务
        NeuralHeuristicSearch heuristicSearch;
        if (!heuristicSearch.initialize(pythonPath, scriptPath, modelPath, tokenizerPath))
        {
            std::cerr << "初始化神经网络服务失败" << std::endl;
            result.success = false;
            result.visitedStates = 0;
            result.duration = 0;
            return result;
        }

        // 2. 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();

        // 3. 执行神经网络启发式DFS搜索
        bool search_success = heuristicSearch.search(kb, goal, maxIterations);

        // 4. 记录结束时间和用时
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // 5. 获取访问状态数
        int visited_states = heuristicSearch.getVisitedStatesCount();

        // 6. 设置结果
        result.success = search_success;
        result.visitedStates = visited_states;
        result.duration = duration;

        // 7. 输出搜索结果
        if (search_success)
        {
            std::cout << "证明成功! 用时: " << duration << " 毫秒, 访问状态数: " << visited_states << std::endl;
        }
        else
        {
            std::cout << "证明失败，达到最大迭代次数或搜索空间已耗尽。用时: " << duration
                      << " 毫秒, 访问状态数: " << visited_states << std::endl;
        }

        return result;
    }

    void SLIHeuristicProver::setConfig(const std::string &python, const std::string &script,
                                       const std::string &model, const std::string &tokenizer)
    {
        pythonPath = python;
        scriptPath = script;
        modelPath = model;
        tokenizerPath = tokenizer;
    }

    void SLIHeuristicProver::setMaxIterations(int iterations)
    {
        maxIterations = iterations;
    }
}