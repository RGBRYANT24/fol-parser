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
        
        // 设置实验模式（如果有指定）
        if (experimentMode != "both_phases") {
            heuristicSearch.setExperimentMode(experimentMode);
        }
        
        // 初始化两个阶段的神经网络
        bool init_success = false;
        
        // 检查是否已经配置了第二阶段模型
        if (!phase2ModelPath.empty() && !phase2ScriptPath.empty()) {
            // 使用两阶段初始化
            init_success = heuristicSearch.initialize(
                pythonPath, 
                phase1ScriptPath, phase1ModelPath, phase1TokenizerPath,
                phase2ScriptPath, phase2ModelPath, phase2TokenizerPath
            );
            
            if (!init_success) {
                std::cerr << "初始化两阶段神经网络服务失败" << std::endl;
            }
            else{
                std::cout << "初始化两阶段神经网络服务成功" <<std::endl;
            }
        } else {
            // 向后兼容，仅初始化第一阶段
            init_success = heuristicSearch.initialize(
                pythonPath, phase1ScriptPath, phase1ModelPath, phase1TokenizerPath
            );
            
            if (!init_success) {
                std::cerr << "初始化神经网络服务失败" << std::endl;
            }
        }
        
        if (!init_success) {
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

    // 设置第一阶段配置
    void SLIHeuristicProver::setPhase1Config(const std::string &python, const std::string &script,
                                           const std::string &model, const std::string &tokenizer)
    {
        pythonPath = python;
        phase1ScriptPath = script;
        phase1ModelPath = model;
        phase1TokenizerPath = tokenizer;
    }
    
    // 设置第二阶段配置
    void SLIHeuristicProver::setPhase2Config(const std::string &script, const std::string &model,
                                           const std::string &tokenizer)
    {
        phase2ScriptPath = script;
        phase2ModelPath = model;
        phase2TokenizerPath = tokenizer;
    }
    
    // 设置实验模式
    void SLIHeuristicProver::setExperimentMode(const std::string &mode)
    {
        if (mode == "phase1_only" || mode == "phase2_only" || mode == "both_phases") {
            experimentMode = mode;
        } else {
            std::cerr << "无效的实验模式: " << mode << "，使用默认模式: both_phases" << std::endl;
            experimentMode = "both_phases";
        }
    }

    // 向后兼容的旧配置方法
    void SLIHeuristicProver::setConfig(const std::string &python, const std::string &script,
                                       const std::string &model, const std::string &tokenizer)
    {
        pythonPath = python;
        phase1ScriptPath = script;
        phase1ModelPath = model;
        phase1TokenizerPath = tokenizer;
        
        // 清空第二阶段配置，表示只使用第一阶段
        phase2ScriptPath = "";
        phase2ModelPath = "";
        phase2TokenizerPath = "";
    }

    void SLIHeuristicProver::setMaxIterations(int iterations)
    {
        maxIterations = iterations;
    }
}