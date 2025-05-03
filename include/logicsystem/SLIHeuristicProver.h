// SLIHeuristicProver.h
#ifndef SLI_HEURISTIC_PROVER_H
#define SLI_HEURISTIC_PROVER_H

#include <string>
#include "KnowledgeBase.h"
#include "Clause.h"
#include "SLITree.h"

namespace LogicSystem
{
    class SLIHeuristicProver
    {
    private:
        KnowledgeBase &kb;
        Clause goal;
        
        // 神经网络模型配置
        std::string pythonPath = "python";  // 可根据系统配置修改
        
        // 第一阶段模型配置
        std::string phase1ScriptPath = "../../../neural_network/neural_server/neural_server.py";
        std::string phase1ModelPath = "../../../neural_network/train/first_stage_model_revise.pth";
        std::string phase1TokenizerPath = "../../../neural_network/unified_tokenizer.pkl";
        
        // 第二阶段模型配置
        std::string phase2ScriptPath = "../../../neural_network/neural_server/neural_server2.py";
        std::string phase2ModelPath = "../../../neural_network/second_stage_model.pth";
        std::string phase2TokenizerPath = "../../../neural_network/unified_tokenizer.pkl";
        
        // 实验模式设置（可选）
        std::string experimentMode = "both_phases"; // 可选: "phase1_only", "phase2_only", "both_phases"
        
        // 搜索配置
        int maxIterations = 10000;  // 最大迭代次数

    public:
        SLIHeuristicProver(KnowledgeBase &kb, const Clause &goal);
        SLIHeuristicProver(KnowledgeBase &kb, const Clause &goal, int setMaxIterations) 
            : maxIterations(setMaxIterations), kb(kb), goal(goal) {};
            
        SearchResult prove(const std::string &save_dir = "");
        
        // 设置第一阶段配置参数
        void setPhase1Config(const std::string& python, const std::string& script, 
                            const std::string& model, const std::string& tokenizer);
        
        // 设置第二阶段配置参数
        void setPhase2Config(const std::string& script, const std::string& model, 
                            const std::string& tokenizer);
        
        // 设置实验模式
        void setExperimentMode(const std::string& mode);
        
        // 向后兼容的旧设置方法 (只设置第一阶段)
        void setConfig(const std::string& python, const std::string& script, 
                      const std::string& model, const std::string& tokenizer);
        
        // 设置最大迭代次数
        void setMaxIterations(int iterations);
    };
}

#endif // SLI_HEURISTIC_PROVER_H