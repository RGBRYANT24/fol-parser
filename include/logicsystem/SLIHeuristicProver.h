// SLIHeuristicProver.h
#ifndef SLI_HEURISTIC_PROVER_H
#define SLI_HEURISTIC_PROVER_H

#include <string>
#include "KnowledgeBase.h"
#include "Clause.h"

namespace LogicSystem
{
    class SLIHeuristicProver
    {
    private:
        KnowledgeBase &kb;
        Clause goal;
        
        // 神经网络模型配置
        std::string pythonPath = "python";  // 可根据系统配置修改
        std::string scriptPath = "../../../neural_network/neural_server/neural_server.py";
        std::string modelPath = "../../../neural_network/first_stage_model.pth";
        std::string tokenizerPath = "../../../neural_network/unified_tokenizer.pkl";
        
        // 搜索配置
        int maxIterations = 50000;  // 最大迭代次数

    public:
        SLIHeuristicProver(KnowledgeBase &kb, const Clause &goal);
        bool prove(const std::string &save_dir = "");
        
        // 设置配置参数
        void setConfig(const std::string& python, const std::string& script, 
                      const std::string& model, const std::string& tokenizer);
        
        // 设置最大迭代次数
        void setMaxIterations(int iterations);
    };
}

#endif // SLI_HEURISTIC_PROVER_H