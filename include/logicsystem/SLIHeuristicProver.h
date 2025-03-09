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
        std::string scriptPath = "neural_server.py";
        std::string modelPath = "first_stage_model.pth";
        std::string tokenizerPath = "unified_tokenizer.pkl";

    public:
        SLIHeuristicProver(KnowledgeBase &kb, const Clause &goal);
        bool prove(const std::string &save_dir = "");
        
        // 设置配置参数
        void setConfig(const std::string& python, const std::string& script, 
                      const std::string& model, const std::string& tokenizer);
    };
}

#endif // SLI_HEURISTIC_PROVER_H