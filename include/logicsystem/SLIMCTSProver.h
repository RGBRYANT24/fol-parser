#ifndef SLIMCTSPROVER_H
#define SLIMCTSPROVER_H

#include "KnowledgeBase.h"
#include "Clause.h"
#include "SLITree.h"

namespace LogicSystem {
    // 前置声明，实际定义请参考 SLIMCTSState.h 和 SLIMCTSAction.h
    class SLIMCTSState;
    class SLIMCTSAction;
}
namespace LogicSystem
{
    class SLIMCTSProver {
        public:
            // 构造函数：传入知识库引用和目标子句
            SLIMCTSProver(KnowledgeBase &kb, const Clause &goal);
        
            // 证明函数，返回证明是否成功
            // SearchResult prove(const std::string &save_dir="");
            bool prove(const std::string &save_dir="");
            SearchResult prove_search_result(const std::string &save_dir = "");
        
        private:
            KnowledgeBase &kb;
            Clause goal;
        
            // 采集训练样本的函数
            void collectTrainingSamples(const LogicSystem::SLIMCTSState &state);
        };
}


#endif // SLIMCTSPROVER_H