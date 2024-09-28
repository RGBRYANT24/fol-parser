#ifndef LOGIC_SYSTEM_FACT_H
#define LOGIC_SYSTEM_FACT_H

#include <vector>
#include <string>

namespace LogicSystem
{
    class KnowledgeBase;

    class Fact
    {
    public:
        Fact(int predId, const std::vector<int>& argIds);

        int getPredicateId() const;
        const std::vector<int>& getArgumentIds() const;

        bool operator==(const Fact& other) const;

        std::string toString(const KnowledgeBase& kb) const;

    private:
        int predicateId;
        std::vector<int> argumentIds;
    };
}

#endif // LOGIC_SYSTEM_FACT_H