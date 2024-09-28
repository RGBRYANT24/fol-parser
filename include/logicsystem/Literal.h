#ifndef LOGIC_SYSTEM_LITERAL_H
#define LOGIC_SYSTEM_LITERAL_H

#include <vector>
#include <string>

namespace LogicSystem
{
    class KnowledgeBase;

    class Literal
    {
    public:
        Literal(int predId, const std::vector<int>& argIds, bool negated);

        std::string toString(const KnowledgeBase& kb) const;

        int getPredicateId() const;
        const std::vector<int>& getArgumentIds() const;
        bool isNegated() const;


        
    private:
        int predicateId;
        std::vector<int> argumentIds;
        bool negated;
        
    };
}

#endif // LOGIC_SYSTEM_LITERAL_H