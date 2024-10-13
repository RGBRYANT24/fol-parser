#ifndef LOGIC_SYSTEM_SEARCHSTRATEGY_H
#define LOGIC_SYSTEM_SEARCHSTRATEGY_H

#include "KnowledgeBase.h"
#include "ResolutionPair.h"

namespace LogicSystem
{
    class SearchStrategy
    {
    public:
        virtual void addPair(const ResolutionPair &pair) = 0;
        virtual bool isEmpty() const = 0;
        virtual ResolutionPair getNext() = 0;
        virtual ~SearchStrategy() = default;
    };
}
#endif // LOGIC_SYSTEM_SEARCHSTRATEGY_H
