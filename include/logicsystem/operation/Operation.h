#ifndef LOGIC_SYSTEM_OPERATION_H
#define LOGIC_SYSTEM_OPERATION_H

namespace LogicSystem
{
    class Operation
    {
    public:
        virtual ~Operation() = default;
        virtual void undo() = 0;
    };
}

#endif // LOGIC_SYSTEM_OPERATION_H