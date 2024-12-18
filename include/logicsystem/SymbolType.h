// SymbolId.h
#ifndef LOGIC_SYSTEM_SYMBOL_ID_H
#define LOGIC_SYSTEM_SYMBOL_ID_H
#include <functional>
namespace LogicSystem
{
    enum class SymbolType
    {
        CONSTANT,
        VARIABLE
    };

    struct SymbolId
    {
        SymbolType type;
        int id;

        bool operator==(const SymbolId &other) const
        {
            return type == other.type && id == other.id;
        }

        bool operator!=(const SymbolId &other) const
        {
            return !(*this == other);
        }

        // 添加 operator<
        bool operator<(const SymbolId &other) const
        {
            if (type != other.type)
            {
                return type < other.type;
            }
            return id < other.id;
        }
    };
}
// 在 std 命名空间中特化 hash 结构体
namespace std
{
    template <>
    struct hash<LogicSystem::SymbolId>
    {
        std::size_t operator()(const LogicSystem::SymbolId &symbolId) const
        {
            // 使用 std::hash 来哈希 SymbolType 和 id
            std::size_t h1 = std::hash<int>{}(static_cast<int>(symbolId.type));
            std::size_t h2 = std::hash<int>{}(symbolId.id);

            // 组合两个哈希值
            return h1 ^ (h2 << 1);
        }
    };
}

#endif // LOGIC_SYSTEM_SYMBOL_ID_H