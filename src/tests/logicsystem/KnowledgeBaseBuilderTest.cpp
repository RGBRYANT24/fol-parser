#include <gtest/gtest.h>
#include "KnowledgeBaseBuilder.h"
// #include "LogicSystem.h"
#include <filesystem>
#include <fstream>

class KnowledgeBaseBuilderTest : public ::testing::Test {
protected:
    LogicSystem::KnowledgeBaseBuilder builder;
    LogicSystem::KnowledgeBase kb;
    const std::string test_input_dir = "kb_builder_test_inputs";

    void SetUp() override {
        // 确保测试目录存在
        std::filesystem::create_directory(test_input_dir);
    }

    void TearDown() override {
        // 清理测试文件
        for (const auto& entry : std::filesystem::directory_iterator(test_input_dir)) {
            std::filesystem::remove(entry.path());
        }
    }

    void createTestFile(const std::string& filename, const std::vector<std::string>& lines) {
        std::ofstream file(test_input_dir + "/" + filename);
        for (const auto& line : lines) {
            file << line << std::endl;
        }
        file.close();
    }
};

TEST_F(KnowledgeBaseBuilderTest, ParseSingleFile) {
    createTestFile("test1.txt", {"P(x)", "Q(y)", "R(z)"});

    ASSERT_TRUE(builder.parseDirectory(test_input_dir, kb));
    EXPECT_EQ(kb.getClauses().size(), 1);
    EXPECT_TRUE(kb.getPredicateId("P").has_value());
    EXPECT_TRUE(kb.getPredicateId("Q").has_value());
    EXPECT_TRUE(kb.getPredicateId("R").has_value());
    kb.print();
}

// TEST_F(KnowledgeBaseBuilderTest, ParseMultipleFiles) {
//     createTestFile("test1.txt", {"P(x)", "Q(y)"});
//     createTestFile("test2.txt", {"R(z)", "S(w)"});

//     ASSERT_TRUE(builder.parseDirectory(test_input_dir, kb));
//     EXPECT_EQ(kb.getClauseCount(), 2);
//     EXPECT_EQ(kb.getPredicateCount(), 4);
//     EXPECT_EQ(kb.getVariableCount(), 4);
// }

// TEST_F(KnowledgeBaseBuilderTest, ParseWithConstants) {
//     createTestFile("test.txt", {"P(a)", "Q(b, c)", "R(x, y, z)"});

//     ASSERT_TRUE(builder.parseDirectory(test_input_dir, kb));
//     EXPECT_EQ(kb.getClauseCount(), 1);
//     EXPECT_EQ(kb.getPredicateCount(), 3);
//     EXPECT_EQ(kb.getConstantCount(), 3);
//     EXPECT_EQ(kb.getVariableCount(), 3);
// }

TEST_F(KnowledgeBaseBuilderTest, ParseWithNegation) {
    createTestFile("test.txt", {"P(x)", "~Q(y)", "R(z)"});

    ASSERT_TRUE(builder.parseDirectory(test_input_dir, kb));
    EXPECT_EQ(kb.getClauses().size(), 1);
    EXPECT_TRUE(kb.getPredicateId("P").has_value());
    EXPECT_TRUE(kb.getPredicateId("Q").has_value());
    EXPECT_TRUE(kb.getPredicateId("R").has_value());
    std::cout << "验证 Q 是否被正确地识别为否定" << std::endl;
    kb.print();
    // 验证 Q 是否被正确地识别为否定
    // 这里需要 KnowledgeBase 类提供一个方法来检查特定谓词的否定状态
    // EXPECT_TRUE(kb.isPredicateNegated("Q"));
}

TEST_F(KnowledgeBaseBuilderTest, ParseEmptyFile) {
    createTestFile("empty.txt", {});

    ASSERT_TRUE(builder.parseDirectory(test_input_dir, kb));
    EXPECT_EQ(kb.getClauses().size(), 0);
}

TEST_F(KnowledgeBaseBuilderTest, ParseInvalidSyntax) {
    createTestFile("invalid.txt", {"P(x", "Q(y))", "R(,z)"});

    ASSERT_FALSE(builder.parseDirectory(test_input_dir, kb));
    EXPECT_EQ(kb.getClauses().size(), 0);
}

// 如果您的 KnowledgeBaseBuilder 支持解析复杂的逻辑表达式，可以添加更多测试
TEST_F(KnowledgeBaseBuilderTest, ParseComplexExpression) {
    createTestFile("complex.txt", {"P(x) & Q(y) | ~R(z)"});

    ASSERT_TRUE(builder.parseDirectory(test_input_dir, kb));
    // 根据您的实现，验证复杂表达式是否被正确解析
}
