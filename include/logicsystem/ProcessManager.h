// ProcessManager.h
#ifndef PROCESS_MANAGER_H
#define PROCESS_MANAGER_H

#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <cstdio>
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

// Linux特定头文件
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>

using json = nlohmann::json;

class ProcessManager
{
private:
    pid_t childPid = -1;
    int inputPipe[2];  // 用于写入到子进程的stdin
    int outputPipe[2]; // 用于从子进程的stdout读取
    bool processActive = false;

    // 缓冲区大小
    static constexpr size_t BUFFER_SIZE = 8192; // 增大缓冲区以处理更复杂的状态

    // 启动Python进程
    bool startProcess(const std::string &pythonPath, const std::string &scriptPath,
                      const std::string &modelPath, const std::string &tokenizerPath)
    {
        // 创建管道
        if (pipe(inputPipe) < 0 || pipe(outputPipe) < 0)
        {
            std::cerr << "创建管道失败" << std::endl;
            return false;
        }

        // 创建子进程
        childPid = fork();
        if (childPid < 0)
        {
            std::cerr << "创建进程失败" << std::endl;
            return false;
        }

        if (childPid == 0)
        {
            // 子进程代码
            // 重定向标准输入/输出
            dup2(inputPipe[0], STDIN_FILENO);
            dup2(outputPipe[1], STDOUT_FILENO);

            // 关闭不需要的管道端
            close(inputPipe[0]);
            close(inputPipe[1]);
            close(outputPipe[0]);
            close(outputPipe[1]);

            // 执行Python脚本
            std::vector<std::string> args = {
                pythonPath,
                scriptPath,
                "--model", modelPath,
                "--tokenizer", tokenizerPath};

            std::vector<char *> args_c;
            for (const auto &arg : args)
            {
                args_c.push_back(const_cast<char *>(arg.c_str()));
            }
            args_c.push_back(nullptr);

            execvp(args_c[0], args_c.data());

            // 如果执行到这里，说明execvp失败
            std::cerr << "执行Python脚本失败: " << strerror(errno) << std::endl;
            exit(1);
        }
        else
        {
            // 父进程代码
            // 关闭不需要的管道端
            close(inputPipe[0]);
            close(outputPipe[1]);
        }

        processActive = true;

        // 等待Python进程准备就绪
        std::string readyMsg;
        if (!readResponse(readyMsg))
        {
            std::cerr << "等待Python进程准备就绪时出错" << std::endl;
            return false;
        }

        if (readyMsg != "READY")
        {
            std::cerr << "Python进程未能正确启动，收到: " << readyMsg << std::endl;
            return false;
        }

        std::cout << "Python神经网络服务已启动" << std::endl;
        std::cout << "ProcessManager::startProcess Model path " << modelPath << std::endl;
        return true;
    }

    // 读取Python进程的响应
    bool readResponse(std::string &response)
    {
        response.clear();
        char buffer[BUFFER_SIZE];
        std::string accumulated_response;

        // 首先等待并读取所有可用数据
        fd_set readfds;
        struct timeval tv;
        int ready;

        FD_ZERO(&readfds);
        FD_SET(outputPipe[0], &readfds);
        tv.tv_sec = 10; // 设置超时时间为10秒
        tv.tv_usec = 0;

        ready = select(outputPipe[0] + 1, &readfds, NULL, NULL, &tv);
        if (ready <= 0)
        {
            std::cerr << "等待响应超时或发生错误" << std::endl;
            return false;
        }

        ssize_t bytesRead = read(outputPipe[0], buffer, BUFFER_SIZE - 1);
        if (bytesRead <= 0)
        {
            std::cerr << "读取响应失败: " << (bytesRead < 0 ? strerror(errno) : "子进程关闭") << std::endl;
            return false;
        }
        buffer[bytesRead] = '\0';
        accumulated_response = buffer;

        // 特殊处理READY消息
        if (accumulated_response.find("READY") != std::string::npos)
        {
            response = "READY";
            return true;
        }

        // 处理多行输出，查找有效的JSON
        std::istringstream stream(accumulated_response);
        std::string line;
        std::string last_line;

        while (std::getline(stream, line))
        {
            // 移除尾部的空白
            line.erase(line.find_last_not_of(" \n\r\t") + 1);

            if (!line.empty())
            {
                // 尝试解析每一行，找到有效的JSON
                try
                {
                    auto parsed_json = json::parse(line);
                    // 如果成功解析，这是我们想要的行
                    response = line;
                    return true;
                }
                catch (...)
                {
                    // 不是有效的JSON，可能是调试输出
                    std::cerr << "跳过非JSON输出: " << line << std::endl;
                    last_line = line; // 保存最后一行，以防所有行都不是有效JSON
                }
            }
        }

        // 如果没有找到有效JSON，尝试使用最后一行
        response = last_line;

        // 打印完整响应供调试
        std::cerr << "完整响应: " << accumulated_response << std::endl;
        std::cerr << "选择的响应行: " << response << std::endl;

        return !response.empty();
    }

public:
    ProcessManager() {}

    ~ProcessManager()
    {
        stopProcess();
    }

    // 初始化并启动Python进程
    bool initialize(const std::string &pythonPath, const std::string &scriptPath,
                    const std::string &modelPath, const std::string &tokenizerPath)
    {
        return startProcess(pythonPath, scriptPath, modelPath, tokenizerPath);
    }

    // 向Python进程发送请求并获取响应
    json sendRequest(const json &request)
    {
        if (!processActive)
        {
            std::cout << "ProcessManager::sendRequest Python进程未运行 processActive " << processActive << std::endl;
            return {{"status", "error"}, {"error_message", "Python进程未运行"}};
        }

        std::string requestStr = request.dump() + "\n";
        // std::cout << "发送请求: " << requestStr << std::endl;

        ssize_t bytesWritten = write(inputPipe[1], requestStr.c_str(), requestStr.size());
        if (bytesWritten != static_cast<ssize_t>(requestStr.size()))
        {
            std::cerr << "写入请求失败: " << strerror(errno) << std::endl;
            return {{"status", "error"}, {"error_message", "写入请求失败: " + std::string(strerror(errno))}};
        }

        std::string responseStr;
        if (!readResponse(responseStr))
        {
            std::cout << "ProcessManager::sendRequest 读取响应失败" << std::endl;
            return {{"status", "error"}, {"error_message", "读取响应失败"}};
        }

        // std::cout << "接收到原始响应: [" << responseStr << "]" << std::endl;

        try
        {
            auto parsed_response = json::parse(responseStr);
            // std::cout << "成功解析为JSON" << std::endl;
            return parsed_response;
        }
        catch (const json::parse_error &e)
        {
            std::cerr << "解析响应失败: " << e.what() << std::endl;
            std::cerr << "原始响应: [" << responseStr << "]" << std::endl;

            // 尝试清理响应字符串
            if (responseStr.find("None") != std::string::npos)
            {
                std::cerr << "检测到'None'响应，替换为空JSON对象" << std::endl;
                return {{"status", "error"}, {"error_message", "Python返回了None而不是有效的JSON"}};
            }

            return {{"status", "error"}, {"error_message", "解析响应失败: " + std::string(e.what())}};
        }
    }

    // 停止Python进程
    void stopProcess()
    {
        if (!processActive)
        {
            return;
        }

        // 发送退出命令
        std::string exitCommand = "exit\n";
        write(inputPipe[1], exitCommand.c_str(), exitCommand.size());

        // 等待进程结束
        int status;
        struct timespec ts;
        ts.tv_sec = 1;
        ts.tv_nsec = 0;
        nanosleep(&ts, NULL); // 等待1秒

        // 检查进程是否已结束
        int waitResult = waitpid(childPid, &status, WNOHANG);
        if (waitResult == 0)
        {
            // 进程仍在运行，发送终止信号
            std::cout << "发送SIGTERM终止Python进程..." << std::endl;
            kill(childPid, SIGTERM);
            nanosleep(&ts, NULL); // 再等待1秒

            waitResult = waitpid(childPid, &status, WNOHANG);
            if (waitResult == 0)
            {
                // 如果仍未结束，发送强制终止信号
                std::cout << "发送SIGKILL强制终止Python进程..." << std::endl;
                kill(childPid, SIGKILL);
                waitpid(childPid, &status, 0); // 等待进程终止
            }
        }

        // 关闭管道
        close(inputPipe[1]);
        close(outputPipe[0]);

        processActive = false;
        std::cout << "Python神经网络服务已停止" << std::endl;
    }

    // 检查进程是否活动
    bool isActive() const
    {
        return processActive;
    }
};

#endif // PROCESS_MANAGER_H