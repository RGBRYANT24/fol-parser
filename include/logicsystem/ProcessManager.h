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

class ProcessManager {
private:
    pid_t childPid = -1;
    int inputPipe[2];   // 用于写入到子进程的stdin
    int outputPipe[2];  // 用于从子进程的stdout读取
    bool processActive = false;

    // 缓冲区大小
    static constexpr size_t BUFFER_SIZE = 8192; // 增大缓冲区以处理更复杂的状态

    // 启动Python进程
    bool startProcess(const std::string& pythonPath, const std::string& scriptPath, 
                     const std::string& modelPath, const std::string& tokenizerPath) {
        // 创建管道
        if (pipe(inputPipe) < 0 || pipe(outputPipe) < 0) {
            std::cerr << "创建管道失败" << std::endl;
            return false;
        }

        // 创建子进程
        childPid = fork();
        if (childPid < 0) {
            std::cerr << "创建进程失败" << std::endl;
            return false;
        }

        if (childPid == 0) {
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
                "--tokenizer", tokenizerPath
            };
            
            std::vector<char*> args_c;
            for (const auto& arg : args) {
                args_c.push_back(const_cast<char*>(arg.c_str()));
            }
            args_c.push_back(nullptr);

            execvp(args_c[0], args_c.data());
            
            // 如果执行到这里，说明execvp失败
            std::cerr << "执行Python脚本失败: " << strerror(errno) << std::endl;
            exit(1);
        } else {
            // 父进程代码
            // 关闭不需要的管道端
            close(inputPipe[0]);
            close(outputPipe[1]);
        }

        processActive = true;
        
        // 等待Python进程准备就绪
        std::string readyMsg;
        if (!readResponse(readyMsg)) {
            std::cerr << "等待Python进程准备就绪时出错" << std::endl;
            return false;
        }
        
        if (readyMsg != "READY") {
            std::cerr << "Python进程未能正确启动，收到: " << readyMsg << std::endl;
            return false;
        }
        
        std::cout << "Python神经网络服务已启动" << std::endl;
        return true;
    }

    // 读取Python进程的响应
    bool readResponse(std::string& response) {
        response.clear();
        char buffer[BUFFER_SIZE];
        
        ssize_t bytesRead = read(outputPipe[0], buffer, BUFFER_SIZE - 1);
        if (bytesRead <= 0) {
            std::cerr << "读取响应失败: " << (bytesRead < 0 ? strerror(errno) : "子进程关闭") << std::endl;
            return false;
        }
        buffer[bytesRead] = '\0';

        response = buffer;
        // 移除尾部的换行符
        if (!response.empty() && response.back() == '\n') {
            response.pop_back();
        }
        return true;
    }

public:
    ProcessManager() {}
    
    ~ProcessManager() {
        stopProcess();
    }

    // 初始化并启动Python进程
    bool initialize(const std::string& pythonPath, const std::string& scriptPath, 
                   const std::string& modelPath, const std::string& tokenizerPath) {
        return startProcess(pythonPath, scriptPath, modelPath, tokenizerPath);
    }

    // 向Python进程发送请求并获取响应
    json sendRequest(const json& request) {
        if (!processActive) {
            return {{"status", "error"}, {"error_message", "Python进程未运行"}};
        }

        std::string requestStr = request.dump() + "\n";
        
        ssize_t bytesWritten = write(inputPipe[1], requestStr.c_str(), requestStr.size());
        if (bytesWritten != static_cast<ssize_t>(requestStr.size())) {
            std::cerr << "写入请求失败: " << strerror(errno) << std::endl;
            return {{"status", "error"}, {"error_message", "写入请求失败: " + std::string(strerror(errno))}};
        }

        std::string responseStr;
        if (!readResponse(responseStr)) {
            return {{"status", "error"}, {"error_message", "读取响应失败"}};
        }

        try {
            return json::parse(responseStr);
        } catch (const json::parse_error& e) {
            return {{"status", "error"}, {"error_message", "解析响应失败: " + std::string(e.what())}};
        }
    }

    // 停止Python进程
    void stopProcess() {
        if (!processActive) {
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
        nanosleep(&ts, NULL);  // 等待1秒
        
        // 检查进程是否已结束
        int waitResult = waitpid(childPid, &status, WNOHANG);
        if (waitResult == 0) {
            // 进程仍在运行，发送终止信号
            std::cout << "发送SIGTERM终止Python进程..." << std::endl;
            kill(childPid, SIGTERM);
            nanosleep(&ts, NULL);  // 再等待1秒
            
            waitResult = waitpid(childPid, &status, WNOHANG);
            if (waitResult == 0) {
                // 如果仍未结束，发送强制终止信号
                std::cout << "发送SIGKILL强制终止Python进程..." << std::endl;
                kill(childPid, SIGKILL);
                waitpid(childPid, &status, 0);  // 等待进程终止
            }
        }
        
        // 关闭管道
        close(inputPipe[1]);
        close(outputPipe[0]);

        processActive = false;
        std::cout << "Python神经网络服务已停止" << std::endl;
    }

    // 检查进程是否活动
    bool isActive() const {
        return processActive;
    }
};

#endif // PROCESS_MANAGER_H