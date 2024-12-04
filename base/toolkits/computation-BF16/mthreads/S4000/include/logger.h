#pragma once
#include <string>
#include <fstream>
class logger {
    std::ofstream outFile;
    std::string FileName;
public:
    logger();
    ~logger();

    void print(std::string str);
    void print(double val);
    void print(float val);
    void print(int val);
    void print(unsigned int val);
};