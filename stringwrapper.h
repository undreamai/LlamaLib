#pragma once
#include <iostream>
#include <string.h>
#include <mutex>

class StringWrapper {
    public:
        StringWrapper();
        ~StringWrapper();
        void Clear();
        void SetContent(const std::string& input);
        void AddContent(const std::string& input);
        int GetStringSize();
        void GetString(char* buffer, int bufferSize, bool clear=false);

    private:
        char *content = nullptr;
        std::mutex mtx;
};
