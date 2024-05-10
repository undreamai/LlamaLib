#pragma once
#include <iostream>
#include <string.h>

class StringWrapper {
    private:
        char *content = nullptr;

    public:
        StringWrapper();
        void SetContent(std::string input);
        int GetStringSize();
        void GetString(char* buffer, int bufferSize);
};

typedef void (*CompletionCallback)();
class StringWrapperCallback{
    public:
        StringWrapperCallback(StringWrapper* stringWrapper_in, CompletionCallback callback_in) : stringWrapper(stringWrapper_in), callback(callback_in){}
        void Call(std::string content)
        {
            if (callback == nullptr) return;
            if (stringWrapper != nullptr) stringWrapper->SetContent(content);
            callback();
        }
    private:
        StringWrapper* stringWrapper = nullptr;
        CompletionCallback callback = nullptr;
};