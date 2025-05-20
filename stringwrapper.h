#pragma once
#include "defs.h"
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

char* GetStringWrapperContent(StringWrapper* stringWrapper);

extern "C" {
    UNDREAMAI_API StringWrapper* StringWrapper_Construct();
    UNDREAMAI_API const void StringWrapper_Delete(StringWrapper* object);
    UNDREAMAI_API const int StringWrapper_GetStringSize(StringWrapper* object);
    UNDREAMAI_API const void StringWrapper_GetString(StringWrapper* object, char* buffer, int bufferSize, bool clear = false);
}
