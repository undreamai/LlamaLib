#pragma once
#include "defs.h"
#include <iostream>
#include <string.h>
#include <mutex>

class UNDREAMAI_API StringWrapper {
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

extern "C" {
    UNDREAMAI_API StringWrapper* StringWrapper_Construct();
    UNDREAMAI_API void StringWrapper_Delete(StringWrapper* object);
    UNDREAMAI_API const int StringWrapper_GetStringSize(StringWrapper* object);
    UNDREAMAI_API void StringWrapper_GetString(StringWrapper* object, char* buffer, int bufferSize, bool clear = false);
    UNDREAMAI_API char* GetStringWrapperContent(StringWrapper* stringWrapper);
}
