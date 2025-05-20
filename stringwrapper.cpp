#include "stringwrapper.h"

StringWrapper::StringWrapper(): content(nullptr){}

StringWrapper::~StringWrapper(){
    Clear();
}

void StringWrapper::Clear(){
    if (content != nullptr) delete[] content;
    content = nullptr;
}

void StringWrapper::SetContent(const std::string& input){
    std::lock_guard<std::mutex> lock(mtx);
    Clear();
    content = new char[input.length() + 1];
    strcpy(content, input.c_str());
}

void StringWrapper::AddContent(const std::string& input) {
    std::lock_guard<std::mutex> lock(mtx);
    if (content == nullptr) {
        content = new char[input.length() + 1];
        strcpy(content, input.c_str());
    } else {
        char* newContent = new char[strlen(content) + input.length() + 1];
        strcpy(newContent, content);
        strcat(newContent, input.c_str());
        delete[] content;
        content = newContent;
    }
}

int StringWrapper::GetStringSize(){
    std::lock_guard<std::mutex> lock(mtx);
    int num = 0;
    if (content != nullptr) num = strlen(content) + 1;
    return num;
}

void StringWrapper::GetString(char* buffer, int bufferSize, bool clear){
    std::lock_guard<std::mutex> lock(mtx);
    if (buffer != nullptr && content != nullptr && bufferSize > 1){
        size_t copySize = std::min(strlen(content), static_cast<size_t>(bufferSize - 1));
        if(copySize>0) strncpy(buffer, content, copySize);
        buffer[bufferSize - 1] = '\0';
    }
    if(clear) Clear();
}

//============================= API =============================//

StringWrapper* StringWrapper_Construct() {
    return new StringWrapper();
}

const void StringWrapper_Delete(StringWrapper* object) {
    if (object != nullptr) {
        delete object;
        object = nullptr;
    }
}

const int StringWrapper_GetStringSize(StringWrapper* object) {
    if (object == nullptr) return 0;
    return object->GetStringSize();
}

const void StringWrapper_GetString(StringWrapper* object, char* buffer, int bufferSize, bool clear) {
    if (object == nullptr) return;
    return object->GetString(buffer, bufferSize, clear);
}

char* GetStringWrapperContent(StringWrapper* stringWrapper) {
    int bufferSize(stringWrapper->GetStringSize());
    char* content = new char[bufferSize];
    stringWrapper->GetString(content, bufferSize);
    return content;
}