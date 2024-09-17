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
