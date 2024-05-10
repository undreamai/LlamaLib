#include "stringwrapper.h"

StringWrapper::StringWrapper(){}

void StringWrapper::SetContent(std::string input){
    if (content != nullptr) {
        delete[] content;
    }
    content = new char[input.length() + 1];
    strcpy(content, input.c_str());
}

int StringWrapper::GetStringSize(){
    return strlen(content) + 1;
}

void StringWrapper::GetString(char* buffer, int bufferSize){
    strncpy(buffer, content, bufferSize);
    buffer[bufferSize - 1] = '\0';
}
