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
    if(bufferSize > 1) strncpy(buffer, content, bufferSize);
    buffer[bufferSize - 1] = '\0';
}

StringWrapperCallback::StringWrapperCallback(StringWrapper* stringWrapper_in, CompletionCallback callback_in) : stringWrapper(stringWrapper_in), callback(callback_in){}

void StringWrapperCallback::Call(std::string content)
{
    if (callback == nullptr) return;
    if (stringWrapper != nullptr) stringWrapper->SetContent(content);
    callback();
}
