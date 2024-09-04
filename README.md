# llama.cpp libraries for UndreamAI

LLMUnity server CLI startup guide (Linux):

1) Make a directory for your server

2) Within the directory, download the undreamai llamacpp and undreamai server binaries

wget https://github.com/undreamai/LlamaLib/releases/download/v1.1.10/undreamai-v1.1.10-llamacpp.zip

wget https://github.com/undreamai/LlamaLib/releases/download/v1.1.10/undreamai-v1.1.10-server.zip

3) Unzip both of the files in the same folder
unzip <filename>

4) Go to the folder with your device's architecture

5) You can now use the undreamai from the cli with .\undreamai_server executable

To start off, read the help page
.\undreamai_server -h  (for help) 

Quickstart

.\undreamai_server.exe -m {model gguf location} -ngl {GPU Layers} --port {Port to listen on} --template {chat template}

A good default template is 'chatml'
You can print a list of chat templates supported by LLMUnity with the following command in Unity:
Debug.Log(ChatTemplate.templates.Keys);

They are also visible in the LLM gameobject from the samples

6) Once your server is running from the CLI, you can now connect to it within your unity project, simply go to your LLMCharacters game object, flick the remote option and specify your server's IP and port
