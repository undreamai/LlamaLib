# llama.cpp library for UndreamAI

LlamaLib implements an API for the [llama.cpp](https://github.com/ggerganov/llama.cpp) server.
The focus of this project is to:
- build the library in a cross-platform way to support most of the architectures available in llama.cpp
- expose the API in a way that can be imported in Unity (and C#), more specifically for the [LLMUnity](https://github.com/undreamai/LLMUnity) project

Each release contains:
- the built libraries for different architectures in the undreamai-[VERSION]-llamacpp.zip
- built libraries with additional functionality (iQ quants, flash attention) for Nvidia / AMD GPUs in the undreamai-[VERSION]-llamacpp-full.zip
- server binaries that can use the above libraries similarly to the llama.cpp server in the undreamai-[VERSION]-server.zip.<br>Note: you need to extract the libraries and the binaries in the same directory to use them.

The following architectures are provided:
- `*-noavx` (Windows/Linux): support for CPUs without AVX instructions (operates on all AVX as well)
- `*-avx` (Windows/Linux): support for CPUs with AVX instructions
- `*-avx2` (Windows/Linux): support for CPUs with AVX-2 instructions
- `*-avx512` (Windows/Linux): support for CPUs with AVX-512 instructions
- `*-cuda-cu11.7.1` (Windows/Linux): support for Nvidia GPUs with CUDA 11 (CUDA doesn't need to be separately installed)
- `*-cuda-cu12.2.0` (Windows/Linux): support for Nvidia GPUs with CUDA 11 (CUDA doesn't need to be separately installed)
- `*-hip` (Windows/Linux): support for AMD GPUs with AMD HIP (HIP doesn't need to be separately installed)
- `*-vulkan` (Windows/Linux): support for most GPUs independent of manufacturer
- `macos-*-acc` (macOS arm64/x64): support for macOS with the Accelerate framework
- `macos-*-no_acc` (macOS arm64/x64): support for macOS without the Accelerate framework

In addition the windows-archchecker and linux-archchecker libraries are used to determine the presence and type of AVX instructions in Windows and Linux.

The server CLI startup guide can be accessed by running the command `.\undreamai_server -h` on Linux/macOS or `undreamai_server.exe -h` on Windows for the architecture of interest.<br>
More information on the different options can be found on the [llama.cpp server Readme](https://github.com/ggerganov/llama.cpp/tree/master/examples/server).

The server binaries can be used to deploy remote servers for [LLMUnity](https://github.com/undreamai/LLMUnity).<br>
You can print the required command within Unity by running the scene.<br>
More information can be found at the `Use a remote server` section of the [LLMUnity Readme](https://github.com/undreamai/LLMUnity?tab=readme-ov-file#how-to-use).