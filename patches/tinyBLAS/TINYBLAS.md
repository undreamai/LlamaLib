# Add tinyBLAS from llamafile

## Clone llama.cpp and llamafile
``` bash
git clone https://github.com/ggerganov/llama.cpp.git
git clone https://github.com/Mozilla-Ocho/llamafile.git
```
## Copy tinyBLAS from llamafile
``` bash
mkdir tinyBLAS
cp llamafile/llamafile/tinyblas.h tinyBLAS
cp llamafile/llamafile/tinyblas.cu tinyBLAS
```

## Integrate tinyBLAS in ggml-cuda
- Remove llamafile backend code\:
``` bash
cat llamafile/llama.cpp/ggml-cuda.cu | grep -v g_backend |grep -v FLAG_log_disable > tinyBLAS/ggml-cuda.cu
```
- Remove the duplicate ggml_abort
- Add GGML_NO_IQUANTS to not build the iQuants
- Add line `extern "C" GGML_CALL int ggml_backend_cuda_reg_devices();` back in 
