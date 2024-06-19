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
### Option 1: Copy ggml-cuda.cu from llamafile
This is the easiest and best option but depends on the latest llama.cpp upstream of llamafile:
``` bash
cat llamafile/llama.cpp/ggml-cuda.cu | grep -v g_backend |grep -v FLAG_log_disable > tinyBLAS/ggml-cuda.cu
```

### Option 2: Generate ggml-cuda.cu including tinyBLAS
- Note the files added for CUDA in the llama.cpp/CMakeLists.txt
- Roll them up in a single file:
``` bash
cd llama.cpp/ggml-cuda
python3 ../../llamafile/llamafile/rollup.py *.cu* template-instances/fattn-wmma*.cu template-instances/mmq*.cu template-instances/fattn-vec*q4_0-q4_0.cu template-instances/fattn-vec*q8_0-q8_0.cu template-instances/fattn-vec*f16-f16.cu ../ggml-cuda.cu > ../../tinyBLAS/ggml-cuda.cu.rollup
cd ../..
```

- Check that no includes of the ggml-cuda folder exist in there:
``` bash
cat tinyBLAS/ggml-cuda.cu.rollup | grep include
```

- if we are on the same llama.cpp HEAD in both llama.cpp and llamafile we can get the latest diff applied after rollup:
``` bash
git diff --no-index --color-words -- tinyBLAS/ggml-cuda.cu.rollup llamafile/llama.cpp/ggml-cuda.cu > tinyBLAS/ggml-cuda.cu
```

- Manually patch the differences in ggml-cuda.cu.diff and save to tinyBLAS/ggml-cuda.cu

## Create final folder of tinyBLAS
``` bash
release="..."
mkdir tinyBLAS/$release
mv tinyBLAS/ggml-cuda.cu tinyBLAS/tinyblas.h tinyBLAS/tinyblas.cu tinyBLAS/$release
```
