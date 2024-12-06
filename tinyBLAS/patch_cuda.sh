commit=..

BASE_DIR=`pwd`
TINYBLAS_DIR=`pwd`/tinyBLAS
GGML_CUDA_DIR=llama.cpp/ggml/src/ggml-cuda

f1(){
    cd $BASE_DIR
    git checkout $commit
    cd $GGML_CUDA_DIR
    cat $TINYBLAS_DIR/ggml-cuda.cu |grep ROLLUP|cut -d' ' -f3 > files
    ls *cu* ggml-cuda.cu template-instances/* |grep -v generate_cu_files.py >files.new
    cat files files.new |sort |uniq -u
}

f2(){
    files=`cat files`" "`cat files files.new |sort |uniq -u`
    python3 $TINYBLAS_DIR/rollup.py $files > rollup
    cp rollup rollup_patched
    patch rollup_patched -i $TINYBLAS_DIR/ggml-cuda.rollup.patch
}

f3(){
    git diff --no-index $TINYBLAS_DIR/ggml-cuda.cu rollup_patched
}

f4(){
    cp rollup_patched $TINYBLAS_DIR/ggml-cuda.cu
    git diff --no-index rollup rollup_patched > $TINYBLAS_DIR/ggml-cuda.rollup.patch
}

f5(){
    cd $BASE_DIR
    sed -i 's/LLAMACPP_VERSION: .*/LLAMACPP_VERSION: b4dadsad/' .github/workflows/test.yaml
    git add .github/workflows/test.yaml tinyBLAS/ggml-cuda.cu tinyBLAS/ggml-cuda.rollup.patch
    git commit -m $commit
}