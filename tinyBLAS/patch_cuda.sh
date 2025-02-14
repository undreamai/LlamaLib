start()
{
    BASE_DIR=`pwd`
    TINYBLAS_DIR=`pwd`/tinyBLAS
    GGML_CUDA_DIR=`pwd`/llama.cpp/ggml/src/ggml-cuda
    version=`awk -F': ' '/LLAMACPP_VERSION:/ {print $2}' .github/workflows/build_library.yaml`

    cd $GGML_CUDA_DIR
    git checkout master
    git pull
    git log $version..master --format="%h" -- . > commits.new
    cat commits.new | wc -l
}

next()
{
    cd $GGML_CUDA_DIR
    num=`cat commits.new |wc -l`
    if [ $num -eq 0 ];then
      echo "FINISHED";
    else
        commit=`sed -n '$p' commits.new`
        sed -i '$d' commits.new
        echo $commit, `cat commits.new |wc -l` remaining
    fi
}

f1(){
    git checkout $commit
    cat $TINYBLAS_DIR/ggml-cuda.cu |grep ROLLUP|cut -d' ' -f3 > files
    ls *cu* ggml-cuda.cu template-instances/* |grep -v generate_cu_files.py >files.new
    cat files files.new |sort |uniq -u
}

f2(){
    files=`cat files`" "`cat files files.new |sort |uniq -u`
    python3 $TINYBLAS_DIR/rollup.py $files > rollup
    cp rollup rollup_patched
    patch rollup_patched -i $TINYBLAS_DIR/ggml-cuda.cu.rollup.patch
}

f3(){
    git diff --no-index $TINYBLAS_DIR/ggml-cuda.cu.rollup rollup
}

f3d(){
    git diff --no-index $TINYBLAS_DIR/ggml-cuda.cu rollup_patched
}

f4(){
    cp rollup_patched $TINYBLAS_DIR/ggml-cuda.cu
    cp rollup $TINYBLAS_DIR/ggml-cuda.cu.rollup
    git diff --no-index rollup rollup_patched > $TINYBLAS_DIR/ggml-cuda.cu.rollup.patch
}

f5()
{
    rm files files.new rollup rollup_patched rollup_patched.orig
    cd $BASE_DIR
    sed -i "s/^\( *LLAMACPP_VERSION: *\).*/\1$commit/" .github/workflows_template/build_library_template.yaml
    sed -i "s/^\( *LLAMACPP_VERSION: *\).*/\1$commit/" .github/workflows/build_library.yaml
    git add .github/workflows/build_library.yaml .github/workflows_template/build_library_template.yaml tinyBLAS/ggml-cuda.cu tinyBLAS/ggml-cuda.cu.rollup tinyBLAS/ggml-cuda.cu.rollup.patch
    git commit -m $commit
}