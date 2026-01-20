restore()
{
    git reset --hard
    git restore .
    git clean -fd
}

start()
{
    BASE_DIR=~/codes/LlamaLib
    LLAMACPP_DIR=$BASE_DIR/third_party/llama.cpp
    GGML_CUDA_DIR=$LLAMACPP_DIR/ggml/src/ggml-cuda

    cd $LLAMACPP_DIR
    version=`git rev-parse HEAD`

    restore
    git checkout master
    git pull
    git log $version..master --format="%h" -- $GGML_CUDA_DIR > $BASE_DIR/commits.new
    cat $BASE_DIR/commits.new | wc -l
}

last_working()
{
    commits=()
    patch1=0
    patch2=0
    while IFS= read -r commit; do
        restore
        git checkout $commit
        git apply $BASE_DIR/patches/llama.cpp.patch
        patch1=$?
        git apply $BASE_DIR/patches/tinyBLAS.patch
        patch2=$?
        num=`expr $patch1 + $patch2`
        if [ $num -gt 0 ];then break; fi
        sed -i '$d' $BASE_DIR/commits.new
    done < <(tac "$BASE_DIR/commits.new")
    restore
    echo $commit
}

save_patches()
{
    rm `find . -name *.orig` `find . -name *.rej`
    git add common tools ggml
    git status
    git diff --cached  ggml/src/ggml-cuda > ../../patches/tinyBLAS.patch
    git diff --cached ':!ggml/src/ggml-cuda' > ../../patches/llama.cpp.patch
}


start

# iterate:
last_working
patch -p1 < $BASE_DIR/patches/llama.cpp.patch
patch -p1 < $BASE_DIR/patches/tinyBLAS.patch
save_patches

# check server
commit=`git rev-parse HEAD`
git log $version..$commit --format="%h" tools/server/server.cpp
git diff $version..$commit tools/server/server.cpp

restore
