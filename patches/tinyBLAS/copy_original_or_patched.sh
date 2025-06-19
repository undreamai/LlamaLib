
#!/bin/bash
rm -r ggml-cuda-diff
mkdir -p ggml-cuda-diff/template-instances
for f in `find ggml-cuda -type f|sort`;do
    clear
    echo $f;
    d=`diff -w third_party/llama.cpp/ggml/src/$f $f`
    num=`diff -w third_party/llama.cpp/ggml/src/$f $f|wc -l`
    if [ $num -eq 0 ];then continue;fi
    echo $num
    echo "-------"
    diff -w third_party/llama.cpp/ggml/src/$f $f

    read -p "copy patched?" copy
    echo $copy
    outf=`echo $f|sed -e 's:ggml-cuda/:ggml-cuda-diff/:g'`
    if [ "$copy" == "y" ];then
        cp $f $outf;
    fi
done
