#!/bin/bash

version=v1.1.1

if [ ! -d undreamai ];then
  curl -o undreamai-$version-llamacpp.zip -L "https://github.com/undreamai/LlamaLib/releases/download/$version/undreamai-$version-llamacpp.zip"
  unzip undreamai-$version-llamacpp.zip -d undreamai
fi

if [ ! -f model.gguf ];then
  curl -o model.gguf -L "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf?download=true"
fi

mkdir -p undreamai/cuda/lib64
for cuda in 11.7.1 12.2.0;do
  if [ -d cuda-$cuda-linux ];then continue; fi
  curl -o cuda-$cuda-linux.zip -L "https://github.com/undreamai/LlamaLib/releases/download/$version/cuda-$cuda-linux.zip"
  unzip cuda-$cuda-linux.zip -d cuda-$cuda-linux
  for f in `pwd`/cuda-$cuda-linux/cuda/lib64/*;do ln -s $f undreamai/cuda/lib64;done
done


wget https://raw.githubusercontent.com/undreamai/LlamaLib/main/benchmark.sh
chmod +x benchmark.sh

archs="
noavx 
avx
avx2
avx512
clblast
cuda-cu11.7.1
cuda-cu12.2.0
"

rm -f benchmark.csv
for arch in ${archs};do
  ./benchmark.sh ./undreamai/linux-$arch/undreamai_server $arch.txt 10 >/dev/null 2>/dev/null
  results=`cat $arch.txt`
  echo "$arch "`echo $results` >> benchmark.csv
done