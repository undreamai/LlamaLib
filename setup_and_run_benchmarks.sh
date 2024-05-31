#!/bin/bash

version=v1.1.1
runs=5

################### SETUP ###################
if [ ! -d undreamai ];then
  curl -o undreamai-$version-llamacpp.zip -L "https://github.com/undreamai/LlamaLib/releases/download/$version/undreamai-$version-llamacpp.zip"
  unzip undreamai-$version-llamacpp.zip -d undreamai
fi

if [ ! -f model.gguf ];then
  curl -o model.gguf -L "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf?download=true"
fi

for lfv in 0.6.2 0.8.6;do
  if [ -f llamafile-$lfv ];then continue; fi
  curl -o llamafile-$lfv -L "https://github.com/Mozilla-Ocho/llamafile/releases/download/$lfv/llamafile-$lfv"
done

ape=ape-`uname -m`.elf
if [ ! -f $ape ];then
  curl -o $ape https://cosmo.zip/pub/cosmos/bin/$ape
fi

mkdir -p undreamai/cuda/lib64
for cuda in 11.7.1 12.2.0;do
  if [ -d cuda-$cuda-linux ];then continue; fi
  curl -o cuda-$cuda-linux.zip -L "https://github.com/undreamai/LlamaLib/releases/download/$version/cuda-$cuda-linux.zip"
  unzip cuda-$cuda-linux.zip -d cuda-$cuda-linux
  for f in `pwd`/cuda-$cuda-linux/cuda/lib64/*;do ln -s $f undreamai/cuda/lib64;done
done

curl -o benchmark.sh -L "https://raw.githubusercontent.com/undreamai/LlamaLib/main/benchmark.sh"
chmod +x -R .

################### EXPERIMENTS ###################

lscpu > system_info.txt
nvidia-smi >> system_info.txt

rm -f benchmark.csv

archs="
noavx 
avx
avx2
avx512
"
for arch in ${archs};do
  ./benchmark.sh "./undreamai/linux-$arch/undreamai_server -ngl 0" $arch.txt $runs >/dev/null 2>/dev/null
  results=`cat $arch.txt`
  echo "$arch "`echo $results` >> benchmark.csv
done >> experiments.txt

archs="
clblast
cuda-cu11.7.1
cuda-cu12.2.0
"
for arch in ${archs};do
  ./benchmark.sh "./undreamai/linux-$arch/undreamai_server -ngl 9999" $arch.txt $runs >/dev/null 2>/dev/null
  results=`cat $arch.txt`
  echo "$arch "`echo $results` >> benchmark.csv
done

for lfv in 0.6.2 0.8.6;do
  arch=llamafile-$lfv-cpu
  ./benchmark.sh "./$ape ./llamafile-$lfv --nobrowser --nocompile -ngl 0" $arch.txt $runs >/dev/null 2>/dev/null
  results=`cat $arch.txt`
  echo "$arch "`echo $results` >> benchmark.csv
done

for lfv in 0.6.2 0.8.6;do
  arch=llamafile-$lfv-tinyblas
  ./benchmark.sh "./$ape ./llamafile-$lfv --nobrowser --nocompile -ngl 9999" $arch.txt $runs >/dev/null 2>/dev/null
  results=`cat $arch.txt`
  echo "$arch "`echo $results` >> benchmark.csv
done

for lfv in 0.6.2 0.8.6;do
  arch=llamafile-$lfv-cuda
  ./benchmark.sh "./$ape ./llamafile-$lfv --nobrowser -ngl 9999" $arch.txt $runs >/dev/null 2>/dev/null
  results=`cat $arch.txt`
  echo "$arch "`echo $results` >> benchmark.csv
done
