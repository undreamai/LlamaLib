#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
root_dir=$script_dir/../..

# folder where the artifacts where downloaded
release_dir=$1

cd $release_dir
ls -R
# extract servers
mkdir servers
for arch in win-x64_noavx linux-x64_noavx osx-arm64_no-acc osx-x64_no-acc;do
    unzip -o $arch.zip/$arch.zip -d servers llamalib*server*
done

# extract runtimes
for d in *.zip;do
    platform=`echo $d|cut -d'.' -f1|cut -d'_' -f1`
    mkdir -p runtimes/$platform/native
    unzip -o $d/$d -d runtimes/$platform/native -x llamalib*server*
done

rm -r *.zip

# copy includes
includes=$(awk '/set\s*\(\s*include_paths/,/\)/' $root_dir/cmake/LlamaLibCommon.cmake | grep -o '".*"' | sed 's/"//g')
for d in ${includes};do
    mkdir -p $d
    cp $root_dir/$d/*.h* $d
done

# licenses
mkdir -p third_party_licenses
cp $root_dir/third_party/llama.cpp/LICENSE third_party_licenses/llama.cpp.LICENSE.txt
curl -o third_party_licenses/llamafile.LICENSE.txt -L https://raw.githubusercontent.com/Mozilla-Ocho/llamafile/main/LICENSE

# copy files from repo
cp $root_dir/LICENSE ./
cp $root_dir/VERSION ./
cp $root_dir/cmake/*.cmake ./
