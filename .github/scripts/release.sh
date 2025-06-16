#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
root_dir=$script_dir/../..

build_dir=$1
release_dir=$2

mkdir -p $release_dir

includes=$(awk '/set\s*\(\s*include_paths/,/\)/' ../cmake/LlamaLibCommon.cmake | grep -o '".*"' | sed 's/"//g')
for d in ${includes};do
    mkdir -p $release_dir/$d
    cp $root_dir/$d/*.h $root_dir/$d/*.hpp $release_dir/$d
done

cp $root_dir/LICENSE $release_dir/
cp $root_dir/VERSION $release_dir/
cp $root_dir/cmake/*.cmake $release_dir/
if [ "$build_dir" != "$release_dir" ];then
    cp -R $build_dir/libs $release_dir/
fi
