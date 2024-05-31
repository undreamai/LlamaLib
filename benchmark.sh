#!/bin/bash

command=$1
benchmark=$2
runs=$3

log=$benchmark.log
prompt="\"write me anything you know about dogs\""
seed=1
num=0
prenum=0

$command -m model.gguf --port 13333 -sm none -mg 0 > $log &
pid=$!
echo $pid
sleep 2;
if [ ! -d /proc/$pid ]; then exit 0; fi
while [ 1 ];do
  ex=`cat $log | grep "all slots are idle"`
  if [ "$ex" != "" ]; then break; fi
done

while [ $num -lt $runs ];do
  curl --request POST --url http://localhost:13333/completion   --header "Content-Type: application/json"   --data "{\"prompt\": $prompt, \"n_predict\": 256, \"stream\": true, \"temperature\": 1, \"seed\": $seed}" > /tmp/la
  cat $log|grep generation|grep "256 runs" | grep -oP '\d+(\.\d+)?(?= tokens per second)' > $benchmark
  num=`cat $benchmark|wc -l`
  if [ $num -eq $prenum ];then
    seed=$((seed+1))
  else
    prenum=$num
  fi
done

kill -9 $pid
rm $log