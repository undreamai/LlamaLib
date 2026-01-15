#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ ! -f $SCRIPT_DIR/model.gguf ];then
    curl -L -o $SCRIPT_DIR/model.gguf https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
fi
if [ ! -f $SCRIPT_DIR/model_embedding.gguf ];then
    curl -L -o $SCRIPT_DIR/model_embedding.gguf https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q4_k_m.gguf
fi