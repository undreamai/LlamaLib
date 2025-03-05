#!/bin/bash

#Specify Version
version=v1.2.3

# Create a server directory and navigate into it
mkdir -p server
cd server

sudo apt install -y zip wget

wget https://github.com/undreamai/LlamaLib/releases/download/$version/undreamai-$version-llamacpp.zip
wget https://github.com/undreamai/LlamaLib/releases/download/$version/undreamai-$version-server.zip

unzip undreamai-$version-llamacpp.zip
unzip undreamai-$version-server.zip
