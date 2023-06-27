#!/bin/bash

sudo apt install git-lfs
git lfs install

git clone https://huggingface.co/decapoda-research/llama-13b-hf
cd llama-13b-hf
git lfs pull