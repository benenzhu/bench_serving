#!/bin/bash

# 安装 lm_eval
pip install lm_eval[api]

# 下载 ShareGPT 数据集
mkdir -p /A/datasets
wget -P /A/datasets/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
