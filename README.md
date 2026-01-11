# 🤟 Sign Language Real-Time Translator

A group project (5 members) aimed at creating a real-time sign language translation model, complete with an interactive Gradio interface.

# 🚀 Overview

This project focuses on building a machine learning pipeline capable of translating sign language gestures into text in real time.
It includes:

* 📷 Real-time video input
* 🤖 A trained sign-language recognition model
* 🖥️ A Gradio-based web interface
* 🔁 Smooth end-to-end inference loop

# Features

* Real-time gesture detection
* User-friendly Gradio UI
* Localhost interface for easy testing

# How to run

1. Install required dependencies, look for requirement.txt

```
pip install --upgrade gradio
```

1. Run the app

```
python app.py
```

1. Finally, open in local browser: <http://localhost:7860>

# 🧑‍🤝‍🧑 Contribution Guide

* Fork the repository
* Create new feature branch
* Commit and push to remote feature branch
* Create Pull Request before merging into main branch

# 🤝 Team Members

Contributed by Jia Herng, Jasper, Zoe, Zen Yu, Zhi Yang

```bash
uv run train-sign-language-model --train-npz .\data\wlasl_reduced\tensors\kps\train.npz --test-npz .\data\wlasl_reduced\tensors\kps\test.npz --gloss-map-path .\data\wlasl_reduced\gloss_map.json --modalities kps --epochs 500 --kp-model-type transformer

uv run train-sign-language-model --train-npz .\data\wlasl_reduced\tensors\kps\train.npz --test-npz .\data\wlasl_reduced\tensors\kps\test.npz --gloss-map-path .\data\wlasl_reduced\gloss_map.json --modalities kps --epochs 500 --kp-model-type lstm

uv run train-sign-language-model --train-npz .\data\wlasl_reduced\tensors\rgb\train.npz --test-npz .\data\wlasl_reduced\tensors\rgb\test.npz --gloss-map-path .\data\wlasl_reduced\gloss_map.json --modalities rgb --epochs 500

uv run train-sign-language-model --train-npz .\data\wlasl_reduced\tensors\rgb_flow\train.npz --test-npz .\data\wlasl_reduced\tensors\rgb_flow\test.npz --gloss-map-path .\data\wlasl_reduced\gloss_map.json --modalities rgb+flow --epochs 500

uv run train-sign-language-model --train-npz .\data\wlasl_reduced\tensors\kps_rgb_flow\train.npz --test-npz .\data\wlasl_reduced\tensors\kps_rgb_flow\test.npz --gloss-map-path .\data\wlasl_reduced\gloss_map.json --modalities kps+rgb+flow --epochs 500 --pretrained-kps-ckpt checkpoints\kps_transformer_bs32_lr0.001_1768166411\best_model.pth --pretrained-rgb-flow-ckpt checkpoints\rgb+flow_bs32_lr0.001_1768166994\best_model.pth
```
