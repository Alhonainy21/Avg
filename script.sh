#!/bin/bash
apt-get update -y
apt-get install nano -y
apt install net-tools -y
apt-get install curl -y
apt install python3-pip -y
pip install torch==1.11.0
pip install torchvision==0.12.0
pip install flwr==0.16.0
pip install numpy==1.22.1
pip install tqdm
pip install efficientnet_pytorch
pip install psutil
