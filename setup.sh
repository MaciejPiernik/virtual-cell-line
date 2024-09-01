#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Update and install Python 3.11
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-distutils -y

# Install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.11 get-pip.py
rm get-pip.py

# Create and activate virtual environment
python3.11 -m venv vcl-env
source vcl-env/bin/activate

# Install requirements
pip install -r requirements_train.txt

# Create data directory
mkdir -p /data

# Print completion message
echo "Setup completed successfully!"