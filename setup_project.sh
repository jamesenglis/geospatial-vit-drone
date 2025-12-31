#!/bin/bash
echo "Starting setup..."
conda create -n geospatial-vit python=3.9 -y
conda activate geospatial-vit
pip install torch torchvision numpy pandas matplotlib pillow
pip install -e .
echo "Setup complete!"
