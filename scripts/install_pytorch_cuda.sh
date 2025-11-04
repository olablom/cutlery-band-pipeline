#!/bin/bash
# scripts/install_pytorch_cuda.sh
# Install PyTorch with CUDA support for RTX 5090

echo "Installing PyTorch with CUDA support..."
echo "Choose version:"
echo "1) CUDA 12.1 (stable)"
echo "2) CUDA 12.4 nightly (latest)"
read -p "Enter choice [1 or 2]: " choice

if [ "$choice" = "2" ]; then
    echo "Installing CUDA 12.4 nightly..."
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
else
    echo "Installing CUDA 12.1 stable..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

echo ""
echo "Verifying installation..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

