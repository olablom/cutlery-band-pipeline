@echo off
REM scripts/install_pytorch_cuda.bat
REM Install PyTorch with CUDA support for RTX 5090 (Windows)

echo Installing PyTorch with CUDA support...
echo.
echo For CUDA 12.1 (stable):
echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.
echo For CUDA 12.4 nightly (latest):
echo   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
echo.

REM Install CUDA 12.1 stable by default
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

pause

