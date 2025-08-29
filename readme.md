conda create --name whisper311 python=3.11
pip install soundfile

https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-ubuntu.html
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
conda install -c conda-forge ffmpeg
pip install openai-whisper
