conda create --name whisper311 python=3.11
pip install soundfile

https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-ubuntu.html
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
conda install -c conda-forge ffmpeg
pip install openai-whisper

.gitignore
/outputs
/source

Conda Environment Setup
Create a new Conda environment (Python 3.11 recommended):
conda create --name whisper311 python=3.11
conda activate whisper311

Install required Python packages:
pip install soundfile torch torchvision huggingface_hub

For Apple Silicon Macs (MLX-Whisper):
pip install mlx-whisper

For AMD ROCm (Linux): See: ROCm Install Guide

Install ffmpeg (for audio processing):
conda install -c conda-forge ffmpeg

Install Whisper:
pip install openai-whisper

(Optional) If using Hugging Face models:

Place your Hugging Face token in .hf_token or as specified in config.ini.
Configuration:

Edit config.ini to set your desired options (audio file, model, language, etc.).

Project Workflow and Logic
The main transcription logic is implemented in app-mlx.py. Here is an overview of how the code works:
Configuration Loading: Reads settings from config.ini or creates a default config if missing. This includes model selection, chunking options, language, and authentication.
Authentication: Handles Hugging Face authentication using environment variables, token files, or CLI login.
Audio File Detection: Automatically selects an audio file from the source directory. Validates file format and existence.
Audio Duration & Chunking: Calculates audio length and determines if chunking is needed (for long files). Chunks are created with overlap for smoother transcription.
Backend Selection: Chooses between MLX-Whisper (Apple Silicon) or standard Whisper (CPU/GPU) based on hardware and config. Loads the appropriate model.
Transcription:
If chunking is enabled, processes each chunk separately and merges results.
Otherwise, transcribes the whole file at once.
Supports both translation (e.g., Chinese to English) and transcription (preserving original language).
Saving Results: Formats and saves the transcription output to the outputs directory, including metadata such as model, backend, language, and timestamps.
Logging: All steps and errors are logged to transcription.log and the console for easy debugging and tracking.
Typical Usage: Place your audio file in the source folder, configure options in config.ini, and run the script. The output will be saved in the outputs folder.
