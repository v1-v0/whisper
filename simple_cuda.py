import torch
import whisper
import os
import sys
import soundfile as sf
import logging
import argparse
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_audio_file(file_path: str) -> bool:
    """Validate if the audio file exists and has a supported extension."""
    supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    if not os.path.isfile(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return False
    if not any(file_path.lower().endswith(ext) for ext in supported_extensions):
        logger.error(f"Unsupported audio file format: {file_path}. Supported: {', '.join(supported_extensions)}")
        return False
    return True

def get_audio_duration(file_path: str) -> float:
    """Return the duration of the audio file in seconds."""
    try:
        info = sf.info(file_path)
        return info.frames / info.samplerate
    except Exception as e:
        logger.error(f"Error reading audio duration: {e}")
        return 0.0

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper with CUDA.")
    parser.add_argument("audio_file", nargs="?", default=None, help="Path to audio file (optional, defaults to first file in 'source' directory)")
    parser.add_argument("--model", default="large-v3-turbo", choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'large-v3-turbo'], help="Whisper model to use")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for transcription")
    return parser.parse_args()

def main():
    args = parse_args()
    audio_file: Optional[str] = args.audio_file
    model_name: str = args.model
    output_dir: str = args.output_dir

    # Input handling
    if audio_file:
        if not validate_audio_file(audio_file):
            sys.exit(1)
    else:
        input_dir = "source"
        if not os.path.exists(input_dir):
            logger.error(f"Input directory '{input_dir}' does not exist.")
            sys.exit(1)
        files = [f for f in os.listdir(input_dir) if validate_audio_file(os.path.join(input_dir, f))]
        if not files:
            logger.error(f"No valid audio files found in '{input_dir}'.")
            sys.exit(1)
        elif len(files) > 1:
            logger.warning(f"Multiple audio files found in '{input_dir}': {files}. Using the first: {files[0]}")
        audio_file = os.path.join(input_dir, files[0])
    logger.info(f"Audio file to transcribe: {audio_file}")

    # Check audio duration
    duration = get_audio_duration(audio_file)
    if duration > 3600:
        logger.warning(f"Audio file is {duration/60:.1f} minutes long. Consider splitting for large files.")

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device != "cuda":
        logger.warning("CUDA not available, falling back to CPU with FP16 disabled. Ensure CUDA drivers and PyTorch with CUDA support are installed.")

    # Validate model name
    valid_models = {'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'large-v3-turbo'}
    if model_name not in valid_models:
        logger.error(f"Invalid model name: {model_name}. Valid models: {', '.join(valid_models)}")
        sys.exit(1)

    # Load Whisper model
    logger.info(f"Loading model '{model_name}'...")
    try:
        model = whisper.load_model(model_name, device=device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # Transcribe the audio file
    logger.info(f"Transcribing audio file '{audio_file}'...")
    try:
        result = model.transcribe(audio_file, fp16=(device == "cuda"))
        transcription: str = result['text']
        if not isinstance(transcription, str):
            logger.error(f"Expected string transcription, got {type(transcription)}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        sys.exit(1)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Output the transcription
    logger.info("Transcription Result:")
    print(transcription)  # Print to console for user visibility

    # Save transcription to a file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_{model_name.replace('.', '_')}_transcription.txt")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        logger.info(f"Transcription saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving transcription: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()