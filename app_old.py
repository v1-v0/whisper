import os
import time
from datetime import datetime
import logging
import configparser
import soundfile as sf
import torch
import whisper
from pathlib import Path
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.ini') -> configparser.ConfigParser:
    """Load configuration from an INI file or create default if not exists."""
    config = configparser.ConfigParser()
    default_config = {
        'DEFAULT': {
            'audio_file': '',  # Will be set dynamically from 'source' folder
            'whisper_model': 'large-v3-turbo',
            'output_dir': 'outputs'
        }
    }
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, creating default")
        config.read_dict(default_config)
        with open(config_path, 'w') as configfile:
            config.write(configfile)
    else:
        config.read(config_path)
    return config

def validate_audio_file(file_path: str) -> bool:
    """Validate if the audio file exists and has a supported extension."""
    supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    if not os.path.isfile(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return False
    if not any(file_path.lower().endswith(ext) for ext in supported_extensions):
        logger.error(f"Unsupported audio file format: {file_path}")
        return False
    return True

def get_audio_duration(file_path: str) -> Optional[float]:
    """
    Returns the duration of the audio file in seconds using sf.info.
    Handles errors if the file cannot be read.
    """
    try:
        info = sf.info(file_path)
        return info.frames / float(info.samplerate)
    except sf.SoundFileError as e:
        logger.error(f"Error reading audio file '{file_path}': {e}")
        return None
    except FileNotFoundError:
        logger.error(f"Audio file not found: {file_path}")
        return None

def format_duration(seconds: float) -> Tuple[int, int]:
    """Convert seconds to hours and minutes."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return hours, minutes

def transcribe_audio(model: whisper.Whisper, audio_file: str, use_fp16: bool = False) -> Optional[dict]:
    """Transcribe audio file using Whisper model."""
    try:
        return model.transcribe(audio_file, fp16=use_fp16)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None

def save_transcription(result: dict, audio_file: str, model_name: str, start_time: float, output_dir: str) -> None:
    """Save transcription to a file."""
    try:
        model_name_for_file = model_name.replace('.', '_')
        filename = Path(audio_file).stem
        start_dt = datetime.fromtimestamp(start_time)
        output_filename = f"{filename}_{model_name_for_file}_{start_dt.strftime('%Y%m%d%H%M')}.txt"
        output_path = Path(output_dir) / output_filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result['text'])
        logger.info(f"Transcription written to: {output_path}")
        print(f"\nSummary:\n  Audio file: {audio_file}\n  Model: {model_name}\n  Output: {output_path}\n")
    except Exception as e:
        logger.error(f"Error saving transcription: {e}")

def main():
    """Main function to orchestrate audio transcription."""
    import sys
    try:
        # Load configuration
        config = load_config()
        whisper_model: str = config['DEFAULT']['whisper_model']
        output_dir: str = config['DEFAULT']['output_dir']

        # Set audio_file to the first file in 'source' folder if available
        source_dir = 'source'
        audio_file: Optional[str] = None
        if os.path.isdir(source_dir):
            files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
            if len(files) == 1:
                audio_file = os.path.join(source_dir, files[0])
                logger.info(f"Using audio file from 'source': {audio_file}")
            elif len(files) == 0:
                logger.error(f"No files found in '{source_dir}' directory.")
                sys.exit(1)
            else:
                logger.warning(f"Multiple files found in '{source_dir}': {files}. Using the first: {files[0]}")
                audio_file = os.path.join(source_dir, files[0])
        else:
            logger.error(f"Directory '{source_dir}' does not exist.")
            sys.exit(1)

        # Validate audio file
        if not validate_audio_file(audio_file):
            sys.exit(1)

        # Get audio duration
        audio_length = get_audio_duration(audio_file)
        if audio_length is None:
            logger.error("Aborting due to audio file error")
            sys.exit(1)
        src_hours, src_minutes = format_duration(audio_length)
        logger.info(f"Audio Duration: {src_hours} hours, {src_minutes} minutes")

        # Check device availability
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load Whisper model
        try:
            model = whisper.load_model(whisper_model).to(device)
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            sys.exit(1)

        # Start transcription
        start_time = time.time()
        start_dt = datetime.fromtimestamp(start_time)
        logger.info(f"Start Time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

        # Transcribe
        result = transcribe_audio(model, audio_file)
        if result is None:
            sys.exit(1)

        # Log duration
        end_time = time.time()
        duration = end_time - start_time
        d_hours, d_minutes = format_duration(duration)
        logger.info(f"Transcription Duration: {d_hours} hours, {d_minutes} minutes")

        # Save transcription
        save_transcription(result, audio_file, whisper_model, start_time, output_dir)

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()