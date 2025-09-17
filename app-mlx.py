import os
import time
from datetime import datetime
import logging
import configparser
import soundfile as sf
import torch
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
            'whisper_model': 'large-v3',  # Changed from large-v3-turbo for MLX compatibility
            'output_dir': 'outputs',
            'use_mlx': 'true',  # Use MLX-Whisper on Mac for better performance
            'force_cpu': 'false',  # Force CPU usage if needed
            'hf_token': '',  # Hugging Face token for private models
            'language': '',  # Auto-detect if empty, or specify language code (e.g., 'en', 'zh', 'ja')
            'task': 'transcribe'  # 'transcribe' keeps original language, 'translate' converts to English
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

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def check_mlx_availability() -> bool:
    """Check if MLX-Whisper is available."""
    try:
        import mlx_whisper
        return True
    except ImportError:
        return False

def setup_hf_authentication(hf_token: str = None) -> None:
    """Setup Hugging Face authentication if token is provided."""
    if hf_token and hf_token.strip():
        try:
            from huggingface_hub import login
            login(token=hf_token.strip())
            logger.info("Successfully authenticated with Hugging Face Hub")
        except Exception as e:
            logger.warning(f"Failed to authenticate with Hugging Face Hub: {e}")
    else:
        # Check if already logged in
        try:
            from huggingface_hub import whoami
            user_info = whoami()
            if user_info:
                logger.info(f"Already authenticated as: {user_info['name']}")
        except Exception:
            logger.info("No Hugging Face authentication found - using public models only")

def get_mlx_model_name(whisper_model: str) -> str:
    """Convert standard Whisper model names to MLX-compatible names."""
    # MLX-Whisper model mapping
    mlx_model_mapping = {
        'tiny': 'mlx-community/whisper-tiny-mlx',
        'base': 'mlx-community/whisper-base-mlx',
        'small': 'mlx-community/whisper-small-mlx',
        'medium': 'mlx-community/whisper-medium-mlx',
        'large': 'mlx-community/whisper-large-mlx',
        'large-v2': 'mlx-community/whisper-large-v2-mlx',
        'large-v3': 'mlx-community/whisper-large-v3-mlx',
        'large-v3-turbo': 'mlx-community/whisper-large-v3-turbo'
    }
    
    return mlx_model_mapping.get(whisper_model, whisper_model)

def detect_language_first_pass(model_name: str, audio_file: str, use_mlx: bool = True) -> Optional[str]:
    """Detect language using a quick first pass with a smaller model."""
    try:
        if use_mlx and check_mlx_availability():
            import mlx_whisper
            # Use a smaller model for quick language detection
            small_model = get_mlx_model_name('small')
            logger.info(f"Detecting language using MLX model: {small_model}")
            # Transcribe just the first 30 seconds for language detection
            result = mlx_whisper.transcribe(
                audio_file, 
                path_or_hf_repo=small_model,
                clip_timestamps=[(0, 30)]  # Only process first 30 seconds
            )
        else:
            import whisper
            logger.info("Detecting language using standard Whisper")
            model = whisper.load_model('base')  # Use base model for quick detection
            # Load audio and detect language
            audio = whisper.load_audio(audio_file)
            audio = whisper.pad_or_trim(audio)  # Trim to 30 seconds
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            logger.info(f"Detected language: {detected_language} (confidence: {probs[detected_language]:.2f})")
            return detected_language
            
        if result and 'language' in result:
            detected_language = result['language']
            logger.info(f"Detected language: {detected_language}")
            return detected_language
        
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
    
    return None

def transcribe_audio_mlx(model_name: str, audio_file: str, language: str = None, task: str = 'transcribe') -> Optional[dict]:
    """Transcribe audio file using MLX-Whisper model."""
    try:
        import mlx_whisper
        logger.info(f"Using MLX-Whisper for transcription with model: {model_name}")
        
        # Convert model name to MLX format
        mlx_model_name = get_mlx_model_name(model_name)
        logger.info(f"Using MLX model: {mlx_model_name}")
        
        # Set up transcription parameters
        transcribe_params = {
            'audio': audio_file,
            'path_or_hf_repo': mlx_model_name,
            'task': task  # 'transcribe' keeps original language, 'translate' converts to English
        }
        
        if language:
            transcribe_params['language'] = language
            logger.info(f"Using specified language: {language}")
        
        result = mlx_whisper.transcribe(**transcribe_params)
        
        # Log the detected/used language
        if result and 'language' in result:
            logger.info(f"Final transcription language: {result['language']}")
        
        return result
        
    except Exception as e:
        logger.error(f"MLX transcription failed: {e}")
        return None

def transcribe_audio_standard(model, audio_file: str, language: str = None, task: str = 'transcribe', use_fp16: bool = False) -> Optional[dict]:
    """Transcribe audio file using standard Whisper model."""
    try:
        logger.info(f"Using standard Whisper for transcription")
        
        # Set up transcription parameters
        transcribe_params = {
            'audio': audio_file,
            'fp16': use_fp16,
            'task': task  # 'transcribe' keeps original language, 'translate' converts to English
        }
        
        if language:
            transcribe_params['language'] = language
            logger.info(f"Using specified language: {language}")
        
        result = model.transcribe(**transcribe_params)
        
        # Log the detected/used language
        if result and 'language' in result:
            logger.info(f"Final transcription language: {result['language']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Standard transcription failed: {e}")
        return None

def get_optimal_device(force_cpu: bool = False) -> str:
    """Determine the optimal device for computation."""
    if force_cpu:
        return "cpu"
    
    # Check for Apple Metal Performance Shaders (MPS)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    
    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"
    
    return "cpu"

def save_transcription(result: dict, audio_file: str, model_name: str, start_time: float, output_dir: str, backend: str, language: str = None) -> None:
    """Save transcription to a file."""
    try:
        model_name_for_file = model_name.replace('.', '_').replace('/', '_')
        filename = Path(audio_file).stem
        start_dt = datetime.fromtimestamp(start_time)
        
        # Include language in filename if available
        lang_suffix = f"_{language}" if language else ""
        output_filename = f"{filename}_{model_name_for_file}_{backend}{lang_suffix}_{start_dt.strftime('%Y%m%d%H%M')}.txt"
        output_path = Path(output_dir) / output_filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result['text'])
        
        logger.info(f"Transcription written to: {output_path}")
        
        # Enhanced summary with language information
        detected_language = result.get('language', 'unknown')
        print(f"\nSummary:")
        print(f"  Audio file: {audio_file}")
        print(f"  Model: {model_name}")
        print(f"  Backend: {backend}")
        print(f"  Language: {detected_language}")
        print(f"  Output: {output_path}\n")
        
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
        use_mlx: bool = config['DEFAULT'].getboolean('use_mlx', fallback=True)
        force_cpu: bool = config['DEFAULT'].getboolean('force_cpu', fallback=False)
        hf_token: str = config['DEFAULT'].get('hf_token', '')
        specified_language: str = config['DEFAULT'].get('language', '').strip()
        task: str = config['DEFAULT'].get('task', 'transcribe').strip()

        # Validate task parameter
        if task not in ['transcribe', 'translate']:
            logger.warning(f"Invalid task '{task}'. Using 'transcribe' to maintain original language.")
            task = 'transcribe'

        logger.info(f"Task: {task} ({'maintains original language' if task == 'transcribe' else 'translates to English'})")

        # Setup Hugging Face authentication
        setup_hf_authentication(hf_token)

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

        # Language detection and setup
        language_to_use = None
        if specified_language:
            language_to_use = specified_language
            logger.info(f"Using user-specified language: {language_to_use}")
        else:
            logger.info("No language specified, will auto-detect during transcription")

        # Determine transcription method and setup
        use_mlx_backend = False
        model = None
        backend_name = "standard"
        
        # Check if we should use MLX-Whisper
        if use_mlx and is_apple_silicon() and check_mlx_availability():
            use_mlx_backend = True
            backend_name = "mlx"
            logger.info(f"Using MLX-Whisper backend with model: {whisper_model}")
        else:
            # Fall back to standard Whisper
            if use_mlx and is_apple_silicon() and not check_mlx_availability():
                logger.warning("MLX-Whisper not available. Install with: pip install mlx-whisper")
                logger.info("Falling back to standard Whisper")
            elif use_mlx and not is_apple_silicon():
                logger.info("MLX-Whisper only available on Apple Silicon. Using standard Whisper")
            
            # Setup standard Whisper
            device = get_optimal_device(force_cpu)
            logger.info(f"Using device: {device}")
            
            try:
                import whisper
                model = whisper.load_model(whisper_model)
                if device != "cpu":
                    model = model.to(device)
                backend_name = f"standard-{device}"
                logger.info(f"Loaded Whisper model: {whisper_model} on {device}")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                sys.exit(1)

        # Start transcription
        start_time = time.time()
        start_dt = datetime.fromtimestamp(start_time)
        logger.info(f"Start Time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

        # Transcribe using appropriate backend
        result = None
        if use_mlx_backend:
            result = transcribe_audio_mlx(whisper_model, audio_file, language=language_to_use, task=task)
        else:
            # Determine if we should use FP16
            device = get_optimal_device(force_cpu)
            use_fp16 = device in ["cuda", "mps"] and not force_cpu
            result = transcribe_audio_standard(model, audio_file, language=language_to_use, task=task, use_fp16=use_fp16)

        if result is None:
            logger.error("Transcription failed")
            sys.exit(1)

        # Log duration
        end_time = time.time()
        duration = end_time - start_time
        d_hours, d_minutes = format_duration(duration)
        logger.info(f"Transcription Duration: {d_hours} hours, {d_minutes} minutes")

        # Calculate speed ratio
        speed_ratio = audio_length / duration if duration > 0 else 0
        logger.info(f"Speed ratio: {speed_ratio:.2f}x real-time")

        # Save transcription
        final_language = result.get('language', language_to_use)
        save_transcription(result, audio_file, whisper_model, start_time, output_dir, backend_name, final_language)

    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()