
import os
import time
from datetime import datetime
import logging
import configparser
import soundfile as sf
import torch
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import re
import platform

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

def get_downloads_path() -> str:
    """Get the Downloads folder path based on the current operating system."""
    system = platform.system().lower()
    home = Path.home()
    
    if system == 'darwin':  # macOS
        return str(home / 'Downloads')
    elif system == 'windows':
        return str(home / 'Downloads')
    elif system == 'linux':
        # Try XDG user directories first, fallback to ~/Downloads
        xdg_download = os.environ.get('XDG_DOWNLOAD_DIR')
        if xdg_download and os.path.exists(xdg_download):
            return xdg_download
        return str(home / 'Downloads')
    else:
        # Fallback for unknown systems
        return str(home / 'Downloads')

def resolve_output_path(output_dir: str) -> str:
    """Resolve output directory path, handling special cases like ~/Downloads."""
    if output_dir.strip().lower() in ['~/downloads', '~/Downloads']:
        return get_downloads_path()
    elif output_dir.startswith('~'):
        return str(Path(output_dir).expanduser())
    else:
        return output_dir

def load_config(config_path: str = 'config.ini') -> configparser.ConfigParser:
    """Load configuration from an INI file or create default if not exists."""
    config = configparser.ConfigParser()
    default_config = {
        'DEFAULT': {
            'audio_file': '',
            'whisper_model': 'large-v3-turbo',
            'output_dir': '~/Downloads',
            'use_mlx': 'true',
            'force_cpu': 'false',
            'hf_token_file': '.hf_token',
            'language': 'en',  # Set to English for English transcription
            'task': 'transcribe',  # Use translate to convert Chinese to English
            # Chunking options
            'enable_chunking': 'true',
            'chunk_length': '120',
            'overlap_length': '20',
            'max_segment_length': '1000',
            'verbose_output': 'false'
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

def chunk_audio_timestamps(audio_length: float, chunk_length: float = 30.0, overlap: float = 5.0) -> List[Tuple[float, float]]:
    """Generate timestamp chunks for processing long audio files."""
    chunks = []
    start = 0.0
    
    while start < audio_length:
        end = min(start + chunk_length, audio_length)
        chunks.append((float(start), float(end)))  # Ensure float values
        
        # Move start position (with overlap)
        start = end - overlap
        
        # Break if we've covered the entire audio
        if end >= audio_length:
            break
    
    return chunks

def merge_overlapping_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge overlapping segments based on timestamps."""
    if not segments:
        return []
    
    # Sort by start time
    segments.sort(key=lambda x: x.get('start', 0))
    
    merged = [segments[0]]
    for current in segments[1:]:
        last = merged[-1]
        if current['start'] <= last['end']:
            # Merge if overlap
            last['end'] = max(last['end'], current['end'])
            last['text'] = (last.get('text', '') + ' ' + current.get('text', '')).strip()
        else:
            merged.append(current)
    return merged

def remove_repetitions(text: str) -> str:
    """Remove excessive repetitions in text."""
    # Remove long sequences of the same word (e.g., 'fifth fifth ...')
    text = re.sub(r'\b(\w+)( \1\b){10,}', r'\1', text)
    
    # Remove shorter consecutive repeats, allowing up to 1 duplicate
    words = text.split()
    deduped = []
    prev = ''
    count = 0
    for word in words:
        if word == prev:
            count += 1
            if count < 2:  # Allow 1 repeat
                deduped.append(word)
        else:
            deduped.append(word)
            prev = word
            count = 0
    return ' '.join(deduped)

def merge_chunk_results(chunk_results: List[Optional[Dict[str, Any]]], chunks: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Merge results from multiple audio chunks into a single result."""
    if not chunk_results:
        return {"text": "", "segments": [], "language": "en"}
    
    # Filter out None results
    valid_results = [r for r in chunk_results if r is not None]
    if not valid_results:
        return {"text": "", "segments": [], "language": "en"}
    
    merged_result = {
        "text": "",
        "segments": [],
        "language": valid_results[0].get("language", "en")
    }
    
    all_segments = []
    
    for i, chunk_result in enumerate(valid_results):
        # Use the actual chunk start time as offset
        chunk_offset = chunks[i][0]
        
        # Process segments
        segments = chunk_result.get("segments", [])
        for segment in segments:
            adjusted_segment = segment.copy()
            if "start" in adjusted_segment:
                adjusted_segment["start"] += chunk_offset
            if "end" in adjusted_segment:
                adjusted_segment["end"] += chunk_offset
            all_segments.append(adjusted_segment)
    
    # Merge overlapping segments
    all_segments = merge_overlapping_segments(all_segments)
    
    # Reconstruct text from merged segments
    text_parts = [seg.get("text", "").strip() for seg in all_segments if seg.get("text")]
    
    # Basic deduplication on parts
    deduped_text = []
    prev_part = ""
    for part in text_parts:
        if part != prev_part:
            deduped_text.append(part)
            prev_part = part
    
    merged_text = " ".join(deduped_text)
    
    # Post-process to remove repetitions
    merged_text = remove_repetitions(merged_text)
    
    merged_result["text"] = merged_text
    merged_result["segments"] = all_segments
    
    return merged_result

def format_long_text(text: str, max_line_length: int = 1000) -> str:
    """Format long text into manageable lines."""
    if len(text) <= max_line_length:
        return text
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1
        
        if current_length + word_length > max_line_length and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += word_length
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)

def get_hf_token(config: configparser.ConfigParser) -> Optional[str]:
    """Get Hugging Face token from multiple sources."""
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        logger.info("Using HF token from HF_TOKEN environment variable")
        return hf_token.strip()
    
    hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if hf_token:
        logger.info("Using HF token from HUGGING_FACE_HUB_TOKEN environment variable")
        return hf_token.strip()
    
    token_file = config['DEFAULT'].get('hf_token_file', '.hf_token')
    if token_file and os.path.isfile(token_file):
        try:
            with open(token_file, 'r', encoding='utf-8') as f:
                token = f.read().strip()
                if token and not token.startswith('#'):
                    logger.info(f"Using HF token from file: {token_file}")
                    return token
        except Exception as e:
            logger.warning(f"Error reading token file {token_file}: {e}")
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        if user_info:
            logger.info(f"Using existing HF CLI authentication for user: {user_info['name']}")
            return "CLI_AUTH"
    except Exception:
        pass
    
    logger.info("No Hugging Face authentication found")
    return None

def setup_hf_authentication(config: configparser.ConfigParser) -> bool:
    """Setup Hugging Face authentication."""
    hf_token = get_hf_token(config)
    
    if not hf_token:
        logger.info("No HF token found - using public models only")
        return False
    
    if hf_token == "CLI_AUTH":
        return True
    
    try:
        from huggingface_hub import login
        login(token=hf_token)
        logger.info("Successfully authenticated with Hugging Face Hub")
        return True
    except Exception as e:
        logger.warning(f"Failed to authenticate with Hugging Face Hub: {e}")
        return False

def create_token_file_if_needed(config_path: str = 'config.ini') -> None:
    """Create a template token file if it doesn't exist."""
    config = configparser.ConfigParser()
    config.read(config_path)
    
    token_file = config['DEFAULT'].get('hf_token_file', '.hf_token')
    
    if not os.path.isfile(token_file):
        try:
            with open(token_file, 'w', encoding='utf-8') as f:
                f.write("# Place your Hugging Face token here\n")
                f.write("# Get your token from: https://huggingface.co/settings/tokens\n")
                f.write("# Remove these comment lines and paste your token below\n")
                f.write("# hf_...\n")
            
            if hasattr(os, 'chmod'):
                os.chmod(token_file, 0o600)
            
            logger.info(f"Created token file template: {token_file}")
            
        except Exception as e:
            logger.warning(f"Could not create token file {token_file}: {e}")

def validate_audio_file(file_path: str) -> bool:
    """Validate audio file."""
    supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    if not os.path.isfile(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return False
    if not any(file_path.lower().endswith(ext) for ext in supported_extensions):
        logger.error(f"Unsupported audio file format: {file_path}")
        return False
    return True

def get_audio_duration(file_path: str) -> Optional[float]:
    """Get audio duration in seconds."""
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

def get_mlx_model_name(whisper_model: str) -> str:
    """Convert standard Whisper model names to MLX-compatible names."""
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

def load_audio_chunk(audio_file: str, start_time: float, end_time: float) -> Optional[str]:
    """Load a specific chunk of audio and save to temporary file."""
    try:
        import tempfile
        import subprocess
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_filename = f"chunk_{start_time}_{end_time}_{int(time.time())}.wav"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Use ffmpeg to extract the chunk
        cmd = [
            'ffmpeg', '-i', audio_file,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c', 'copy',
            '-y', temp_path
        ]
        
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return temp_path
        else:
            logger.warning(f"FFmpeg failed for chunk {start_time}-{end_time}: {result.stderr}")
            return None
            
    except Exception as e:
        logger.warning(f"Error creating audio chunk: {e}")
        return None

def transcribe_audio_mlx_chunked(model_name: str, audio_file: str, chunks: List[Tuple[float, float]], language: Optional[str] = None, task: str = 'transcribe', verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Transcribe audio file using MLX-Whisper with chunking."""
    try:
        import mlx_whisper
        mlx_model_name = get_mlx_model_name(model_name)
        
        chunk_results = []
        total_chunks = len(chunks)
        temp_files = []  # Keep track of temp files to clean up
        
        logger.info(f"Processing {total_chunks} chunks with MLX-Whisper")
        logger.info(f"Task: {task}, Target language: {language or 'auto-detect'}")
        
        for i, (start_time, end_time) in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{total_chunks}: {start_time:.1f}s - {end_time:.1f}s")
            
            # Create audio chunk file
            chunk_file = load_audio_chunk(audio_file, start_time, end_time)
            if chunk_file is None:
                logger.warning(f"Skipping chunk {i+1} due to audio processing error")
                chunk_results.append(None)
                continue
            
            temp_files.append(chunk_file)
            
            try:
                # Set up transcription parameters for MLX
                transcribe_params: Dict[str, Any] = {
                    'audio': chunk_file,
                    'path_or_hf_repo': mlx_model_name,
                    'task': task,  # 'translate' will convert to English
                    'verbose': verbose,
                    'temperature': 0.0,
                    'condition_on_previous_text': False,
                    'initial_prompt': "Transcribe the orthopedic lecture on 3D printing, ignoring echoes, technical issues, and repetitions."
                }
                
                # For MLX-Whisper, don't set language when task is 'translate'
                # Use 'zh-TW' for Chinese transcription if detected or specified
                effective_language = language
                if task == 'transcribe':
                    if language == 'zh' or (language is None and mlx_whisper.transcribe(chunk_file, path_or_hf_repo=mlx_model_name, task='transcribe').get('language') in ['zh', 'zh-CN', 'zh-TW']):
                        effective_language = 'zh-TW'
                    if effective_language:
                        transcribe_params['language'] = effective_language
                
                chunk_result = mlx_whisper.transcribe(**transcribe_params)
                chunk_results.append(chunk_result)
                
                # Log progress
                if chunk_result and 'text' in chunk_result:
                    text_preview = str(chunk_result['text'])[:100] + "..." if len(str(chunk_result['text'])) > 100 else str(chunk_result['text'])
                    logger.info(f"Chunk {i+1} completed. Preview: {text_preview}")
                
            except Exception as e:
                logger.warning(f"Failed to process chunk {i+1}: {e}")
                chunk_results.append(None)
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {e}")
        
        return merge_chunk_results(chunk_results, chunks)
        
    except Exception as e:
        logger.error(f"MLX chunked transcription failed: {e}")
        return None

def transcribe_audio_mlx(model_name: str, audio_file: str, language: Optional[str] = None, task: str = 'transcribe', verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Transcribe audio file using MLX-Whisper model."""
    try:
        import mlx_whisper
        logger.info(f"Using MLX-Whisper for transcription with model: {model_name}")
        
        mlx_model_name = get_mlx_model_name(model_name)
        logger.info(f"Using MLX model: {mlx_model_name}")
        logger.info(f"Task: {task}, Target language: {language or 'auto-detect'}")
        
        transcribe_params: Dict[str, Any] = {
            'audio': audio_file,
            'path_or_hf_repo': mlx_model_name,
            'task': task,
            'verbose': verbose,
            'temperature': 0.0,
            'condition_on_previous_text': False,
            'initial_prompt': "Transcribe the orthopedic lecture on 3D printing, ignoring echoes, technical issues, and repetitions."
        }
        
        # For MLX-Whisper, don't set language when task is 'translate'
        # Use 'zh-TW' for Chinese transcription if detected or specified
        effective_language = language
        if task == 'transcribe':
            if language == 'zh' or (language is None and mlx_whisper.transcribe(audio_file, path_or_hf_repo=mlx_model_name, task='transcribe').get('language') in ['zh', 'zh-CN', 'zh-TW']):
                effective_language = 'zh-TW'
            if effective_language:
                transcribe_params['language'] = effective_language
                logger.info(f"Using specified language: {effective_language}")
        elif task == 'translate':
            logger.info("Translating to English (language parameter not needed)")
        
        result = mlx_whisper.transcribe(**transcribe_params)
        
        if result and 'language' in result:
            logger.info(f"Detected/Final language: {result['language']}")
        
        return result
        
    except Exception as e:
        logger.error(f"MLX transcription failed: {e}")
        return None

def transcribe_audio_standard(model: Any, audio_file: str, language: Optional[str] = None, task: str = 'transcribe', use_fp16: bool = False, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Transcribe audio file using standard Whisper model."""
    try:
        logger.info(f"Using standard Whisper for transcription")
        logger.info(f"Task: {task}, Target language: {language or 'auto-detect'}")
        
        transcribe_params: Dict[str, Any] = {
            'audio': audio_file,
            'fp16': use_fp16,
            'task': task,
            'verbose': verbose,
            'temperature': 0.0,
            'beam_size': 5,
            'condition_on_previous_text': False,
            'initial_prompt': "Transcribe the orthopedic lecture on 3D printing, ignoring echoes, technical issues, and repetitions."
        }
        
        # For standard Whisper, don't set language when task is 'translate'
        # Use 'zh-TW' for Chinese transcription if detected or specified
        effective_language = language
        if task == 'transcribe':
            if language == 'zh' or (language is None and model.transcribe(audio_file, task='transcribe').get('language') in ['zh', 'zh-CN', 'zh-TW']):
                effective_language = 'zh-TW'
            if effective_language:
                transcribe_params['language'] = effective_language
                logger.info(f"Using specified language: {effective_language}")
        elif task == 'translate':
            logger.info("Translating to English (language parameter not needed)")
        
        result = model.transcribe(**transcribe_params)
        
        if result and 'language' in result:
            logger.info(f"Detected/Final language: {result['language']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Standard transcription failed: {e}")
        return None

def get_optimal_device(force_cpu: bool = False) -> str:
    """Determine the optimal device for computation."""
    if force_cpu:
        return "cpu"
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    
    if torch.cuda.is_available():
        return "cuda"
    
    return "cpu"

def save_transcription(result: Dict[str, Any], audio_file: str, model_name: str, start_time: float, output_dir: str, backend: str, task: str, language: Optional[str] = None, max_line_length: int = 1000) -> None:
    """Save transcription to a file with proper formatting."""
    try:
        model_name_for_file = model_name.replace('.', '_').replace('/', '_')
        filename = Path(audio_file).stem
        start_dt = datetime.fromtimestamp(start_time)
        
        # Use 'en' for translated content, 'zh-TW' for Chinese transcription
        final_language = 'en' if task == 'translate' else result.get('language', language or 'en')
        if final_language in ['zh', 'zh-CN', 'zh-TW']:
            final_language = 'zh-TW'
        lang_suffix = f"_{final_language}"
        output_filename = f"{filename}_{model_name_for_file}_{backend}{lang_suffix}_{start_dt.strftime('%Y%m%d%H%M')}.txt"
        output_path = Path(output_dir) / output_filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Format the text to avoid long lines
        formatted_text = format_long_text(result['text'], max_line_length)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        
        logger.info(f"Transcription written to: {output_path}")
        
        print(f"\nSummary:")
        print(f"  Audio file: {audio_file}")
        print(f"  Model: {model_name}")
        print(f"  Backend: {backend}")
        print(f"  Final language: {final_language}")
        print(f"  Output: {output_path}")
        print(f"  Text length: {len(result['text'])} characters\n")
        
    except Exception as e:
        logger.error(f"Error saving transcription: {e}")

def main() -> None:
    """Main function to orchestrate audio transcription."""
    import sys
    try:
        config = load_config()
        whisper_model: str = config['DEFAULT']['whisper_model']
        # Always save outputs to the current user's Downloads folder
        output_dir: str = str(Path.home() / "Downloads")
        logger.info(f"Output directory set to: {output_dir}")
        use_mlx: bool = config['DEFAULT'].getboolean('use_mlx', fallback=True)
        force_cpu: bool = config['DEFAULT'].getboolean('force_cpu', fallback=False)
        specified_language: str = config['DEFAULT'].get('language', '').strip()
        task: str = config['DEFAULT'].get('task', 'transcribe').strip()
        
        # Chunking options
        enable_chunking: bool = config['DEFAULT'].getboolean('enable_chunking', fallback=True)
        chunk_length: float = config['DEFAULT'].getfloat('chunk_length', fallback=120.0)
        overlap_length: float = config['DEFAULT'].getfloat('overlap_length', fallback=20.0)
        max_segment_length: int = config['DEFAULT'].getint('max_segment_length', fallback=1000)
        verbose_output: bool = config['DEFAULT'].getboolean('verbose_output', fallback=False)

        create_token_file_if_needed()
        auth_success = setup_hf_authentication(config)
        if not auth_success:
            logger.info("Proceeding with public models only")

        if task not in ['transcribe', 'translate']:
            logger.warning(f"Invalid task '{task}'. Using 'translate' for English output.")
            task = 'translate'

        # Log task explanation
        if task == 'translate':
            logger.info("Task: translate (converts any language to English)")
        else:
            logger.info(f"Task: transcribe (maintains original language or specified language: {specified_language or 'auto-detect'})")

        # Audio file handling
        source_dir = 'source'
        audio_file: Optional[str] = None

        if os.path.isdir(source_dir):
            # Only select mp3 files
            files = [f for f in os.listdir(source_dir)
                     if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith('.mp3')]
            if len(files) == 1:
                audio_file = os.path.join(source_dir, files[0])
                logger.info(f"Using MP3 audio file from 'source': {audio_file}")
            elif len(files) == 0:
                logger.error(f"No MP3 files found in '{source_dir}' directory.")
                sys.exit(1)
            else:
                logger.warning(f"Multiple MP3 files found in '{source_dir}': {files}. Using the first: {files[0]}")
                audio_file = os.path.join(source_dir, files[0])
        else:
            logger.error(f"Directory '{source_dir}' does not exist.")
            sys.exit(1)

        if not validate_audio_file(audio_file):
            sys.exit(1)

        audio_length = get_audio_duration(audio_file)
        if audio_length is None:
            logger.error("Aborting due to audio file error")
            sys.exit(1)
        
        src_hours, src_minutes = format_duration(audio_length)
        logger.info(f"Audio Duration: {src_hours} hours, {src_minutes} minutes")

        # Determine if chunking should be used
        use_chunking = enable_chunking and audio_length > chunk_length * 2
        chunks: Optional[List[Tuple[float, float]]] = None
        if use_chunking:
            chunks = chunk_audio_timestamps(audio_length, chunk_length, overlap_length)
            logger.info(f"Using chunking: {len(chunks)} chunks of {chunk_length}s with {overlap_length}s overlap")
        else:
            logger.info("Processing entire audio file at once")

        # Backend selection
        use_mlx_backend = False
        model = None
        backend_name = "standard"
        
        if use_mlx and is_apple_silicon() and check_mlx_availability():
            use_mlx_backend = True
            backend_name = "mlx"
            logger.info(f"Using MLX-Whisper backend with model: {whisper_model}")
        else:
            if use_mlx and is_apple_silicon() and not check_mlx_availability():
                logger.warning("MLX-Whisper not available. Install with: pip install mlx-whisper")
                logger.info("Falling back to standard Whisper")
            elif use_mlx and not is_apple_silicon():
                logger.info("MLX-Whisper only available on Apple Silicon. Using standard Whisper")
            
            device = get_optimal_device(force_cpu)
            logger.info(f"Using device: {device}")
            
            try:
                import whisper  # type: ignore[import-not-found]
                model = whisper.load_model(whisper_model)  # type: ignore[attr-defined]
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
        logger.info(f"Start Time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} Emmy")

        result = None
        if use_mlx_backend:
            if use_chunking and chunks is not None:
                result = transcribe_audio_mlx_chunked(
                    whisper_model, audio_file, chunks, 
                    language=specified_language if task == 'transcribe' else None, 
                    task=task, verbose=verbose_output
                )
            else:
                result = transcribe_audio_mlx(
                    whisper_model, audio_file, 
                    language=specified_language if task == 'transcribe' else None, 
                    task=task, verbose=verbose_output
                )
        else:
            device = get_optimal_device(force_cpu)
            use_fp16 = device in ["cuda", "mps"] and not force_cpu
            result = transcribe_audio_standard(
                model, audio_file, 
                language=specified_language if task == 'transcribe' else None, 
                task=task, use_fp16=use_fp16, verbose=verbose_output
            )

        if result is None:
            logger.error("Transcription failed")
            sys.exit(1)

        # Log duration
        end_time = time.time()
        duration = end_time - start_time
        d_hours, d_minutes = format_duration(duration)
        logger.info(f"Transcription Duration: {d_hours} hours, {d_minutes} minutes")

        speed_ratio = audio_length / duration if duration > 0 else 0
        logger.info(f"Speed ratio: {speed_ratio:.2f}x real-time")

        # Save transcription
        final_language = 'en' if task == 'translate' else result.get('language', specified_language or 'en')
        if final_language in ['zh', 'zh-CN', 'zh-TW']:
            final_language = 'zh-TW'
        save_transcription(result, audio_file, whisper_model, start_time, output_dir, backend_name, task, final_language, max_segment_length)

    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
