import os
import time
from datetime import datetime
import logging
import configparser
import soundfile as sf
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import re
import platform
import sys

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
        xdg_download = os.environ.get('XDG_DOWNLOAD_DIR')
        if xdg_download and os.path.exists(xdg_download):
            return xdg_download
        return str(home / 'Downloads')
    else:
        return str(home / 'Downloads')

def resolve_output_path(output_dir: str) -> str:
    """Resolve output directory path, handling special cases like ~/Downloads."""
    if not output_dir:
        return get_downloads_path()
        
    if output_dir.strip().lower() in ['~/downloads', '~/Downloads']:
        return get_downloads_path()
    elif output_dir.startswith('~'):
        return str(Path(output_dir).expanduser())
    else:
        return str(Path(output_dir).absolute())

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
            'language': 'en',
            'task': 'transcribe',
            'initial_prompt': 'Use proper punctuation, capitalization, and sentence structure.',
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
        chunks.append((float(start), float(end)))
        start = end - overlap
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
        # Allow a small epsilon for floating point comparisons
        if current['start'] < last['end'] - 0.1:
            # If current segment is completely contained in last, skip it
            if current['end'] <= last['end']:
                continue
                
            # Merge logic: Append text only if it adds new content
            # This is a simplified merge strategy
            last['end'] = max(last['end'], current['end'])
            
            # Simple text concatenation isn't always ideal for overlaps, 
            # but usually sufficient for simple chunking
            if current.get('text', '').strip() not in last.get('text', ''):
                 last['text'] = (last.get('text', '') + ' ' + current.get('text', '')).strip()
        else:
            merged.append(current)
    return merged

def remove_repetitions(text: str) -> str:
    """Remove excessive repetitions in text."""
    if not text:
        return ""
        
    # 1. Remove immediate word repetitions (e.g., "the the the")
    text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1', text, flags=re.IGNORECASE)
    
    # 2. Remove phrase repetitions (e.g., "and then he went and then he went")
    # Limit to phrases of 2-4 words to save CPU time on massive texts
    for n in range(4, 1, -1):
        pattern = r'(\b(?:\w+\s+){' + str(n-1) + r'}\w+\b)(\s+\1)+'
        text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)

    return text

def post_process_text(text: str) -> str:
    """Apply grammar and punctuation rules to clean up the transcription."""
    if not text:
        return ""

    text = remove_repetitions(text)

    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])(?=[a-zA-Z])', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Capitalization fixes
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    
    def capitalize_match(match):
        return match.group(1) + match.group(2).upper()
    
    text = re.sub(r'([.!?]\s+)([a-z])', capitalize_match, text)

    return text

def ensure_text_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        try:
            return " ".join(str(v) for v in value if v is not None).strip()
        except Exception:
            return str(value)
    return str(value) if value is not None else ""

def merge_chunk_results(chunk_results: List[Optional[Dict[str, Any]]], chunks: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Merge results from multiple audio chunks into a single result."""
    if not chunk_results:
        return {"text": "", "segments": [], "language": "en"}
    
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
        chunk_offset = chunks[i][0]
        segments = chunk_result.get("segments", [])
        for segment in segments:
            adjusted_segment = segment.copy()
            if "start" in adjusted_segment:
                adjusted_segment["start"] += chunk_offset
            if "end" in adjusted_segment:
                adjusted_segment["end"] += chunk_offset
            all_segments.append(adjusted_segment)
    
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
    merged_text = post_process_text(merged_text)
    
    merged_result["text"] = merged_text
    merged_result["segments"] = all_segments
    
    return merged_result

def format_long_text(text: str, max_line_length: int = 1000) -> str:
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
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    if hf_token:
        logger.info("Using HF token from environment variable")
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
    
    return None

def setup_hf_authentication(config: configparser.ConfigParser) -> bool:
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
    config = configparser.ConfigParser()
    config.read(config_path)
    token_file = config['DEFAULT'].get('hf_token_file', '.hf_token')
    
    if not os.path.isfile(token_file):
        try:
            with open(token_file, 'w', encoding='utf-8') as f:
                f.write("# Place your Hugging Face token here\n")
            if hasattr(os, 'chmod'):
                os.chmod(token_file, 0o600)
        except Exception:
            pass

def validate_audio_file(file_path: str) -> bool:
    supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    if not os.path.isfile(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return False
    if not any(file_path.lower().endswith(ext) for ext in supported_extensions):
        logger.error(f"Unsupported audio file format: {file_path}")
        return False
    return True

def get_audio_info(file_path: str) -> Tuple[Optional[float], Optional[int]]:
    """Get audio duration and sample rate."""
    try:
        info = sf.info(file_path)
        return info.frames / float(info.samplerate), info.samplerate
    except Exception as e:
        logger.error(f"Error reading audio info: {e}")
        return None, None

def format_duration(seconds: float) -> Tuple[int, int]:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return hours, minutes

def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def check_mlx_availability() -> bool:
    try:
        import mlx_whisper
        return True
    except ImportError:
        return False

def get_mlx_model_name(whisper_model: str) -> str:
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

def transcribe_audio_mlx_chunked(
    model_name: str, 
    audio_file: str, 
    chunks: List[Tuple[float, float]], 
    language: Optional[str] = None, 
    task: str = 'transcribe', 
    verbose: bool = False,
    initial_prompt: str = ""
) -> Optional[Dict[str, Any]]:
    """Transcribe audio file using MLX-Whisper with in-memory chunking (No FFmpeg)."""
    try:
        import mlx_whisper
        mlx_model_name = get_mlx_model_name(model_name)
        
        chunk_results = []
        total_chunks = len(chunks)
        
        logger.info(f"Processing {total_chunks} chunks with MLX-Whisper (In-Memory)")
        
        # Read the full audio file into memory once
        # MLX Whisper expects float32
        logger.info("Loading audio file into memory...")
        audio_data, sample_rate = sf.read(audio_file, dtype='float32')
        
        # If stereo, convert to mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        for i, (start_time, end_time) in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{total_chunks}: {start_time:.1f}s - {end_time:.1f}s")
            
            # Calculate sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Slice the numpy array
            audio_chunk = audio_data[start_sample:end_sample]
            
            try:
                transcribe_params: Dict[str, Any] = {
                    'audio': audio_chunk, # Pass numpy array directly
                    'path_or_hf_repo': mlx_model_name,
                    'task': task,
                    'verbose': verbose,
                    'temperature': 0.0,
                    'condition_on_previous_text': False,
                    'initial_prompt': initial_prompt
                }
                
                effective_language = language
                if task == 'transcribe':
                    # For chunked, we rely on the main config or auto-detect from first chunk usually,
                    # but here we let MLX handle it per chunk if not specified.
                    if effective_language:
                        transcribe_params['language'] = effective_language
                
                chunk_result = mlx_whisper.transcribe(**transcribe_params)
                chunk_results.append(chunk_result)
                
            except Exception as e:
                logger.warning(f"Failed to process chunk {i+1}: {e}")
                chunk_results.append(None)
        
        return merge_chunk_results(chunk_results, chunks)
        
    except Exception as e:
        logger.error(f"MLX chunked transcription failed: {e}")
        return None

def transcribe_audio_mlx(
    model_name: str, 
    audio_file: str, 
    language: Optional[str] = None, 
    task: str = 'transcribe', 
    verbose: bool = False,
    initial_prompt: str = ""
) -> Optional[Dict[str, Any]]:
    """Transcribe audio file using MLX-Whisper model."""
    try:
        import mlx_whisper
        mlx_model_name = get_mlx_model_name(model_name)
        logger.info(f"Using MLX model: {mlx_model_name}")
        
        transcribe_params: Dict[str, Any] = {
            'audio': audio_file,
            'path_or_hf_repo': mlx_model_name,
            'task': task,
            'verbose': verbose,
            'temperature': 0.0,
            'condition_on_previous_text': False,
            'initial_prompt': initial_prompt
        }
        
        effective_language = language
        if task == 'transcribe':
            if language == 'zh' or (language is None and mlx_whisper.transcribe(audio_file, path_or_hf_repo=mlx_model_name, task='transcribe').get('language') in ['zh', 'zh-CN', 'zh-TW']):
                effective_language = 'zh-TW'
            if effective_language:
                transcribe_params['language'] = effective_language
                logger.info(f"Using specified language: {effective_language}")
        
        result = mlx_whisper.transcribe(**transcribe_params)
        
        if result and 'text' in result:
            normalized_text = ensure_text_str(result['text'])
            result['text'] = post_process_text(normalized_text)
        
        return result
        
    except Exception as e:
        logger.error(f"MLX transcription failed: {e}")
        return None

def transcribe_audio_standard(
    model: Any, 
    audio_file: str, 
    language: Optional[str] = None, 
    task: str = 'transcribe', 
    use_fp16: bool = False, 
    verbose: bool = False,
    initial_prompt: str = ""
) -> Optional[Dict[str, Any]]:
    """Transcribe audio file using Faster-Whisper model."""
    try:
        logger.info(f"Using Faster-Whisper for transcription")
        
        options: Dict[str, Any] = {
            'beam_size': 5,
            'vad_filter': True
        }
        if task in ['transcribe', 'translate']:
            options['task'] = task
        if initial_prompt:
            options['initial_prompt'] = initial_prompt
        if language:
            options['language'] = 'zh-TW' if language in ['zh', 'zh-CN', 'zh-TW'] else language
        
        segments, info = model.transcribe(audio_file, **options)
        
        seg_list: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        for seg in segments:
            seg_list.append({
                "id": getattr(seg, "id", None),
                "seek": getattr(seg, "seek", None),
                "start": float(seg.start) if getattr(seg, "start", None) is not None else None,
                "end": float(seg.end) if getattr(seg, "end", None) is not None else None,
                "text": getattr(seg, "text", "") or "",
                "tokens": getattr(seg, "tokens", None),
                "temperature": getattr(seg, "temperature", None),
                "avg_logprob": getattr(seg, "avg_logprob", None),
                "compression_ratio": getattr(seg, "compression_ratio", None),
                "no_speech_prob": getattr(seg, "no_speech_prob", None),
            })
            if getattr(seg, "text", None):
                text_parts.append(seg.text.strip())
        
        combined_text = post_process_text(" ".join(text_parts))
        detected_language = getattr(info, "language", None) or language or "en"
        if detected_language in ['zh', 'zh-CN', 'zh-TW']:
            detected_language = 'zh-TW'
        
        return {"text": combined_text, "segments": seg_list, "language": detected_language}
        
    except Exception as e:
        logger.error(f"Standard transcription failed: {e}")
        return None

def get_optimal_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def save_transcription(result: Dict[str, Any], audio_file: str, model_name: str, start_time: float, output_dir: str, backend: str, task: str, language: Optional[str] = None, max_line_length: int = 1000) -> None:
    try:
        model_name_for_file = model_name.replace('.', '_').replace('/', '_')
        filename = Path(audio_file).stem
        start_dt = datetime.fromtimestamp(start_time)
        
        final_language = 'en' if task == 'translate' else result.get('language', language or 'en')
        if final_language in ['zh', 'zh-CN', 'zh-TW']:
            final_language = 'zh-TW'
        
        lang_suffix = f"_{final_language}"
        output_filename = f"{filename}_{model_name_for_file}_{backend}{lang_suffix}_{start_dt.strftime('%Y%m%d%H%M')}.txt"
        
        # Ensure output directory exists
        out_path_obj = Path(output_dir)
        out_path_obj.mkdir(parents=True, exist_ok=True)
        
        output_path = out_path_obj / output_filename
        
        formatted_text = format_long_text(result['text'], max_line_length)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        
        logger.info(f"Transcription written to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving transcription: {e}")

def main() -> None:
    try:
        config = load_config()
        
        # Configuration
        whisper_model = config['DEFAULT']['whisper_model']
        # Fix: Use the config value, don't hardcode override
        raw_output_dir = config['DEFAULT'].get('output_dir', '~/Downloads')
        output_dir = resolve_output_path(raw_output_dir)
        
        use_mlx = config['DEFAULT'].getboolean('use_mlx', fallback=True)
        force_cpu = config['DEFAULT'].getboolean('force_cpu', fallback=False)
        specified_language = config['DEFAULT'].get('language', '').strip()
        task = config['DEFAULT'].get('task', 'transcribe').strip()
        initial_prompt = config['DEFAULT'].get('initial_prompt', '').strip()
        
        enable_chunking = config['DEFAULT'].getboolean('enable_chunking', fallback=True)
        chunk_length = config['DEFAULT'].getfloat('chunk_length', fallback=120.0)
        overlap_length = config['DEFAULT'].getfloat('overlap_length', fallback=20.0)
        max_segment_length = config['DEFAULT'].getint('max_segment_length', fallback=1000)
        verbose_output = config['DEFAULT'].getboolean('verbose_output', fallback=False)

        logger.info(f"Output directory set to: {output_dir}")
        create_token_file_if_needed()
        setup_hf_authentication(config)

        if task not in ['transcribe', 'translate']:
            logger.warning(f"Invalid task '{task}'. Using 'translate'.")
            task = 'translate'

        # Audio file handling
        source_dir = 'source'
        audio_file: Optional[str] = None

        if os.path.isdir(source_dir):
            files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(('.mp3', '.wav', '.m4a'))]
            if len(files) == 1:
                audio_file = os.path.join(source_dir, files[0])
                logger.info(f"Found audio file: {audio_file}")
            elif len(files) == 0:
                logger.error(f"No audio files found in '{source_dir}' directory.")
                sys.exit(1)
            else:
                logger.warning(f"Multiple audio files found. Using: {files[0]}")
                audio_file = os.path.join(source_dir, files[0])
        else:
            logger.error(f"Directory '{source_dir}' does not exist.")
            sys.exit(1)

        if not validate_audio_file(audio_file):
            sys.exit(1)

        audio_duration, sample_rate = get_audio_info(audio_file)
        if audio_duration is None:
            sys.exit(1)
        
        src_hours, src_minutes = format_duration(audio_duration)
        logger.info(f"Audio Duration: {src_hours}h {src_minutes}m")

        # Chunking Logic
        use_chunking = enable_chunking and audio_duration > chunk_length * 2
        chunks = None
        if use_chunking:
            chunks = chunk_audio_timestamps(audio_duration, chunk_length, overlap_length)
            logger.info(f"Using chunking: {len(chunks)} chunks")
        
        # Backend Selection
        use_mlx_backend = False
        model = None
        backend_name = "standard"
        
        if use_mlx and is_apple_silicon() and check_mlx_availability():
            use_mlx_backend = True
            backend_name = "mlx"
        else:
            if use_mlx:
                logger.info("MLX not available/applicable. Falling back to standard Whisper.")
            
            device = get_optimal_device(force_cpu)
            try:
                
                from faster_whisper import WhisperModel
                device_fw = "metal" if device == "mps" else device
                compute_type = "float16" if device in ["cuda", "mps"] and not force_cpu else "int8"
                model = WhisperModel(whisper_model, device=device_fw, compute_type=compute_type)
                backend_name = f"standard-{device}"
            except Exception as e:
                logger.error(f"Error loading Faster-Whisper model: {e}")
                sys.exit(1)

        # Start Transcription
        start_time = time.time()
        result = None
        
        if use_mlx_backend:
            if use_chunking and chunks:
                result = transcribe_audio_mlx_chunked(
                    whisper_model, audio_file, chunks, 
                    language=specified_language if task == 'transcribe' else None, 
                    task=task, verbose=verbose_output, initial_prompt=initial_prompt
                )
            else:
                result = transcribe_audio_mlx(
                    whisper_model, audio_file, 
                    language=specified_language if task == 'transcribe' else None, 
                    task=task, verbose=verbose_output, initial_prompt=initial_prompt
                )
        else:
            device = get_optimal_device(force_cpu)
            use_fp16 = device in ["cuda", "mps"] and not force_cpu
            result = transcribe_audio_standard(
                model, audio_file, 
                language=specified_language if task == 'transcribe' else None, 
                task=task, use_fp16=use_fp16, verbose=verbose_output, initial_prompt=initial_prompt
            )

        if result is None:
            logger.error("Transcription failed")
            sys.exit(1)

        # Statistics and Saving
        end_time = time.time()
        duration = end_time - start_time
        speed_ratio = audio_duration / duration if duration > 0 else 0
        logger.info(f"Finished in {duration:.1f}s ({speed_ratio:.2f}x speed)")

        final_language = 'en' if task == 'translate' else result.get('language', specified_language or 'en')
        if final_language in ['zh', 'zh-CN', 'zh-TW']:
            final_language = 'zh-TW'
            
        save_transcription(result, audio_file, whisper_model, start_time, output_dir, backend_name, task, final_language, max_segment_length)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()