import torch
import whisper
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduce memory fragmentation
import sys
import soundfile as sf
import logging
import argparse
import tempfile
import numpy as np
import datetime
from typing import Optional, List, Tuple
from pathlib import Path

# Configure logging with a consistent format for debugging and monitoring
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handle audio processing with multiple fallback methods for robust audio handling."""
    
    def __init__(self):
        self.pydub_available = self._check_pydub()
        self.librosa_available = self._check_librosa()
    
    def _check_pydub(self) -> bool:
        """Check if pydub is available and functional."""
        try:
            from pydub import AudioSegment
            # Test basic pydub functionality with a silent audio segment
            AudioSegment.silent(duration=100)
            logger.info("Pydub is available")
            return True
        except Exception as e:
            logger.warning(f"Pydub not available: {str(e)}")
            return False
    
    def _check_librosa(self) -> bool:
        """Check if librosa is available for audio processing."""
        try:
            import librosa
            logger.info("Librosa is available")
            return True
        except ImportError as e:
            logger.warning(f"Librosa not available: {str(e)}")
            return False
    
    def get_capabilities(self) -> str:
        """Return a string of available audio processing libraries."""
        capabilities = ["soundfile (native)"]
        if self.pydub_available:
            capabilities.append("pydub")
        if self.librosa_available:
            capabilities.append("librosa")
        return ", ".join(capabilities)

def validate_audio_file(file_path: str) -> bool:
    """Validate that the audio file exists and has a supported extension."""
    supported_extensions = {
        '.wav', '.mp3', '.flac', '.ogg', 
        '.m4a', '.aac', '.wma'
    }
    
    file_path_obj = Path(file_path)
    if not file_path_obj.is_file():
        logger.error(f"Audio file not found: {file_path}")
        return False
    
    if not any(file_path.lower().endswith(ext) for ext in supported_extensions):
        logger.error(f"Unsupported format for {file_path}. Supported: {', '.join(supported_extensions)}")
        return False
    
    return True

def get_audio_info(file_path: str) -> Tuple[float, int]:
    """Retrieve duration and sample rate of the audio file."""
    try:
        info = sf.info(file_path)
        duration = float(info.frames) / float(info.samplerate)
        return duration, int(info.samplerate)
    except Exception as e:
        logger.error(f"Error reading audio info for {file_path}: {str(e)}")
        return 0.0, 0

from typing import Union

def process_transcription_result(result_text: Union[str, List[str]]) -> str:
    """Process transcription result, handling both string and list outputs."""
    if isinstance(result_text, str):
        return result_text.strip()
    elif isinstance(result_text, list):
        # Handle case where result['text'] is a list (e.g., segmented transcriptions)
        cleaned_text = " ".join(str(item).strip() for item in result_text if item)
        logger.warning(f"Transcription result is a list, joined to string: {len(result_text)} segments")
        return cleaned_text
    else:
        logger.error(f"Unexpected transcription result type: {type(result_text)}")
        return ""

def split_with_soundfile(file_path: str, model, chunk_length_s: int = 600) -> str:
    """Process audio in chunks using soundfile for reliable handling of large files."""
    try:
        logger.info(f"Processing audio with soundfile: {file_path}")
        
        # Load audio data as mono
        audio_data, sample_rate = sf.read(file_path, dtype='float32')
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        total_duration = len(audio_data) / sample_rate
        logger.info(f"Loaded audio: {total_duration:.1f}s at {sample_rate}Hz")
        
        chunk_samples = int(chunk_length_s * sample_rate)
        num_chunks = int(np.ceil(len(audio_data) / chunk_samples))
        logger.info(f"Processing {num_chunks} chunks of {chunk_length_s}s each")
        
        transcription_parts: List[str] = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, len(audio_data))
            chunk_data = audio_data[start_idx:end_idx]
            chunk_duration = len(chunk_data) / sample_rate
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                try:
                    sf.write(tmp_file.name, chunk_data, sample_rate)
                    logger.info(f"Transcribing chunk {i+1}/{num_chunks} ({chunk_duration:.1f}s)")
                    
                    result = model.transcribe(
                        tmp_file.name, 
                        fp16=torch.cuda.is_available(),
                        verbose=False
                    )
                    
                    chunk_text = process_transcription_result(result['text'])
                    if chunk_text:
                        transcription_parts.append(chunk_text)
                        logger.info(f"Chunk {i+1}: {len(chunk_text)} characters transcribed")
                    else:
                        logger.warning(f"Chunk {i+1}: Empty transcription")
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {str(e)}")
                finally:
                    try:
                        os.unlink(tmp_file.name)
                    except OSError:
                        logger.debug(f"Failed to delete temp file: {tmp_file.name}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        del audio_data
        return " ".join(transcription_parts)
        
    except Exception as e:
        logger.error(f"Soundfile processing failed: {str(e)}")
        return ""

def split_with_pydub(file_path: str, model, chunk_length_s: int = 600) -> str:
    """Process audio in chunks using pydub as a fallback method."""
    try:
        from pydub import AudioSegment
        logger.info(f"Processing audio with pydub: {file_path}")
        
        audio = AudioSegment.from_file(file_path)
        chunk_length_ms = chunk_length_s * 1000
        duration_ms = len(audio)
        
        transcription_parts: List[str] = []
        
        for i in range(0, duration_ms, chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                try:
                    chunk.export(tmp_file.name, format="wav")
                    logger.info(f"Transcribing pydub chunk {i // chunk_length_ms + 1}")
                    
                    result = model.transcribe(
                        tmp_file.name, 
                        fp16=torch.cuda.is_available(),
                        verbose=False
                    )
                    
                    chunk_text = process_transcription_result(result['text'])
                    if chunk_text:
                        transcription_parts.append(chunk_text)
                except Exception as e:
                    logger.error(f"Error processing pydub chunk {i // chunk_length_ms + 1}: {str(e)}")
                finally:
                    try:
                        os.unlink(tmp_file.name)
                    except OSError:
                        logger.debug(f"Failed to delete temp file: {tmp_file.name}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        return " ".join(transcription_parts)
        
    except Exception as e:
        logger.error(f"Pydub processing failed: {str(e)}")
        return ""

def split_with_librosa(file_path: str, model, chunk_length_s: int = 600) -> str:
    """Process audio in chunks using librosa as a fallback method."""
    try:
        import librosa
        logger.info(f"Processing audio with librosa: {file_path}")
        
        audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        chunk_samples = int(chunk_length_s * sample_rate)
        
        transcription_parts: List[str] = []
        
        for i in range(0, len(audio_data), chunk_samples):
            chunk_data = audio_data[i:i + chunk_samples]
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                try:
                    sf.write(tmp_file.name, chunk_data, sample_rate)
                    logger.info(f"Transcribing librosa chunk {i // chunk_samples + 1}")
                    
                    result = model.transcribe(
                        tmp_file.name, 
                        fp16=torch.cuda.is_available(),
                        verbose=False
                    )
                    
                    chunk_text = process_transcription_result(result['text'])
                    if chunk_text:
                        transcription_parts.append(chunk_text)
                except Exception as e:
                    logger.error(f"Error processing librosa chunk {i // chunk_samples + 1}: {str(e)}")
                finally:
                    try:
                        os.unlink(tmp_file.name)
                    except OSError:
                        logger.debug(f"Failed to delete temp file: {tmp_file.name}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        return " ".join(transcription_parts)
        
    except Exception as e:
        logger.error(f"Librosa processing failed: {str(e)}")
        return ""

def find_audio_file(input_dir: str = "source") -> Optional[str]:
    """Find the first valid audio file in the specified directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Directory '{input_dir}' does not exist")
        return None
    
    audio_files = [str(f) for f in input_path.iterdir() if f.is_file() and validate_audio_file(str(f))]
    
    if not audio_files:
        logger.error(f"No valid audio files found in '{input_dir}'")
        return None
    
    if len(audio_files) > 1:
        logger.warning(f"Multiple audio files found, using first: {Path(audio_files[0]).name}")
    
    return audio_files[0]

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the transcription script."""
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper with CUDA support")
    
    parser.add_argument(
        "audio_file", 
        nargs="?", 
        help="Path to the audio file (optional, will search 'source' directory if not provided)"
    )
    
    parser.add_argument(
        "--model", 
        default="large-v3-turbo",  # Changed to base to reduce memory usage
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'large-v3-turbo'],
        help="Whisper model to use (default: base)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="outputs", 
        help="Directory to save transcription output (default: outputs)"
    )
    
    parser.add_argument(
        "--chunk-length", 
        type=int, 
        default=600,  # Reduced from 600 to 300 to lower memory usage
        help="Length of audio chunks in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--threshold", 
        type=int, 
        default=600, 
        help="Duration threshold for enabling chunking in seconds (default: 600)"
    )
    
    parser.add_argument(
        "--force-chunk", 
        action="store_true", 
        help="Force chunking regardless of audio duration"
    )
    
    parser.add_argument(
        "--force-cpu", 
        action="store_true", 
        help="Force processing on CPU even if CUDA is available"
    )
    
    return parser.parse_args()

def generate_output_filename(audio_file: str, model_name: str) -> str:
    """Generate a unique output filename with timestamp."""
    base_name = Path(audio_file).stem
    model_safe = model_name.replace('.', '_')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{model_safe}_{timestamp}_transcription.txt"

def log_gpu_memory() -> None:
    """Log current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GiB
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GiB
        reserved = torch.cuda.memory_reserved(0) / 1024**3  # GiB
        logger.info(
            f"GPU Memory: Total {total_memory:.2f} GiB, "
            f"Allocated {allocated:.2f} GiB, "
            f"Reserved {reserved:.2f} GiB"
        )

def main():
    """Main function to orchestrate audio transcription process."""
    args = parse_args()
    
    # Initialize audio processor and log capabilities
    processor = AudioProcessor()
    logger.info(f"Available audio processing libraries: {processor.get_capabilities()}")
    
    # Log GPU memory before processing
    log_gpu_memory()
    
    # Resolve audio file path
    audio_file = args.audio_file or find_audio_file()
    if not audio_file:
        logger.error("No valid audio file provided or found")
        sys.exit(1)
    
    logger.info(f"Processing audio file: {audio_file}")
    
    # Retrieve audio metadata
    duration, sample_rate = get_audio_info(audio_file)
    if duration == 0:
        logger.error("Failed to read audio file metadata")
        sys.exit(1)
    
    logger.info(f"Audio duration: {duration / 60:.1f} minutes, Sample rate: {sample_rate}Hz")
    
    # Determine processing strategy
    needs_chunking = duration > args.threshold or args.force_chunk
    logger.info(f"Chunking {'enabled' if needs_chunking else 'disabled'}")
    
    # Set up device for processing
    device = "cpu" if args.force_cpu else "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load Whisper model
    logger.info(f"Loading Whisper model: {args.model}")
    try:
        model = whisper.load_model(args.model, device=device)
        logger.info("Model loaded successfully")
        log_gpu_memory()  # Log memory after model load
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        sys.exit(1)
    
    # Transcribe audio
    transcription = ""
    try:
        if needs_chunking:
            logger.info("Processing audio in chunks")
            transcription = split_with_soundfile(audio_file, model, args.chunk_length)
            
            if not transcription and processor.pydub_available:
                logger.info("Falling back to pydub processing")
                transcription = split_with_pydub(audio_file, model, args.chunk_length)
            
            if not transcription and processor.librosa_available:
                logger.info("Falling back to librosa processing")
                transcription = split_with_librosa(audio_file, model, args.chunk_length)
            
            if not transcription:
                logger.error("All chunking methods failed")
                sys.exit(1)
        else:
            logger.info("Processing audio as single file")
            result = model.transcribe(audio_file, fp16=(device == "cuda"), verbose=False)
            transcription = process_transcription_result(result['text'])
            
            if not transcription:
                logger.error("Transcription produced no text")
                sys.exit(1)
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up model and GPU memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Save transcription to file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / generate_output_filename(audio_file, args.model)
    
    try:
        output_path.write_text(transcription, encoding='utf-8')
        logger.info(f"Transcription saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save transcription: {str(e)}")
        sys.exit(1)
    
    # Log and display results
    logger.info("=" * 60)
    logger.info("TRANSCRIPTION COMPLETE")
    logger.info("=" * 60)
    print(transcription)
    logger.info(f"Transcription length: {len(transcription)} characters, {len(transcription.split())} words")
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()
