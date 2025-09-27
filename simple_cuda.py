import torch
import whisper
import os
import sys
import soundfile as sf
import logging
import argparse
import tempfile
import numpy as np
import datetime
from typing import Optional, List, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handle audio processing with multiple fallback methods."""
    
    def __init__(self):
        self.pydub_available = self._check_pydub()
        self.librosa_available = self._check_librosa()
    
    def _check_pydub(self) -> bool:
        """Check if pydub is available and working."""
        try:
            from pydub import AudioSegment
            # Test basic functionality
            AudioSegment.silent(duration=100)
            logger.info("Pydub is available")
            return True
        except Exception as e:
            logger.warning(f"Pydub not available: {e}")
            return False
    
    def _check_librosa(self) -> bool:
        """Check if librosa is available."""
        try:
            import librosa
            logger.info("Librosa is available")
            return True
        except ImportError:
            logger.warning("Librosa not available")
            return False
    
    def get_capabilities(self) -> str:
        """Return available audio processing capabilities."""
        capabilities = []
        if self.pydub_available:
            capabilities.append("pydub")
        if self.librosa_available:
            capabilities.append("librosa")
        capabilities.append("soundfile (native)")
        return ", ".join(capabilities)

def validate_audio_file(file_path: str) -> bool:
    """Validate audio file exists and has supported extension."""
    supported_extensions = {
        '.wav', '.mp3', '.flac', '.ogg', 
        '.m4a', '.aac', '.wma'
    }
    
    if not Path(file_path).is_file():
        logger.error(f"Audio file not found: {file_path}")
        return False
    
    file_lower = file_path.lower()
    has_supported_ext = any(
        file_lower.endswith(ext) for ext in supported_extensions
    )
    
    if not has_supported_ext:
        ext_list = ', '.join(supported_extensions)
        logger.error(f"Unsupported format. Supported: {ext_list}")
        return False
    
    return True

def get_audio_info(file_path: str) -> Tuple[float, int]:
    """Return duration and sample rate of audio file."""
    try:
        info = sf.info(file_path)
        duration = float(info.frames) / float(info.samplerate)
        return duration, int(info.samplerate)
    except Exception as e:
        logger.error(f"Error reading audio info: {e}")
        return 0.0, 0

def split_with_soundfile(
    file_path: str, 
    model, 
    chunk_length_s: int = 600
) -> str:
    """Split audio using soundfile - most reliable method."""
    try:
        logger.info(f"Using soundfile for chunking: {file_path}")
        
        # Load audio data
        audio_data, sample_rate = sf.read(file_path, dtype='float32')
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        total_duration = len(audio_data) / sample_rate
        logger.info(f"Loaded: {total_duration:.1f}s at {sample_rate}Hz")
        
        # Calculate chunks - fix the type issue
        chunk_samples = int(chunk_length_s * sample_rate)
        num_chunks = int(np.ceil(len(audio_data) / chunk_samples))
        
        logger.info(
            f"Processing {num_chunks} chunks of "
            f"{chunk_length_s}s each"
        )
        
        transcription_parts: List[str] = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, len(audio_data))
            
            chunk_data = audio_data[start_idx:end_idx]
            chunk_duration = len(chunk_data) / sample_rate
            
            # Create temporary file for this chunk
            with tempfile.NamedTemporaryFile(
                suffix='.wav', 
                delete=False
            ) as tmp_file:
                try:
                    sf.write(tmp_file.name, chunk_data, sample_rate)
                    
                    chunk_num = i + 1
                    logger.info(
                        f"Transcribing chunk {chunk_num}/"
                        f"{num_chunks} ({chunk_duration:.1f}s)..."
                    )
                    
                    result = model.transcribe(
                        tmp_file.name, 
                        fp16=torch.cuda.is_available(),
                        verbose=False  # Reduce whisper output
                    )
                    
                    chunk_text = result['text']
                    if isinstance(chunk_text, str):
                        chunk_text = chunk_text.strip()
                        if chunk_text:
                            transcription_parts.append(chunk_text)
                            char_count = len(chunk_text)
                            logger.info(
                                f"Chunk {chunk_num}: "
                                f"{char_count} chars transcribed"
                            )
                        else:
                            logger.warning(
                                f"Chunk {chunk_num}: "
                                f"No transcription generated"
                            )
                    else:
                        result_type = type(chunk_text)
                        logger.error(
                            f"Chunk {chunk_num}: "
                            f"Unexpected result type: {result_type}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
                    continue
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_file.name)
                    except OSError:
                        pass
        
        # Clean up audio data from memory
        del audio_data
        
        result_text = " ".join(transcription_parts)
        char_count = len(result_text)
        logger.info(f"Total transcription: {char_count} characters")
        return result_text
        
    except Exception as e:
        logger.error(f"Soundfile chunking failed: {e}")
        return ""

def split_with_pydub(
    file_path: str, 
    model, 
    chunk_length_s: int = 600
) -> str:
    """Split audio using pydub (fallback method)."""
    try:
        from pydub import AudioSegment
        
        logger.info(f"Using pydub for chunking: {file_path}")
        audio = AudioSegment.from_file(file_path)
        
        chunk_length_ms = chunk_length_s * 1000
        duration_ms = len(audio)
        
        transcription_parts: List[str] = []
        
        for i in range(0, duration_ms, chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            
            with tempfile.NamedTemporaryFile(
                suffix='.wav', 
                delete=False
            ) as tmp_file:
                try:
                    chunk.export(tmp_file.name, format="wav")
                    
                    chunk_num = i // chunk_length_ms + 1
                    logger.info(
                        f"Transcribing pydub chunk {chunk_num}..."
                    )
                    
                    result = model.transcribe(
                        tmp_file.name, 
                        fp16=torch.cuda.is_available()
                    )
                    
                    chunk_text = result['text']
                    if isinstance(chunk_text, str):
                        chunk_text = chunk_text.strip()
                        if chunk_text:
                            transcription_parts.append(chunk_text)
                        
                except Exception as e:
                    logger.error(f"Error with pydub chunk: {e}")
                    continue
                finally:
                    try:
                        os.unlink(tmp_file.name)
                    except OSError:
                        pass
        
        return " ".join(transcription_parts)
        
    except Exception as e:
        logger.error(f"Pydub chunking failed: {e}")
        return ""

def split_with_librosa(
    file_path: str, 
    model, 
    chunk_length_s: int = 600
) -> str:
    """Split audio using librosa (another fallback method)."""
    try:
        import librosa
        
        logger.info(f"Using librosa for chunking: {file_path}")
        audio_data, sample_rate = librosa.load(
            file_path, 
            sr=16000, 
            mono=True
        )
        
        total_duration = len(audio_data) / sample_rate
        chunk_samples = int(chunk_length_s * sample_rate)
        
        transcription_parts: List[str] = []
        
        for i in range(0, len(audio_data), chunk_samples):
            chunk_data = audio_data[i:i + chunk_samples]
            
            with tempfile.NamedTemporaryFile(
                suffix='.wav', 
                delete=False
            ) as tmp_file:
                try:
                    sf.write(tmp_file.name, chunk_data, sample_rate)
                    
                    result = model.transcribe(
                        tmp_file.name, 
                        fp16=torch.cuda.is_available()
                    )
                    chunk_text = result['text']
                    
                    if isinstance(chunk_text, str):
                        chunk_text = chunk_text.strip()
                        if chunk_text:
                            transcription_parts.append(chunk_text)
                        
                except Exception as e:
                    logger.error(f"Error with librosa chunk: {e}")
                    continue
                finally:
                    try:
                        os.unlink(tmp_file.name)
                    except OSError:
                        pass
        
        return " ".join(transcription_parts)
        
    except Exception as e:
        logger.error(f"Librosa chunking failed: {e}")
        return ""

def find_audio_file(input_dir: str = "source") -> Optional[str]:
    """Find first valid audio file in specified directory."""
    if not Path(input_dir).exists():
        logger.error(f"Directory '{input_dir}' does not exist.")
        return None
    
    audio_files = []
    for file_path in Path(input_dir).iterdir():
        if file_path.is_file() and validate_audio_file(str(file_path)):
            audio_files.append(str(file_path))
    
    if not audio_files:
        logger.error(f"No valid audio files found in '{input_dir}'.")
        return None
    
    if len(audio_files) > 1:
        first_file = Path(audio_files[0]).name
        logger.warning(f"Multiple files found. Using: {first_file}")
    
    return audio_files[0]

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Whisper with CUDA."
    )
    
    parser.add_argument(
        "audio_file", 
        nargs="?", 
        help="Path to audio file"
    )
    
    model_choices = [
        'tiny', 'base', 'small', 'medium', 'large', 
        'large-v2', 'large-v3', 'large-v3-turbo'
    ]
    parser.add_argument(
        "--model", 
        default="large-v3-turbo",
        choices=model_choices,
        help="Whisper model to use"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="outputs", 
        help="Output directory"
    )
    
    parser.add_argument(
        "--chunk-length", 
        type=int, 
        default=600, 
        help="Chunk length in seconds"
    )
    
    parser.add_argument(
        "--threshold", 
        type=int, 
        default=600, 
        help="Duration threshold for chunking (seconds)"
    )
    
    parser.add_argument(
        "--force-chunk", 
        action="store_true", 
        help="Force chunking regardless of duration"
    )
    
    return parser.parse_args()

def generate_output_filename(audio_file: str, model_name: str) -> str:
    """Generate output filename with timestamp."""
    base_name = Path(audio_file).stem
    model_safe = model_name.replace('.', '_')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    filename = (
        f"{base_name}_{model_safe}_"
        f"{timestamp}_transcription.txt"
    )
    return filename

def main():
    args = parse_args()
    
    # Initialize audio processor
    processor = AudioProcessor()
    capabilities = processor.get_capabilities()
    logger.info(f"Available audio processing: {capabilities}")
    
    # Find audio file
    if args.audio_file:
        if not validate_audio_file(args.audio_file):
            sys.exit(1)
        audio_file = args.audio_file
    else:
        audio_file = find_audio_file()
        if not audio_file:
            sys.exit(1)
    
    logger.info(f"Processing: {audio_file}")
    
    # Get audio info
    duration, sample_rate = get_audio_info(audio_file)
    if duration == 0:
        logger.error("Could not read audio file.")
        sys.exit(1)
    
    duration_minutes = duration / 60
    logger.info(
        f"Duration: {duration_minutes:.1f} minutes, "
        f"Sample rate: {sample_rate}Hz"
    )
    
    # Check if chunking is needed
    needs_chunking = duration > args.threshold or args.force_chunk
    
    # Setup CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model '{args.model}'...")
    try:
        model = whisper.load_model(args.model, device=device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Process audio
    transcription = ""
    try:
        if needs_chunking:
            logger.info("Using chunked processing...")
            
            # Try methods in order of reliability
            transcription = split_with_soundfile(
                audio_file, 
                model, 
                args.chunk_length
            )
            
            if not transcription and processor.pydub_available:
                logger.info("Trying pydub fallback...")
                transcription = split_with_pydub(
                    audio_file, 
                    model, 
                    args.chunk_length
                )
            
            if not transcription and processor.librosa_available:
                logger.info("Trying librosa fallback...")
                transcription = split_with_librosa(
                    audio_file, 
                    model, 
                    args.chunk_length
                )
            
            if not transcription:
                logger.error("All chunking methods failed!")
                sys.exit(1)
        else:
            logger.info("Processing entire file...")
            fp16_enabled = (device == "cuda")
            result = model.transcribe(audio_file, fp16=fp16_enabled)
            transcription_result = result['text']
            
            if isinstance(transcription_result, str):
                transcription = transcription_result
            else:
                result_type = type(transcription_result)
                logger.error(
                    f"Unexpected transcription type: {result_type}"
                )
                sys.exit(1)
        
        if not transcription.strip():
            logger.warning("Empty transcription result!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except:
            pass
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_name = generate_output_filename(audio_file, args.model)
    output_path = output_dir / output_name
    
    try:
        output_path.write_text(
            transcription.strip(), 
            encoding='utf-8'
        )
        logger.info(f"Saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save: {e}")
        sys.exit(1)
    
    # Display results
    logger.info("=" * 60)
    logger.info("TRANSCRIPTION COMPLETE")
    logger.info("=" * 60)
    print(transcription.strip())
    logger.info("=" * 60)
    
    char_count = len(transcription)
    word_count = len(transcription.split())
    logger.info(f"Characters: {char_count}")
    logger.info(f"Words: {word_count}")
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()