import torch
import whisper
import os
import sys
import soundfile as sf

def validate_audio_file(file_path: str) -> bool:
    """Validate if the audio file exists and has a supported extension."""
    supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    if not os.path.isfile(file_path):
        print(f"Audio file not found: {file_path}")
        return False
    if not any(file_path.lower().endswith(ext) for ext in supported_extensions):
        print(f"Unsupported audio file format: {file_path}")
        return False
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_cuda.py <audio_file>")
        sys.exit(1)

    audio_file = ""

    if not validate_audio_file(audio_file):
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure you have a compatible GPU and the correct drivers installed.")
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "large-v3-turbo"
    print(f"Loading model '{model_name}'...")

    try:
        model = whisper.load_model(model_name, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Transcribe the audio file
    print(f"Transcribing audio file '{audio_file}'...")
    try:
        result = model.transcribe(audio_file, fp16=True)
        transcription = str(result['text'])
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)

    # Output the transcription
    print("\nTranscription Result:")
    print(transcription)

    # Save transcription to a file
    output_dir = "outputs"
    try:
        with open(os.path.join(output_dir, f"{os.path.basename(audio_file)}_{model_name.replace('.', '_')}_transcription.txt"), "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"Transcription saved to '{output_dir}' directory.")
    except Exception as e:
        print(f"Error saving transcription: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


