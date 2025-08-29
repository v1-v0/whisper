
import os
import time
from datetime import datetime
import soundfile as sf
import torch
import whisper

def get_audio_duration(file_path):
    """
    Returns the duration of the audio file in seconds using sf.info (does not load the file into memory).
    Handles errors if the file cannot be read.
    """
    try:
        info = sf.info(file_path)
        return info.frames / float(info.samplerate)
    except RuntimeError as e:
        print(f"Error reading audio file '{file_path}': {e}")
        return None
    except FileNotFoundError:
        print(f"Audio file not found: {file_path}")
        return None

def format_duration(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return int(hours), int(minutes)

def main():
    audio_file = '2024Nov11_optogenetics.wav'
    whisper_model = "large-v3"
    
    # Get audio duration
    audio_length = get_audio_duration(audio_file)
    if audio_length is None:
        print("Aborting due to audio file error.")
        return
    src_hours, src_minutes = format_duration(audio_length)
    print(f"Audio Duration: {src_hours} hours, {src_minutes} minutes")

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    # Load the Whisper model with error handling
    try:
        model = whisper.load_model(whisper_model).to(device)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return

    # Start timestamp
    start_time = time.time()
    start_dt = datetime.fromtimestamp(start_time)
    print(f"Start Time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # Transcribe audio with explicit fp16 setting
    result = model.transcribe(audio_file, fp16=False)
    # print(result['text'])

    # End timestamp and duration
    end_time = time.time()
    duration = end_time - start_time
    d_hours, d_minutes = format_duration(duration)
    print(f"Transcription Duration: {d_hours} hours, {d_minutes} minutes")

    # Write output to a file
    # Clean up model name for output filename
    model_name_for_file = whisper_model.replace('.', '_')

    filename = os.path.splitext(os.path.basename(audio_file))[0]
    output = f"{filename}_{model_name_for_file}_{start_dt.strftime('%Y%m%d%H%M')}.txt"

    with open(output, "w", encoding="utf-8") as f:
        f.write(result['text'])
    print(f"Transcription written to: {output}")

if __name__ == "__main__":
    main()
