import whisper
import glob
import os

model = whisper.load_model("base")

# Find the first mp3 file in the source directory
mp3_files = glob.glob(os.path.join("source", "*.mp3"))
if mp3_files:
    audio_path = mp3_files[0]
    result = model.transcribe(audio_path)
    # Save the transcription to a text file
    output_path = os.path.join("outputs", "test.txt")
    
else:
    print("No mp3 files found in the source directory.")
