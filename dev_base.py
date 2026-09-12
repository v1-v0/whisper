import os
from datetime import datetime

import whisper


model_name = "turbo"


home_dir = os.path.expanduser("~")
download_dir = os.path.join(home_dir, "Downloads")

## Transcribe an audio file

source_dir = download_dir  # Change this to your desired source directory
audio_extensions = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
audio_files = [
    os.path.join(source_dir, filename)
    for filename in (os.listdir(source_dir) if os.path.isdir(source_dir) else ())
    if filename.lower().endswith(audio_extensions)
    and os.path.isfile(os.path.join(source_dir, filename))
]
path = max(audio_files, key=os.path.getmtime) if audio_files else None

# Check if the audio file exists
if path is None:
    print("Audio file not found.")
elif input("Use audio file - \n"
        f"'{path}'? \n"
        "Press ENTER to confirm or else to cancel: ").strip().lower() not in {"", "y", "yes"}:
    print("Transcription cancelled.")
else:
    model = whisper.load_model(model_name)
    print("Transcription started...", flush=True)
    result = model.transcribe(path, verbose=None)
    text = str(result["text"])
    #text = result["text"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.splitext(os.path.basename(path))[0]
    output_dir = download_dir
    output_path = os.path.join(
        output_dir,
        f"{source_name}_{model_name}_{timestamp}.txt",
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(text)

    #print(text)
    print(f"Saved transcription to: {output_path}")
