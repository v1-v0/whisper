import os
from datetime import datetime

import whisper


model_name = "turbo"
model = whisper.load_model(model_name)

## Transcribe an audio file

source_dir = os.path.join(os.getcwd(), "sources")
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
else:
    result = model.transcribe(path)
    text = str(result["text"])
    #text = result["text"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = os.path.splitext(os.path.basename(path))[0]
    output_dir = os.path.join(os.getcwd(), "transcriptions")
    output_path = os.path.join(
        output_dir,
        f"{source_name}_{model_name}_{timestamp}.txt",
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(text)

    #print(text)
    print(f"Saved transcription to: {output_path}")
