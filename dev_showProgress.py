import contextlib
import importlib
import os
import sys
import time
from datetime import datetime
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, Optional

import tqdm

import whisper
from whisper.audio import HOP_LENGTH, SAMPLE_RATE


model_name = "large-v3-turbo"
stream_segments = False  # print each recognized segment above the progress bar

FRAME_SECONDS = HOP_LENGTH / SAMPLE_RATE  # 0.01 s of audio per mel frame

# NOTE: `whisper.transcribe` is the *function*; grab the module explicitly.
whisper_transcribe: ModuleType = importlib.import_module("whisper.transcribe")
# Patch through the namespace dict: ModuleType has no declared `tqdm`/`print`
# attributes, so direct assignment trips Pylance's reportAttributeAccessIssue.
whisper_globals: dict[str, Any] = vars(whisper_transcribe)


def format_clock(seconds: float) -> str:
    seconds = max(int(seconds + 0.5), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class TranscriptionProgressBar(tqdm.tqdm):
    """tqdm bar reporting audio position, ETA, realtime factor and segment count."""

    active: ClassVar[Optional["TranscriptionProgressBar"]] = None

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = False  # whisper disables the bar when verbose=True
        kwargs.setdefault("unit", "frames")
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("smoothing", 0.05)
        kwargs.setdefault("colour", "green")
        kwargs.setdefault("desc", "Transcribing")
        kwargs.setdefault(
            "bar_format",
            "{desc} {percentage:5.1f}%|{bar}| {postfix} [{elapsed}<{remaining}]",
        )
        super().__init__(*args, **kwargs)
        self.started_at = time.monotonic()
        self.segments_seen = 0
        self._refresh_postfix(self.n)
        self.refresh()

    def __enter__(self):
        TranscriptionProgressBar.active = self
        return super().__enter__()

    def __exit__(self, *exc_info):
        TranscriptionProgressBar.active = None
        return super().__exit__(*exc_info)

    def update(self, n=1):
        self._refresh_postfix(self.n + (n or 0))
        return super().update(n)

    def note_segment(self):
        self.segments_seen += 1

    def _refresh_postfix(self, frames):
        done = frames * FRAME_SECONDS
        total = (self.total or 0) * FRAME_SECONDS
        elapsed = max(time.monotonic() - self.started_at, 1e-6)
        self.set_postfix_str(
            f"{format_clock(done)}/{format_clock(total)} audio, "
            f"{done / elapsed:4.1f}x realtime, {self.segments_seen} seg",
            refresh=False,
        )


def bar_aware_print(*args, **kwargs):
    """Route whisper's segment output through the bar so it stays intact."""
    bar = TranscriptionProgressBar.active
    line = kwargs.get("sep", " ").join(str(arg) for arg in args)
    if bar is None:
        print(line, **kwargs)
        return
    if line.startswith("["):  # "[00:00.000 --> 00:04.120]  text"
        bar.note_segment()
    bar.write(line, file=kwargs.get("file") or sys.stdout)


@contextlib.contextmanager
def enriched_progress():
    """Temporarily patch whisper.transcribe's tqdm and print hooks."""
    original_tqdm = whisper_globals.get("tqdm", tqdm)
    had_print = "print" in whisper_globals  # normally False: print is a builtin
    original_print = whisper_globals.get("print")

    # whisper uses `import tqdm` + `tqdm.tqdm(...)`, but stay tolerant either way.
    whisper_globals["tqdm"] = (
        SimpleNamespace(tqdm=TranscriptionProgressBar)
        if hasattr(original_tqdm, "tqdm")
        else TranscriptionProgressBar
    )
    whisper_globals["print"] = bar_aware_print
    try:
        yield
    finally:
        whisper_globals["tqdm"] = original_tqdm
        if had_print:
            whisper_globals["print"] = original_print
        else:
            whisper_globals.pop("print", None)  # fall back to the builtin


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
    print(f"Loading '{model_name}' model...", end=" ", flush=True)
    load_started = time.monotonic()
    model = whisper.load_model(model_name)
    device = getattr(model, "device", None)
    device_name = getattr(device, "type", str(device or "cpu")).upper()
    print(f"ready on {device_name} in {time.monotonic() - load_started:.1f}s")

    # Decode the audio once so the duration is known before transcribing.
    audio = whisper.load_audio(path)
    duration = len(audio) / SAMPLE_RATE
    print(
        f"Audio: {os.path.basename(path)} "
        f"({os.path.getsize(path) / 2**20:.1f} MiB, {format_clock(duration)})",
        flush=True,
    )

    started = time.monotonic()
    with enriched_progress():
        result = model.transcribe(
            audio,
            verbose=bool(stream_segments),  # True: bar + text, False: bar only
            fp16=device_name == "CUDA",     # silences the CPU fp16 warning
        )
    elapsed = time.monotonic() - started

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
    segments = result.get("segments") or []
    print(
        f"Finished in {format_clock(elapsed)} "
        f"({duration / max(elapsed, 1e-6):.1f}x realtime) - "
        f"{len(segments)} segments, {len(text.split())} words, "
        f"language: {result.get('language', 'unknown')}"
    )
    print(f"Saved transcription to: {output_path}")