audioFile = '2024Nov11_optogenetics.wav'

import soundfile as sf
data, samplerate = sf.read(audioFile)
audio_length = len(data) / samplerate
print(f"Duration: {audio_length:.2f} seconds")
