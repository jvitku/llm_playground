#!/usr/bin/env python3
import pyaudio
import wave

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024
SECONDS = 3
FILENAME = "recorded.wav"

pa = pyaudio.PyAudio()

# --- record ---
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                 input=True, frames_per_buffer=FRAMES_PER_BUFFER)
print("Recording...")
frames = [stream.read(FRAMES_PER_BUFFER) for _ in range(int(RATE / FRAMES_PER_BUFFER * SECONDS))]
stream.stop_stream(); stream.close()
print("Done recording")

# save to wav
wf = wave.open(FILENAME, "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(pa.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b"".join(frames))
wf.close()

# --- playback ---
print("Playing back...")
wf = wave.open(FILENAME, "rb")
stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),
                 channels=wf.getnchannels(),
                 rate=wf.getframerate(),
                 output=True)
data = wf.readframes(FRAMES_PER_BUFFER)
while data:
    stream.write(data)
    data = wf.readframes(FRAMES_PER_BUFFER)
stream.stop_stream(); stream.close(); pa.terminate()
print("Done")

