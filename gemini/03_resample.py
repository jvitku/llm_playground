# python
"""
Drop-in replacement for your script's audio bits: robust device selection + rate fallback.
Requires: pip install google-genai pillow mss opencv-python pyaudio
"""

import os, asyncio, base64, io, traceback, argparse, re, time, struct
import pyaudio
from typing import Optional, Tuple
from google import genai
from google.genai import types

from pathlib import Path
import sys
# add project root (parent of "gemini" dir) to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from gemini.load_env import load_env_file

# --------- MODEL CONFIG ---------
#MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"
MODEL = "models/gemini-2.5-flash-exp-native-audio-thinking-dialog"

SEND_SAMPLE_RATE = 16000        # mic -> model
RECEIVE_SAMPLE_RATE = 24000     # model -> speaker (ideal)
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024
TARGET_DEV_RE = re.compile(r"usb|pnp|audio", re.I)  # prefer USB-ish devices

loaded_map = load_env_file()

# --------- CLIENT ---------
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"),
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)

# ---------- AUDIO UTILS ----------
pya = pyaudio.PyAudio()

def _pick_io_indices() -> Tuple[Optional[int], Optional[int]]:
    in_idx = out_idx = None
    # Try defaults first
    try:
        in_idx = pya.get_default_input_device_info().get("index")
    except Exception:
        pass
    try:
        out_idx = pya.get_default_output_device_info().get("index")
    except Exception:
        pass

    # Scan for USB devices as better candidates
    def name(i: int) -> str:
        try:
            return pya.get_device_info_by_index(i).get("name", "")
        except Exception:
            return ""

    def caps(i: int) -> Tuple[int, int]:
        try:
            d = pya.get_device_info_by_index(i)
            return d.get("maxInputChannels", 0), d.get("maxOutputChannels", 0)
        except Exception:
            return (0, 0)

    dev_count = pya.get_device_count()
    # Prefer USB for input
    if in_idx is None:
        for i in range(dev_count):
            nin, nout = caps(i)
            if nin > 0 and TARGET_DEV_RE.search(name(i)):
                in_idx = i
                break
    # If still none, take any input device
    if in_idx is None:
        for i in range(dev_count):
            if caps(i)[0] > 0:
                in_idx = i
                break

    # Prefer USB for output
    if out_idx is None:
        for i in range(dev_count):
            nin, nout = caps(i)
            if nout > 0 and TARGET_DEV_RE.search(name(i)):
                out_idx = i
                break
    if out_idx is None:
        for i in range(dev_count):
            if caps(i)[1] > 0:
                out_idx = i
                break

    return in_idx, out_idx

def _open_input_stream(in_idx: Optional[int]):
    return pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=in_idx,
        frames_per_buffer=CHUNK_SIZE,
    )

def _open_output_stream(out_idx: Optional[int]) -> Tuple[pyaudio.Stream, int]:
    """
    Try 24 kHz first. If the device chokes, fall back to 16 kHz and resample incoming audio.
    Returns (stream, output_rate).
    """
    try:
        st = pya.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
            output_device_index=out_idx,
        )
        return st, RECEIVE_SAMPLE_RATE
    except Exception:
        # Fall back to 16k
        st = pya.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            output=True,
            output_device_index=out_idx,
        )
        return st, SEND_SAMPLE_RATE

def _resample_16bit_mono(raw: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return raw
    
    # Convert bytes to list of 16-bit samples
    sample_count = len(raw) // 2
    samples = list(struct.unpack(f'<{sample_count}h', raw))
    
    # Simple linear resampling
    ratio = dst_rate / src_rate
    dst_length = int(len(samples) * ratio)
    resampled = []
    
    for i in range(dst_length):
        # Map destination index to source index
        src_index = i / ratio
        
        # Simple nearest neighbor interpolation
        if src_index >= len(samples) - 1:
            resampled.append(samples[-1])
        else:
            # Linear interpolation between two adjacent samples
            left_idx = int(src_index)
            right_idx = left_idx + 1
            frac = src_index - left_idx
            
            left_sample = samples[left_idx]
            right_sample = samples[right_idx] if right_idx < len(samples) else samples[left_idx]
            
            interpolated = int(left_sample + frac * (right_sample - left_sample))
            resampled.append(interpolated)
    
    # Convert back to bytes
    return struct.pack(f'<{len(resampled)}h', *resampled)

def test_audio_recording(in_idx: Optional[int], out_idx: Optional[int], duration: float = 3.0) -> bool:
    """
    Record audio for the specified duration, then play it back.
    Uses the same audio processing pipeline as the LLM session.
    Returns True if successful, False otherwise.
    """
    print(f"\nTesting audio system - recording {duration}s of audio...")
    
    try:
        # Open input stream for recording
        mic_stream = _open_input_stream(in_idx)
        recorded_chunks = []
        
        # Record audio
        chunks_per_second = SEND_SAMPLE_RATE // CHUNK_SIZE
        total_chunks = int(duration * chunks_per_second)
        
        print("Recording... speak now!")
        for i in range(total_chunks):
            try:
                data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                recorded_chunks.append(data)
            except Exception as e:
                print(f"Error during recording: {e}")
                mic_stream.close()
                return False
        
        mic_stream.close()
        print("Recording complete!")
        
        # Combine all recorded chunks
        recorded_audio = b''.join(recorded_chunks)
        
        # Open output stream for playback
        spk_stream, spk_rate = _open_output_stream(out_idx)
        
        # Resample recorded audio to output rate if necessary
        playback_audio = _resample_16bit_mono(recorded_audio, SEND_SAMPLE_RATE, spk_rate)
        
        print("Playing back recorded audio...")
        time.sleep(0.5)  # Brief pause before playback
        
        # Play back the audio
        spk_stream.write(playback_audio)
        spk_stream.close()
        
        print("Playback complete! Audio system test successful.\n")
        return True
        
    except Exception as e:
        print(f"Audio test failed: {e}")
        return False

# ---------- APP ----------
class AudioLoop:
    def __init__(self, video_mode: str = "none"):
        self.video_mode = video_mode
        self.audio_in_queue: asyncio.Queue = asyncio.Queue()
        self.out_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        self.session = None
        self.is_ai_speaking = False
        self.ai_speaking_lock = asyncio.Lock()
        self.mic_stream = None
        self.spk_stream = None
        self.spk_rate = RECEIVE_SAMPLE_RATE
        self.in_idx: Optional[int] = None
        self.out_idx: Optional[int] = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        # pick devices once
        self.in_idx, self.out_idx = _pick_io_indices()

        self.mic_stream = await asyncio.to_thread(_open_input_stream, self.in_idx)
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(self.mic_stream.read, CHUNK_SIZE, **kwargs)
            if not self.is_ai_speaking:
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    if not self.is_ai_speaking:
                        self.is_ai_speaking = True
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
            self.is_ai_speaking = False
            # flush leftover audio if the model was interrupted
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        # open speaker with fallback rate
        self.spk_stream, self.spk_rate = await asyncio.to_thread(_open_output_stream, self.out_idx)
        while True:
            bytestream = await self.audio_in_queue.get()
            # resample on the fly if device rate != model rate
            payload = _resample_16bit_mono(bytestream, RECEIVE_SAMPLE_RATE, self.spk_rate)
            await asyncio.to_thread(self.spk_stream.write, payload)

    async def run(self):
        # First, test audio recording and playback
        print("Initializing audio devices...")
        in_idx, out_idx = await asyncio.to_thread(_pick_io_indices)
        
        # Run audio test before starting LLM session
        audio_test_success = await asyncio.to_thread(test_audio_recording, in_idx, out_idx, 3.0)
        
        if not audio_test_success:
            print("Audio test failed. Please check your microphone and speakers.")
            return
        
        print("Audio test successful! Starting LLM session...")
        print("Type 'q' to quit the session.\n")
        
        try:
            async with (client.aio.live.connect(model=MODEL, config=CONFIG) as session, asyncio.TaskGroup() as tg):
                self.session = session
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                await send_text_task
                raise asyncio.CancelledError("User requested exit")
        except asyncio.CancelledError:
            pass
        except Exception as EG:
            if self.mic_stream:
                self.mic_stream.close()
            if self.spk_stream:
                self.spk_stream.close()
            traceback.print_exception(EG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="none", choices=["camera", "screen", "none"])
    args = parser.parse_args()
    asyncio.run(AudioLoop(video_mode=args.mode).run())
