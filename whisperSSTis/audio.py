import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path
from datetime import timedelta
import logging


def get_audio_devices() -> dict:
    """
    Get available audio input devices with proper sample rate support.
    """
    devices = sd.query_devices()
    input_devices = {}
    for i, device in enumerate(devices):
        try:
            if device['max_input_channels'] > 0:
                sd.check_input_settings(
                    device=i,
                    channels=1,
                    samplerate=16000,
                    dtype=np.float32
                )
                input_devices[f"{device['name']} (ID: {i})"] = i
        except sd.PortAudioError:
            continue
    return input_devices


class AudioStream:
    """
    A class to handle continuous audio streaming with buffer management.
    """
    def __init__(self, device_id=None, samplerate: int = 16000, chunk_size: int = 1024):
        self.device_id = device_id
        self.target_samplerate = samplerate
        self.chunk_size = chunk_size
        self.stream = None
        self.audio_buffer = []
        self.is_recording = False
        self.current_audio_level = 0
        self._setup_device()

    def _setup_device(self):
        """Initialize audio device settings."""
        sd.default.reset()
        self.device_info = sd.query_devices(device=self.device_id, kind='input')
        self.native_samplerate = int(self.device_info['default_samplerate'])
        sd.default.device = (self.device_id, None)
        sd.default.samplerate = self.native_samplerate
        sd.default.channels = (1, None)
        sd.default.dtype = np.float32

    def _audio_callback(self, indata, frames, time, status):
        """Handle incoming audio data."""
        if status:
            logging.warning(f"Audio callback status: {status}")
        
        # Calculate audio level (RMS)
        self.current_audio_level = np.sqrt(np.mean(indata**2))
        
        # Add to buffer
        if self.is_recording:
            self.audio_buffer.append(indata.copy())

    def start_stream(self):
        """Start the audio stream."""
        if self.stream is not None and self.stream.active:
            return

        self.stream = sd.InputStream(
            device=self.device_id,
            channels=1,
            samplerate=self.native_samplerate,
            dtype=np.float32,
            blocksize=self.chunk_size,
            latency='low',
            callback=self._audio_callback
        )
        self.stream.start()
        self.is_recording = True
        self.audio_buffer = []

    def stop_stream(self):
        """Stop the audio stream."""
        self.is_recording = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_audio_level(self):
        """Get the current audio input level."""
        return self.current_audio_level

    def get_audio_chunk(self):
        """Get and process the next chunk of audio data."""
        if not self.audio_buffer:
            return None

        # Get the oldest chunk from buffer
        chunk = np.concatenate(self.audio_buffer[:1])
        self.audio_buffer = self.audio_buffer[1:]

        # Resample if necessary
        if self.native_samplerate != self.target_samplerate:
            from scipy import signal
            chunk = signal.resample(
                chunk,
                int(len(chunk) * self.target_samplerate / self.native_samplerate)
            )

        return chunk.flatten()

def record_audio(duration: int, device_id=None, samplerate: int = 16000):
    """
    Record audio from the selected microphone.
    """
    try:
        print("Recording...")
        stream = AudioStream(device_id=device_id, samplerate=samplerate)
        stream.start_stream()
        
        # Record for specified duration
        audio_chunks = []
        frames_recorded = 0
        frames_needed = int(duration * samplerate)
        
        while frames_recorded < frames_needed:
            chunk = stream.get_audio_chunk()
            if chunk is not None:
                audio_chunks.append(chunk)
                frames_recorded += len(chunk)
        
        stream.stop_stream()
        
        # Combine all chunks
        audio_data = np.concatenate(audio_chunks)
        # Trim to exact duration
        audio_data = audio_data[:frames_needed]
        
        return audio_data
    except Exception as e:
        logging.error(f"Recording error: {str(e)}")
        raise


def get_file_info(audio_data, sample_rate: int):
    """
    Get detailed information about the audio data.
    """
    duration = len(audio_data) / sample_rate
    return {
        "Duration": str(timedelta(seconds=int(duration))),
        "Channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
        "Sample Rate": f"{sample_rate} Hz",
        "File Size": f"{audio_data.nbytes / (1024*1024):.2f} MB"
    }


def load_audio_file(uploaded_file, target_sr: int = 16000):
    """
    Load and preprocess an uploaded audio file.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        audio_data, sr = sf.read(tmp_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        file_info = get_file_info(audio_data, sr)
        if sr != target_sr:
            from scipy import signal
            audio_data = signal.resample(audio_data, int(len(audio_data) * target_sr / sr))
        os.unlink(tmp_path)
        duration = len(audio_data) / target_sr
        return audio_data, duration, file_info
    except Exception as e:
        logging.error(f"Error loading audio file: {str(e)}")
        raise
