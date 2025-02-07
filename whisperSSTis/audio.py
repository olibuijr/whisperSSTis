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


def record_audio(duration: int, device_id=None, samplerate: int = 16000):
    """
    Record audio from the selected microphone.
    """
    try:
        print("Recording...")
        sd.default.reset()
        device_info = sd.query_devices(device=device_id, kind='input')
        native_samplerate = int(device_info['default_samplerate'])
        sd.default.device = (device_id, None)
        sd.default.samplerate = native_samplerate
        sd.default.channels = (1, None)
        sd.default.dtype = np.float32

        num_frames = int(duration * native_samplerate)
        audio_data = np.zeros(num_frames, dtype=np.float32)
        frames_read = 0

        with sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=native_samplerate,
            dtype=np.float32,
            blocksize=1024,
            latency='low'
        ) as stream:
            while frames_read < num_frames:
                data, overflowed = stream.read(min(1024, num_frames - frames_read))
                if overflowed:
                    logging.warning("Audio buffer overflowed - some samples may be lost")
                audio_data[frames_read:frames_read + len(data)] = data.flatten()
                frames_read += len(data)
        
        # Resample to target sample rate if necessary
        if native_samplerate != samplerate:
            from scipy import signal
            audio_data = signal.resample(audio_data, int(len(audio_data) * samplerate / native_samplerate))
        
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