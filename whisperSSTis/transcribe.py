import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging
import math
from datetime import timedelta, datetime
import re
import numpy as np


def load_model():
    """
    Load the Whisper model and processor.
    """
    try:
        model_name = "carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h"
        processor = WhisperProcessor.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        return model, processor
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise


def transcribe_audio(audio_data, model, processor, sample_rate: int = 16000):
    """
    Transcribe audio using the Whisper model.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_features = processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        ).input_features.to(device)

        attention_mask = torch.ones_like(input_features)
        
        # Generate without forcing language tokens
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                task="transcribe",
                language="<|is|>"  # Use language tag instead of forced decoder ids
            )[0]
            
        transcription = processor.decode(predicted_ids, skip_special_tokens=True)
        return transcription
    except Exception as e:
        logging.error(f"Error transcribing audio: {str(e)}")
        raise


def format_timestamp(seconds: float):
    """
    Convert seconds to a timestamp format.
    """
    return str(timedelta(seconds=int(seconds)))


def transcribe_long_audio(audio_data, model, processor, duration: float, chunk_size: int = 30, sample_rate: int = 16000):
    """
    Transcribe long audio by breaking it into chunks with timestamps.
    """
    try:
        chunk_samples = chunk_size * sample_rate
        total_samples = len(audio_data)
        num_chunks = math.ceil(total_samples / chunk_samples)
        transcriptions = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, total_samples)
            chunk = audio_data[start_idx:end_idx]
            input_features = processor(
                chunk,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            ).input_features.to("cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    task="transcribe",
                    language="<|is|>"  # Use language tag instead of forced decoder ids
                )[0]
                
            chunk_transcription = processor.decode(predicted_ids, skip_special_tokens=True)
            start_time = format_timestamp(i * chunk_size)
            end_time = format_timestamp(min((i + 1) * chunk_size, duration))
            transcriptions.append(f"[{start_time} → {end_time}] {chunk_transcription}")
        
        return transcriptions
    except Exception as e:
        logging.error(f"Error transcribing long audio: {str(e)}")
        raise


def create_srt(transcriptions):
    """
    Convert transcriptions to SRT format.
    """
    srt_lines = []
    for i, trans in enumerate(transcriptions, start=1):
        timestamp_match = re.match(r'\[(.*?) → (.*?)\] (.*)', trans)
        if timestamp_match:
            start_time_str, end_time_str, text = timestamp_match.groups()
            start_time = datetime.strptime(start_time_str, "%H:%M:%S")
            end_time = datetime.strptime(end_time_str, "%H:%M:%S")

            start_time_srt = start_time.strftime("%H:%M:%S,%f")[:-3]
            end_time_srt = end_time.strftime("%H:%M:%S,%f")[:-3]

            srt_lines.extend([
                str(i),
                f"{start_time_srt} --> {end_time_srt}",
                text,
                ""
            ])
    return "\n".join(srt_lines)
