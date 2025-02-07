# Import necessary libraries
import streamlit as st  # Web application framework

# Configure page layout - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="WhisperSST.is - Icelandic Speech Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rest of imports
import torch  # Deep learning framework
from transformers import WhisperProcessor, WhisperForConditionalGeneration  # Whisper model components
import numpy as np  # Numerical computing
import sounddevice as sd  # Audio recording
import soundfile as sf  # Audio file handling
import tempfile  # Temporary file operations
import time  # Time-related functions
import os  # Operating system interface
import warnings
from typing import Dict, List
import io
from pathlib import Path
import math
from datetime import timedelta
import re

# Custom CSS for better styling
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
    }
    .stButton > button {
        width: 100%;
    }
    .upload-header {
        margin-bottom: 2rem;
    }
    .file-info {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Ignore specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Disable Streamlit's file watcher for problematic modules
if hasattr(st, '_is_running_with_streamlit'):
    import sys
    sys.modules['torch._classes'] = None
    if 'torch._classes' in sys.modules:
        del sys.modules['torch._classes']

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'processor' not in st.session_state:
    st.session_state['processor'] = None
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the Whisper model and processor."""
    try:
        model_name = "carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h"
        processor = WhisperProcessor.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

def get_audio_devices() -> Dict[str, int]:
    """Get available audio input devices with proper sample rate support."""
    devices = sd.query_devices()
    input_devices = {}
    for i, device in enumerate(devices):
        try:
            if device['max_input_channels'] > 0:
                # Test if device supports our required sample rate
                sd.check_input_settings(
                    device=i,
                    channels=1,
                    samplerate=16000,
                    dtype=np.float32
                )
                # Only add device if it passes the check
                input_devices[f"{device['name']} (ID: {i})"] = i
        except sd.PortAudioError:
            continue  # Skip devices that don't support our requirements
    return input_devices

def record_audio(duration, device_id=None, samplerate=16000):
    """Record audio from selected microphone."""
    try:
        st.write("🎤 Recording...")
        # Configure sounddevice settings
        sd.default.reset()
        
        # Get device info
        device_info = sd.query_devices(device=device_id, kind='input')
        
        # Use device's native sample rate for recording
        native_samplerate = int(device_info['default_samplerate'])
        
        # Configure device
        sd.default.device = (device_id, None)  # Input device only
        sd.default.samplerate = native_samplerate
        sd.default.channels = (1, None)  # Mono input
        sd.default.dtype = np.float32
        
        # Calculate number of frames based on native sample rate
        num_frames = int(duration * native_samplerate)
        
        # Start recording with a small buffer
        with sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=native_samplerate,
            dtype=np.float32,
            blocksize=1024,
            latency='low'
        ) as stream:
            # Pre-allocate the entire buffer
            audio_data = np.zeros(num_frames, dtype=np.float32)
            frames_read = 0
            
            st.write("Recording in progress...")
            
            while frames_read < num_frames:
                data, overflowed = stream.read(min(1024, num_frames - frames_read))
                if overflowed:
                    st.warning("Audio buffer overflowed - some samples may be lost")
                audio_data[frames_read:frames_read + len(data)] = data.flatten()
                frames_read += len(data)
        
        st.write("✅ Recording complete!")
        
        # Resample to target sample rate if necessary
        if native_samplerate != samplerate:
            from scipy import signal
            audio_data = signal.resample(audio_data, 
                                       int(len(audio_data) * samplerate / native_samplerate))
        
        return audio_data
    except Exception as e:
        st.error(f"Recording error: {str(e)}")
        st.error("Try selecting a different input device or checking your microphone permissions")
        return None

def transcribe_audio(audio_data, model, processor):
    """Transcribe audio using the Whisper model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Process audio with attention mask
    input_features = processor(
        audio_data,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_features.to(device)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_features)
    
    # Generate transcription with explicit Icelandic language setting
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="is",  # Explicitly set Icelandic
            task="transcribe",  # Set task to transcribe
            attention_mask=attention_mask  # Add attention mask
        )[0]
    
    transcription = processor.decode(predicted_ids, skip_special_tokens=True)
    return transcription

def format_timestamp(seconds):
    """Convert seconds to timestamp format."""
    return str(timedelta(seconds=int(seconds)))

def get_file_info(audio_data, sample_rate):
    """Get detailed information about the audio file."""
    duration = len(audio_data) / sample_rate
    return {
        "Duration": str(timedelta(seconds=int(duration))),
        "Channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
        "Sample Rate": f"{sample_rate} Hz",
        "File Size": f"{audio_data.nbytes / (1024*1024):.2f} MB"
    }

def load_audio_file(uploaded_file, target_sr=16000):
    """Load and preprocess uploaded audio file."""
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load the audio file using soundfile
        audio_data, sr = sf.read(tmp_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Get file info before preprocessing
        file_info = get_file_info(audio_data, sr)
        
        # Resample if necessary
        if sr != target_sr:
            from scipy import signal
            audio_data = signal.resample(audio_data, 
                                       int(len(audio_data) * target_sr / sr))
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        duration = len(audio_data) / target_sr
        return audio_data, duration, file_info
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None, 0, None

def transcribe_long_audio(audio_data, model, processor, duration, chunk_size=30):
    """Transcribe long audio files in chunks with timestamps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 16000
    chunk_samples = chunk_size * sample_rate
    
    # Calculate number of chunks
    total_samples = len(audio_data)
    num_chunks = math.ceil(total_samples / chunk_samples)
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    transcriptions = []
    
    for i in range(num_chunks):
        # Update progress
        progress = (i + 1) / num_chunks
        progress_bar.progress(progress)
        status_text.text(f"Processing chunk {i + 1} of {num_chunks}...")
        
        # Extract chunk
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, total_samples)
        chunk = audio_data[start_idx:end_idx]
        
        # Process audio chunk
        input_features = processor(
            chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        ).input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="is",
                task="transcribe"
            )[0]
        
        chunk_transcription = processor.decode(predicted_ids, skip_special_tokens=True)
        
        # Add timestamp
        start_time = format_timestamp(i * chunk_size)
        end_time = format_timestamp(min((i + 1) * chunk_size, duration))
        
        transcriptions.append(f"[{start_time} → {end_time}] {chunk_transcription}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return transcriptions

def main():
    """Main application function with improved UI/UX."""
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/ce/Whisper_%28software%29_logo.svg", width=100)
        st.title("WhisperSST.is")
        st.markdown("---")
        
        st.subheader("ℹ️ About")
        st.markdown("""
        WhisperSST.is provides real-time Icelandic speech recognition using 
        OpenAI's Whisper model, fine-tuned for Icelandic.
        """)
        
        st.markdown("---")
        st.subheader("🔧 System Info")
        device = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
        st.info(f"Using: {device}")
        if torch.cuda.is_available():
            st.success(f"GPU: {torch.cuda.get_device_name(0)}")

    # Main content
    st.title("Icelandic Speech Recognition")
    
    # Load model (with custom spinner)
    if st.session_state['model'] is None:
        with st.spinner("🤖 Loading AI model... This might take a minute..."):
            model, processor = load_model()
            st.session_state['model'] = model
            st.session_state['processor'] = processor
        st.success("✅ Model loaded successfully!")

    # Create tabs with icons
    tab1, tab2 = st.tabs(["🎤 Record Audio", "📁 Upload Audio"])
    
    with tab1:
        st.markdown("""
        <div class="upload-header">
        <h3>Record Your Voice</h3>
        <p>Speak in Icelandic and get instant transcription.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Recording controls in a clean card-like container
            with st.container():
                st.markdown("##### Device Settings")
                input_devices = get_audio_devices()
                if not input_devices:
                    st.error("❌ No input devices found!")
                    return
                
                device_name = st.selectbox(
                    "🎙️ Select Input Device",
                    options=list(input_devices.keys()),
                    index=0
                )
                selected_device_id = input_devices[device_name]
                
                duration = st.slider(
                    "⏱️ Recording Duration (seconds)", 
                    min_value=1, 
                    max_value=30, 
                    value=5,
                    help="Choose how long you want to record"
                )
                
                if st.button("🎙️ Start Recording", use_container_width=True):
                    try:
                        with st.spinner("🎤 Recording in progress..."):
                            audio_data = record_audio(duration, selected_device_id)
                            
                        if audio_data is not None:
                            st.session_state['audio_data'] = audio_data
                            
                            # Save and display audio
                            temp_audio_path = "temp_recording.wav"
                            sf.write(temp_audio_path, audio_data, 16000)
                            st.audio(temp_audio_path)
                            
                            # Transcribe with a nice spinner
                            with st.spinner("🤖 Processing your speech..."):
                                transcription = transcribe_audio(
                                    audio_data,
                                    st.session_state['model'],
                                    st.session_state['processor']
                                )
                            
                            # Display transcription in a nice container
                            st.markdown("##### 📝 Transcription")
                            st.markdown(f'<div class="file-info">{transcription}</div>', 
                                      unsafe_allow_html=True)
                            
                            os.remove(temp_audio_path)
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            st.markdown("##### 📋 Instructions")
            st.info("""
            1. Select your microphone from the dropdown
            2. Set your preferred recording duration
            3. Click "Start Recording" and speak in Icelandic
            4. Wait for the transcription to appear
            
            **Note:** Ensure your microphone is properly connected and permitted.
            """)

    with tab2:
        st.markdown("""
        <div class="upload-header">
        <h3>Upload Audio File</h3>
        <p>Process longer recordings like podcasts or interviews.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload section
        upload_col1, upload_col2 = st.columns([3, 2])
        
        with upload_col1:
            st.markdown("##### 📁 Upload Your File")
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'flac'],
                help="Supported formats: WAV, MP3, M4A, FLAC"
            )
            
            if uploaded_file:
                st.audio(uploaded_file)
                
                try:
                    with st.spinner("📊 Analyzing file..."):
                        audio_data, duration, file_info = load_audio_file(uploaded_file)
                    
                    if audio_data is not None and file_info is not None:
                        # File information in a clean layout
                        st.markdown("##### 📊 File Details")
                        cols = st.columns(len(file_info))
                        for col, (key, value) in zip(cols, file_info.items()):
                            col.metric(key, value)
                        
                        # Processing options
                        st.markdown("##### ⚙️ Processing Options")
                        options_col1, options_col2 = st.columns(2)
                        
                        with options_col1:
                            chunk_size = st.slider(
                                "Segment Length",
                                min_value=10,
                                max_value=60,
                                value=30,
                                help="Length of each transcribed segment"
                            )
                        
                        with options_col2:
                            overlap = st.slider(
                                "Segment Overlap",
                                min_value=0,
                                max_value=min(10, chunk_size-1),
                                value=2,
                                help="Overlap between segments for better context"
                            )
                        
                        # Processing estimates
                        num_segments = math.ceil(duration / (chunk_size - overlap))
                        est_time = num_segments * 2
                        st.info(f"""
                        📊 **Estimated Processing:**
                        - Segments: {num_segments}
                        - Processing time: ~{format_timestamp(est_time)}
                        - Audio length: {format_timestamp(duration)}
                        """)
                        
                        # Process button
                        if st.button("🚀 Start Processing", use_container_width=True):
                            with st.spinner("🤖 Processing your audio..."):
                                transcriptions = transcribe_long_audio(
                                    audio_data,
                                    st.session_state['model'],
                                    st.session_state['processor'],
                                    duration,
                                    chunk_size
                                )
                            
                            # Download options
                            st.markdown("##### 💾 Download Options")
                            col1, col2 = st.columns(2)
                            with col1:
                                full_text = "\n\n".join(transcriptions)
                                st.download_button(
                                    "📄 Download TXT",
                                    full_text,
                                    file_name=f"{uploaded_file.name}_transcript.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            
                            with col2:
                                st.download_button(
                                    "🎬 Download SRT",
                                    create_srt(transcriptions),
                                    file_name=f"{uploaded_file.name}_transcript.srt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            
                            # Display transcriptions
                            st.markdown("##### 📝 Transcription")
                            for trans in transcriptions:
                                st.markdown(f'<div class="file-info">{trans}</div>', 
                                          unsafe_allow_html=True)
                                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with upload_col2:
            st.markdown("##### ℹ️ About File Upload")
            st.info("""
            **Supported Features:**
            - Long audio files (podcasts, interviews)
            - Multiple audio formats
            - Timestamped transcription
            - Download as TXT or SRT
            
            **Tips:**
            - For best results, use clear audio
            - Larger files will take longer to process
            - You can adjust segment length for better accuracy
            """)

# Add this new function to create SRT format
def create_srt(transcriptions):
    """Convert transcriptions to SRT format."""
    srt_lines = []
    for i, trans in enumerate(transcriptions, 1):
        # Extract timestamp and text
        timestamp_match = re.match(r'\[(.*?) → (.*?)\] (.*)', trans)
        if timestamp_match:
            start_time, end_time, text = timestamp_match.groups()
            # Convert timestamp format
            start_time = start_time.replace(':', ',')
            end_time = end_time.replace(':', ',')
            # Add leading zeros for milliseconds
            start_time += ',000'
            end_time += ',000'
            
            srt_lines.extend([
                str(i),
                f"{start_time} --> {end_time}",
                text,
                ""  # Empty line between entries
            ])
    
    return "\n".join(srt_lines)

# Standard Python idiom to ensure main() only runs if script is executed directly
if __name__ == "__main__":
    main()
