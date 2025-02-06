# Import necessary libraries
import streamlit as st  # Web application framework
import torch  # Deep learning framework
from transformers import WhisperProcessor, WhisperForConditionalGeneration  # Whisper model components
import numpy as np  # Numerical computing
import sounddevice as sd  # Audio recording
import soundfile as sf  # Audio file handling
import tempfile  # Temporary file operations
import time  # Time-related functions
import os  # Operating system interface
import warnings

# Ignore specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure Streamlit
st.set_page_config(page_title="Icelandic Speech Recognition", layout="wide")

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

def record_audio(duration, samplerate=16000):
    """Record audio from microphone."""
    try:
        st.write("🎤 Recording...")
        # Initialize the stream first
        sd.default.reset()
        sd.default.samplerate = samplerate
        sd.default.channels = 1
        
        audio_data = sd.rec(int(duration * samplerate),
                           samplerate=samplerate,
                           channels=1,
                           dtype=np.float32,
                           blocking=True)  # Use blocking=True instead of sd.wait()
        st.write("✅ Recording complete!")
        return audio_data.flatten()
    except Exception as e:
        st.error(f"Recording error: {str(e)}")
        raise

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

def main():
    """
    Main application function that sets up the UI and handles user interactions.
    
    The function:
    1. Displays the application title
    2. Loads the Whisper model if not already loaded
    3. Creates a two-column layout for controls and instructions
    4. Handles audio recording and transcription
    5. Displays system information in the sidebar
    """
    st.title("🎙️ Icelandic Speech Recognition")
    st.write("Record your voice and get real-time Icelandic transcription!")

    # Load model on first run
    if st.session_state['model'] is None:
        with st.spinner("Loading model... (this may take a few minutes)"):
            model, processor = load_model()
            st.session_state['model'] = model
            st.session_state['processor'] = processor
        st.success("Model loaded successfully!")

    # Recording interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recording Controls")
        duration = st.slider("Recording duration (seconds)", 1, 30, 5)
        
        if st.button("🎤 Start Recording"):
            try:
                audio_data = record_audio(duration)
                st.session_state['audio_data'] = audio_data
                
                # Save audio to temporary file for playback
                temp_audio_path = "temp_recording.wav"
                sf.write(temp_audio_path, audio_data, 16000)
                
                # Add audio playback
                st.audio(temp_audio_path)
                
                # Transcribe
                with st.spinner("Transcribing..."):
                    transcription = transcribe_audio(
                        audio_data,
                        st.session_state['model'],
                        st.session_state['processor']
                    )
                    
                # Display transcription
                st.subheader("Transcription:")
                st.write(transcription)
                
                # Clean up temporary file
                os.remove(temp_audio_path)
                
            except Exception as e:
                st.error(f"Error during recording/transcription: {str(e)}")
    
    with col2:
        st.subheader("Instructions")
        st.markdown("""
        1. Adjust the recording duration using the slider
        2. Click "Start Recording" and speak in Icelandic
        3. Wait for the transcription to appear
        4. You can play back your recording and see the transcription
        
        **Note:** Make sure your microphone is properly connected and permitted.
        """)


    # Display device information
    st.sidebar.subheader("System Information")
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.write(f"Using device: {device}")
    if torch.cuda.is_available():
        st.sidebar.write(f"GPU Model: {torch.cuda.get_device_name(0)}")

# Standard Python idiom to ensure main() only runs if script is executed directly
if __name__ == "__main__":
    main()
