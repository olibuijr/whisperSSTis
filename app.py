import streamlit as st
import os
import math
import torch
import soundfile as sf

from whisperSSTis import audio, transcribe

st.set_page_config(
    page_title="WhisperSST.is - Icelandic Speech Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/ce/Whisper_%28software%29_logo.svg", width=100)
    st.title("WhisperSST.is")
    st.caption("🧪 Alpha Version")
    st.markdown("---")
    st.subheader("i️ About")
    st.markdown("WhisperSST.is provides real-time Icelandic speech recognition using OpenAI's Whisper model, fine-tuned for Icelandic.")
    st.markdown("Developed by **Magnus Smari Smarason**")
    st.markdown("---")
    st.subheader("🏆 Credits")
    st.markdown("""
    - **Original Whisper Model**: [OpenAI](https://github.com/openai/whisper)
    - **Icelandic Fine-tuned Model**: [Carlos Daniel Hernandez Mena](https://huggingface.co/carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h)
    - Built with [Streamlit](https://streamlit.io/), [PyTorch](https://pytorch.org/), and [Hugging Face](https://huggingface.co/)
    """)
    st.markdown("---")
    st.subheader("🔧 System Info")
    device_status = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    st.info(f"Using: {device_status}")
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")

st.title("Icelandic Speech Recognition 🎙️")
st.caption("Powered by fine-tuned Whisper AI for the Icelandic language")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'processor' not in st.session_state:
    st.session_state['processor'] = None
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None

if st.session_state['model'] is None:
    with st.spinner("🤖 Loading AI model... This might take a minute..."):
        model, processor = transcribe.load_model()
        st.session_state['model'] = model
        st.session_state['processor'] = processor
    st.success("✅ Model loaded successfully!")

# Create tabs for recording and uploading
tab1, tab2, tab3 = st.tabs(["🎤 Record Audio", "📁 Upload Audio", "💬 Live Transcription"])

with tab1:
    st.markdown("""
    <div class="upload-header">
    <h3>Record Your Voice</h3>
    <p>Speak in Icelandic and get instant transcription.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3,2])
    
    with col1:
        with st.container():
            st.markdown("##### Device Settings")
            input_devices = audio.get_audio_devices()
            if not input_devices:
                st.error("❌ No input devices found!")
            else:
                device_options = list(input_devices.keys())
                device_choice = st.selectbox("🎙️ Select Input Device", options=device_options)
                selected_device_id = input_devices[device_choice]
                rec_duration = st.slider("⏱️ Recording Duration (seconds)", min_value=1, max_value=30, value=5, help="Choose how long you want to record")
                if st.button("🎙️ Start Recording", use_container_width=True):
                    try:
                        with st.spinner("🎤 Recording in progress..."):
                            audio_data = audio.record_audio(rec_duration, selected_device_id)
                        st.session_state['audio_data'] = audio_data
                        
                        temp_path = "temp_recording.wav"
                        sf.write(temp_path, audio_data, 16000)
                        st.audio(temp_path)
                        
                        with st.spinner("🤖 Processing your speech..."):
                            transcription = transcribe.transcribe_audio(audio_data, st.session_state['model'], st.session_state['processor'])
                        st.markdown("##### 📝 Transcription")
                        st.markdown(f'<div class="file-info">{transcription}</div>', unsafe_allow_html=True)
                        os.remove(temp_path)
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown("##### 📋 Instructions")
        st.info("""
        1. Select your microphone from the dropdown.
        2. Set your preferred recording duration.
        3. Click \"Start Recording\" and speak in Icelandic.
        4. Wait for the transcription to appear.
        
        **Note:** Ensure your microphone is properly connected and permitted.
        """)

with tab2:
    st.markdown("""
    <div class="upload-header">
    <h3>Upload Audio File</h3>
    <p>Process longer recordings like podcasts or interviews.</p>
    </div>
    """, unsafe_allow_html=True)
    
    upload_col1, upload_col2 = st.columns([3,2])
    
    with upload_col1:
        st.markdown("##### 📁 Upload Your File")
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a', 'flac'], help="Supported formats: WAV, MP3, M4A, FLAC")
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            try:
                with st.spinner("📊 Analyzing file..."):
                    audio_data, duration, file_info = audio.load_audio_file(uploaded_file)
                if audio_data is not None and file_info is not None:
                    st.markdown("##### 📊 File Details")
                    detail_cols = st.columns(len(file_info))
                    for col, (key, value) in zip(detail_cols, file_info.items()):
                        col.metric(key, value)
                    
                    st.markdown("##### ⚙️ Processing Options")
                    chunk_size = st.slider("Segment Length", min_value=10, max_value=60, value=30, 
                                         help="Length of each transcribed segment. Shorter segments may improve accuracy but take longer to process.")
                    
                    # Calculate number of segments
                    num_segments = math.ceil(duration / chunk_size)
                    est_processing_time = num_segments * 2  # Rough estimate in seconds
                    
                    st.info(f"""
                    📊 **Estimated Processing:**
                    - Segments: {num_segments}
                    - Processing time: ~{transcribe.format_timestamp(est_processing_time)}
                    - Audio length: {transcribe.format_timestamp(duration)}
                    """)
                    
                    if st.button("🚀 Start Processing", use_container_width=True):
                        with st.spinner("🤖 Processing your audio..."):
                            transcriptions = transcribe.transcribe_long_audio(
                                audio_data,
                                st.session_state['model'],
                                st.session_state['processor'],
                                duration,
                                chunk_size
                            )
                        
                        st.markdown("##### 💾 Download Options")
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            full_text = "\n\n".join(transcriptions)
                            st.download_button("📄 Download TXT", full_text, file_name=f"{uploaded_file.name}_transcript.txt", mime="text/plain", use_container_width=True)
                        with dl_col2:
                            srt_content = transcribe.create_srt(transcriptions)
                            st.download_button("🎬 Download SRT", srt_content, file_name=f"{uploaded_file.name}_transcript.srt", mime="text/plain", use_container_width=True)
                        
                        st.markdown("##### 📝 Transcription")
                        for trans in transcriptions:
                            st.markdown(f'<div class="file-info">{trans}</div>', unsafe_allow_html=True)
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
        - For best results, use clear audio.
        - Larger files will take longer to process.
        - You can adjust segment length for better accuracy.
        """)

with tab3:
    st.markdown("""
    <div class="upload-header">
    <h3>💬 Live Transcription</h3>
    <p>Real-time speech-to-text conversion as you speak.</p>
    </div>
    """, unsafe_allow_html=True)
    live_devices = audio.get_audio_devices()
    if not live_devices:
        st.error("❌ No input devices available for live transcription.")
    else:
        live_device_choice = st.selectbox("🎙️ Select Input Device for Live Transcription",
                                            options=list(live_devices.keys()),
                                            key="live_device")
        live_device_id = live_devices[live_device_choice]
        live_chunk_length = st.slider("Chunk Length (seconds)", 
                                      min_value=1, max_value=10, value=3, key="live_chunk")
        
        # Initialize session state variables for live transcription
        for key in ['live_running', 'live_transcript', 'audio_stream', 'processing', 'error_count']:
            if key not in st.session_state:
                st.session_state[key] = False if key in ['live_running', 'processing'] else "" if key == 'live_transcript' else None if key == 'audio_stream' else 0

        # Create columns for controls and status
        col1, col2, col3 = st.columns([1,1,2])
        
        with col1:
            if not st.session_state.live_running:
                if st.button("🎙️ Start", key="start_live", use_container_width=True):
                    st.session_state.live_running = True
                    st.session_state.live_transcript = ""
                    st.session_state.error_count = 0
                    # Initialize audio stream
                    st.session_state.audio_stream = audio.AudioStream(
                        device_id=live_device_id,
                        samplerate=16000,
                        chunk_size=int(live_chunk_length * 16000)
                    )
                    st.session_state.audio_stream.start_stream()
        
        with col2:
            if st.session_state.live_running:
                if st.button("⏹️ Stop", key="stop_live", use_container_width=True):
                    if st.session_state.audio_stream:
                        st.session_state.audio_stream.stop_stream()
                    st.session_state.live_running = False
                    st.session_state.audio_stream = None

        with col3:
            if st.session_state.live_running and st.session_state.audio_stream:
                # Get and display audio level
                level = st.session_state.audio_stream.get_audio_level()
                level_normalized = min(1.0, level * 5)  # Scale for better visualization
                level_bars = int(level_normalized * 20)
                level_display = "▮" * level_bars + "▯" * (20 - level_bars)
                st.markdown(f"🔴 Recording... Level: {level_display}")
            elif st.session_state.processing:
                st.markdown("⏳ Processing...")

        # Display transcription area with improved styling
        st.markdown("##### 📝 Live Transcript")
        transcript_container = st.container()
        with transcript_container:
            st.markdown(
                f'<div class="live-transcript">{st.session_state.live_transcript}</div>',
                unsafe_allow_html=True
            )

        # Live transcription logic
        if st.session_state.live_running and st.session_state.audio_stream:
            try:
                # Get audio chunk from stream
                chunk = st.session_state.audio_stream.get_audio_chunk()
                if chunk is not None and len(chunk) > 0:
                    st.session_state.processing = True
                    # Process the chunk
                    chunk_transcript = transcribe.transcribe_audio(
                        chunk,
                        st.session_state['model'],
                        st.session_state['processor']
                    )
                    if chunk_transcript.strip():
                        st.session_state.live_transcript += " " + chunk_transcript
                    st.session_state.processing = False
                    st.session_state.error_count = 0
                
            except Exception as e:
                st.session_state.error_count += 1
                if st.session_state.error_count > 3:
                    st.error("❌ Multiple errors occurred. Stopping live transcription.")
                    if st.session_state.audio_stream:
                        st.session_state.audio_stream.stop_stream()
                    st.session_state.live_running = False
                    st.session_state.audio_stream = None
                else:
                    st.warning(f"⚠️ Error during live transcription: {str(e)}")
            finally:
                st.rerun()

# Add custom styles
st.markdown("""
    <style>
    .live-transcript {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        min-height: 200px;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
        font-size: 1.1em;
        line-height: 1.6;
    }
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
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .css-1dp5vir {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .css-1dp5vir:hover {
        box-shadow: 0 3px 6px rgba(0,0,0,0.16);
        transition: all 0.2s ease;
    }
    </style>
""", unsafe_allow_html=True)
