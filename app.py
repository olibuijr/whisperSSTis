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
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("WhisperSST.is provides real-time Icelandic speech recognition using OpenAI's Whisper model, fine-tuned for Icelandic.")
    st.markdown("---")
    st.subheader("🔧 System Info")
    device_status = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    st.info(f"Using: {device_status}")
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")

st.title("Icelandic Speech Recognition")

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
                    opt_col1, opt_col2 = st.columns(2)
                    with opt_col1:
                        chunk_size = st.slider("Segment Length", min_value=10, max_value=60, value=30, help="Length of each transcribed segment")
                    with opt_col2:
                        overlap = st.slider("Segment Overlap", min_value=0, max_value=min(10, chunk_size-1), value=2, help="Overlap between segments for better context")
                    
                    num_segments = math.ceil(duration / (chunk_size - overlap)) if (chunk_size - overlap) else 1
                    est_processing_time = num_segments * 2  # Rough estimate in seconds
                    
                    st.info(f"""
                    📊 **Estimated Processing:**
                    - Segments: {num_segments}
                    - Processing time: ~{transcribe.format_timestamp(est_processing_time)}
                    - Audio length: {transcribe.format_timestamp(duration)}
                    """)
                    
                    if st.button("🚀 Start Processing", use_container_width=True):
                        with st.spinner("🤖 Processing your audio..."):
                            transcriptions = transcribe.transcribe_long_audio(audio_data, st.session_state['model'], st.session_state['processor'], duration, chunk_size)
                        
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
    st.markdown("### Live Transcription")
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
        
        # Initialize live transcription session state variables
        if "live_running" not in st.session_state:
            st.session_state.live_running = False
        if "live_transcript" not in st.session_state:
            st.session_state.live_transcript = ""
        
        # Display a textbox to show the live transcript
        transcript_box = st.text_area("Live Transcript", st.session_state.live_transcript, height=300, key="live_transcript_box")
        
        # Button to start live transcription
        if st.button("Start Live Transcription", key="start_live"):
            st.session_state.live_running = True
            st.session_state.live_transcript = ""
        
        # Button to stop live transcription
        if st.session_state.live_running and st.button("Stop Live Transcription", key="stop_live"):
            st.session_state.live_running = False
        
        if st.session_state.live_running:
            try:
                # Record a chunk of audio (this blocks for 'live_chunk_length' seconds)
                chunk_audio = audio.record_audio(live_chunk_length, live_device_id)
                chunk_transcript = transcribe.transcribe_audio(chunk_audio,
                                                               st.session_state['model'],
                                                               st.session_state['processor'])
                st.session_state.live_transcript += " " + chunk_transcript
            except Exception as e:
                st.error(f"❌ Error during live transcription: {str(e)}")
            st.rerun()

st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb=\"tab-list\"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb=\"tab\"] {
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