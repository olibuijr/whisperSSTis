import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import webbrowser
import torch
import sounddevice as sd
import numpy as np
import threading
from datetime import datetime
import tempfile
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scipy import signal
import logging  # Add at the top with other imports
import queue
import wave
import yt_dlp
import time
import psutil  # Add to imports at top
import pyttsx3

# Add after imports, before class definition
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class WhisperSSTLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WhisperSST.is")
        self.root.geometry("1280x1024")
        
        # Initialize variables first
        self.model = None
        self.processor = None
        self.recording = False
        self.audio_stream = None
        self.current_text = ""
        self.audio_buffer = []
        self.buffer_duration = 3
        self.target_sample_rate = 16000  # What Whisper expects
        self.device_sample_rate = 48000  # Default device rate
        self.full_recording = []  # Add this to store the complete recording
        
        # Initialize device variables before setup_record_tab
        self.devices, self.default_device = self.get_audio_devices()
        self.device_var = tk.StringVar(self.root)  # Initialize with root
        self.device_var.set(self.default_device)  # Set initial value
        self.device_menu = None
        
        # Add these new variables
        self.last_recording = None  # To store the last recording
        self.is_playing = False
        self.playback_thread = None
        
        # Initialize TTS
        try:
            self.tts_engine = pyttsx3.init()
            # Get all available voices
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find Icelandic voice
            icelandic_voice = None
            for voice in voices:
                # Check for Icelandic identifiers in voice name or ID
                if any(identifier in voice.name.lower() for identifier in ['icelandic', 'íslenska', 'is-is', 'islensku']):
                    icelandic_voice = voice
                    break
            
            # Set Icelandic voice if found, otherwise use default
            if icelandic_voice:
                self.tts_engine.setProperty('voice', icelandic_voice.id)
                logging.info(f"Using Icelandic voice: {icelandic_voice.name}")
            else:
                logging.warning("No Icelandic voice found, using default voice")
            
            # Set initial properties
            self.tts_engine.setProperty('rate', 150)  # Speed
            self.tts_engine.setProperty('volume', 1.0)  # Volume
            self.tts_ready = True
            
        except Exception as e:
            logging.error(f"Failed to load TTS: {e}")
            self.tts_ready = False
        
        # Rest of initialization
        try:
            self.root.iconbitmap("whisper_icon.ico")
        except:
            pass
        
        style = ttk.Style()
        style.configure("TButton", padding=10)
        
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="WhisperSST.is",
            font=("Helvetica", 20, "bold")
        )
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = ttk.Label(
            self.main_frame,
            text="Icelandic Speech Recognition",
            font=("Helvetica", 12)
        )
        subtitle_label.pack(pady=5)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs
        self.record_tab = ttk.Frame(self.notebook)
        self.upload_tab = ttk.Frame(self.notebook)
        self.youtube_tab = ttk.Frame(self.notebook)  # Add new tab
        
        self.notebook.add(self.record_tab, text="🎤 Record Audio")
        self.notebook.add(self.upload_tab, text="📁 Upload Audio")
        self.notebook.add(self.youtube_tab, text="▶️ YouTube")  # Add new tab
        
        # Setup recording tab
        self.setup_record_tab()
        
        # Setup upload tab
        self.setup_upload_tab()
        
        # Setup YouTube tab
        self.setup_youtube_tab()  # Add new setup
        
        # Status bar with system info
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Device info
        device_info = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
        if torch.cuda.is_available():
            device_info += f" - {torch.cuda.get_device_name(0)}"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
            device_info += f" ({gpu_memory:.1f}GB)"
        
        # System memory
        system_memory = psutil.virtual_memory().total / (1024**3)  # Convert to GB
        device_info += f" | 💾 RAM: {system_memory:.1f}GB"
        
        self.device_label = ttk.Label(
            status_frame,
            text=device_info,
            font=("Helvetica", 9)
        )
        self.device_label.pack(side=tk.LEFT, padx=10)
        
        # Add memory usage monitoring
        self.memory_label = ttk.Label(
            status_frame,
            text="",
            font=("Helvetica", 9)
        )
        self.memory_label.pack(side=tk.LEFT, padx=10)
        
        # Status label (for model loading etc)
        self.status_label = ttk.Label(
            status_frame,
            text="Loading model...",
            font=("Helvetica", 9)
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Start memory monitoring
        self.update_memory_usage()
        
        # Progress bar for model loading
        self.progress = ttk.Progressbar(
            self.main_frame,
            mode='indeterminate',
            length=300
        )
        
        # Show progress bar immediately
        self.progress.pack(pady=5)
        self.progress.start()
        
        # Load model in background
        self.load_model_async()

    def setup_record_tab(self):
        """Setup the recording tab interface."""
        # Device selection
        device_frame = ttk.LabelFrame(self.record_tab, text="Audio Device", padding=10)
        device_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create device menu only if we have devices
        if self.devices:
            self.device_menu = ttk.OptionMenu(
                device_frame, 
                self.device_var,
                self.default_device, 
                *self.devices.keys()
            )
            self.device_menu.pack(fill=tk.X)
        else:
            # Show error if no devices found
            ttk.Label(device_frame, text="No audio input devices found").pack()
        
        # Recording controls
        control_frame = ttk.Frame(self.record_tab, padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Use tk.Button instead of ttk.Button for recording
        self.record_btn = tk.Button(
            control_frame,
            text="Start Recording",
            command=self.toggle_recording,
            state="disabled",  # Start disabled
            relief="raised",
            bg="#f0f0f0",  # Light gray - default button color
            padx=10,
            pady=5
        )
        self.record_btn.pack(pady=10)
        
        # Add playback controls
        playback_frame = ttk.Frame(self.record_tab, padding=10)
        playback_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.playback_btn = tk.Button(
            playback_frame,
            text="Play Last Recording",
            command=self.toggle_playback,
            state="disabled",
            relief="raised",
            bg="#f0f0f0",
            padx=10,
            pady=5
        )
        self.playback_btn.pack(pady=5)
        
        # Transcription display
        self.transcription_text = tk.Text(
            self.record_tab,
            height=10,
            wrap=tk.WORD,
            font=("Helvetica", 11)
        )
        self.transcription_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add TTS controls
        tts_frame = ttk.LabelFrame(self.record_tab, text="Text-to-Speech", padding=10)
        tts_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add controls
        self.setup_tts_controls(tts_frame)
        
        # Speak button
        self.tts_btn = ttk.Button(
            tts_frame,
            text="🔊 Speak Text",
            command=self.speak_text,
            state="disabled" if not self.tts_ready else "normal"
        )
        self.tts_btn.pack(pady=5)

    def setup_upload_tab(self):
        """Setup the file upload tab interface."""
        upload_frame = ttk.Frame(self.upload_tab, padding=10)
        upload_frame.pack(fill=tk.BOTH, expand=True)
        
        upload_btn = ttk.Button(
            upload_frame,
            text="Choose Audio File",
            command=self.choose_file
        )
        upload_btn.pack(pady=10)
        
        self.file_label = ttk.Label(
            upload_frame,
            text="No file selected",
            font=("Helvetica", 9)
        )
        self.file_label.pack(pady=5)
        
        self.upload_text = tk.Text(
            upload_frame,
            height=10,
            wrap=tk.WORD,
            font=("Helvetica", 11)
        )
        self.upload_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def setup_youtube_tab(self):
        """Setup the YouTube tab interface."""
        # Main container
        youtube_frame = ttk.Frame(self.youtube_tab, padding=10)
        youtube_frame.pack(fill=tk.BOTH, expand=True)
        
        # URL input
        url_frame = ttk.Frame(youtube_frame)
        url_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(url_frame, text="YouTube URL:").pack(side=tk.LEFT, padx=5)
        
        self.url_var = tk.StringVar()
        url_entry = ttk.Entry(url_frame, textvariable=self.url_var, width=50)
        url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Video title label
        self.video_title_label = ttk.Label(
            youtube_frame,
            text="",
            font=("Helvetica", 10, "italic"),
            wraplength=500  # Wrap long titles
        )
        self.video_title_label.pack(pady=5)
        
        # Process button
        self.youtube_btn = ttk.Button(
            youtube_frame,
            text="Process Video",
            command=self.process_youtube
        )
        self.youtube_btn.pack(pady=10)
        
        # Progress frame
        self.progress_frame = ttk.LabelFrame(youtube_frame, text="Progress", padding=10)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        # Download progress
        ttk.Label(self.progress_frame, text="Download:").pack(anchor=tk.W)
        self.download_progress = ttk.Progressbar(
            self.progress_frame,
            mode='determinate',
            length=300
        )
        self.download_progress.pack(fill=tk.X, pady=2)
        
        # Processing progress
        ttk.Label(self.progress_frame, text="Processing:").pack(anchor=tk.W)
        self.processing_progress = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            length=300
        )
        self.processing_progress.pack(fill=tk.X, pady=2)
        
        # Status label
        self.youtube_status = ttk.Label(
            youtube_frame,
            text="Ready",
            font=("Helvetica", 9)
        )
        self.youtube_status.pack(pady=5)
        
        # Transcription display
        self.youtube_text = tk.Text(
            youtube_frame,
            height=10,
            wrap=tk.WORD,
            font=("Helvetica", 11)
        )
        self.youtube_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add TTS button to YouTube tab
        tts_frame = ttk.LabelFrame(youtube_frame, text="Text-to-Speech", padding=10)
        tts_frame.pack(fill=tk.X, pady=5)
        
        # Add controls
        self.setup_tts_controls(tts_frame)
        
        self.youtube_tts_btn = ttk.Button(
            tts_frame,
            text="🔊 Speak Text",
            command=lambda: self.speak_text(source='youtube'),
            state="disabled" if not self.tts_ready else "normal"
        )
        self.youtube_tts_btn.pack(pady=5)

    def get_audio_devices(self):
        """Get available audio input devices."""
        devices = {}
        try:
            # Get default input device
            default_device = sd.default.device[0]
            logging.info(f"System default input device index: {default_device}")
            
            # Get all devices first
            all_devices = sd.query_devices()
            logging.info(f"All available devices: {all_devices}")
            
            # Get default device info
            default_info = all_devices[default_device]
            logging.info(f"Default device info: {default_info}")
            
            # Create consistent key for default device
            default_name = default_info.get('name', 'default')
            default_key = f"Default - {default_name} (ID: {default_device})"
            devices[default_key] = default_device
            logging.info(f"Added default device with key: {default_key}")
            
            # Add other input devices
            for i, dev in enumerate(all_devices):
                if dev['max_input_channels'] > 0 and i != default_device:
                    name = f"{dev['name']} (ID: {i})"
                    devices[name] = i
                    logging.info(f"Added additional device: {name}")
            
            logging.info(f"Final devices dictionary: {devices}")
            logging.info(f"Default key: {default_key}")
            
        except Exception as e:
            logging.error(f"Error getting audio devices: {e}", exc_info=True)
            default_key = "Default System Device"
            devices[default_key] = None
            
        return devices, default_key

    def load_model_async(self):
        """Load the Whisper model in a background thread."""
        def load():
            try:
                model_name = "carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h"
                self.processor = WhisperProcessor.from_pretrained(model_name)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
                
                self.root.after(0, self.model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Failed to load model: {str(e)}"))
        
        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()

    def model_loaded(self):
        """Called when model is loaded successfully."""
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.config(text="Model loaded - Ready")
        self.record_btn.config(state="normal")  # Enable button when model is loaded

    def toggle_recording(self):
        """Toggle audio recording on/off."""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start audio recording."""
        if not self.model:
            self.show_error("Please wait for the model to load")
            return
            
        if not self.devices:
            self.show_error("No audio input devices available")
            return
            
        try:
            selected_device = self.device_var.get()
            if not selected_device:
                self.show_error("No audio device selected")
                return
                
            logging.info(f"Selected device: {selected_device}")
            logging.info(f"Available devices: {self.devices}")
            
            if selected_device not in self.devices:
                logging.error(f"Selected device '{selected_device}' not found in devices dictionary")
                raise KeyError(f"Device '{selected_device}' not found")
                
            device_id = self.devices[selected_device]
            logging.info(f"Device ID: {device_id}")
            
            if device_id is None:
                device_id = sd.default.device[0]
                logging.info(f"Using system default device ID: {device_id}")
            
            # Log device details
            device_info = sd.query_devices(device_id)
            logging.info(f"Device details: {device_info}")
            
            self.recording = True
            self.full_recording = []  # Reset full recording buffer
            self.record_btn.config(
                text="Stop Recording",
                bg="#ff0000",
                activebackground="#cc0000"
            )
            self.transcription_text.delete(1.0, tk.END)
            self.current_text = ""
            
            logging.info("Starting recording thread")
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.daemon = True
            self.record_thread.start()
            
        except Exception as e:
            logging.error(f"Failed to start recording: {e}", exc_info=True)
            self.show_error(f"Failed to start recording: {str(e)}")

    def stop_recording(self):
        """Stop audio recording and process the complete recording."""
        self.recording = False
        self.record_btn.config(
            text="Processing...",
            state="disabled"
        )
        
        # Process the complete recording
        if self.audio_buffer:
            try:
                # Concatenate all audio data
                audio_data = np.concatenate(self.audio_buffer)
                
                # Store for playback
                self.last_recording = {
                    'data': audio_data.copy(),
                    'sample_rate': self.device_sample_rate
                }
                
                # Enable playback button
                self.playback_btn.config(state="normal")
                
                # Prepare audio for processing
                audio_data = audio_data.flatten()
                audio_data = audio_data.astype(np.float32)
                
                # Resample if needed
                if self.device_sample_rate != self.target_sample_rate:
                    logging.info(f"Resampling from {self.device_sample_rate} to {self.target_sample_rate}")
                    audio_data = self.resample_audio(
                        audio_data, 
                        self.device_sample_rate, 
                        self.target_sample_rate
                    )
                
                # Process in separate thread
                process_thread = threading.Thread(
                    target=self.process_audio,
                    args=(audio_data,)
                )
                process_thread.daemon = True
                process_thread.start()
                
            except Exception as e:
                logging.error(f"Error processing recording: {e}")
                self.show_error("Error processing recording")
                
            finally:
                # Clear the buffer
                self.audio_buffer = []
        
        # Reset button state
        self.record_btn.config(
            text="Start Recording",
            bg="#f0f0f0",
            activebackground="#e0e0e0",
            state="normal"
        )

    def record_audio(self):
        """Record and process audio in real-time."""
        try:
            selected_device = self.device_var.get()
            logging.info(f"Recording thread - Selected device: {selected_device}")
            
            device_id = self.devices[selected_device]
            logging.info(f"Recording thread - Device ID: {device_id}")
            
            if device_id is None:
                device_id = sd.default.device[0]
                logging.info(f"Recording thread - Using system default device ID: {device_id}")
            
            # Get device info and its supported sample rate
            device_info = sd.query_devices(device_id)
            self.device_sample_rate = int(device_info['default_samplerate'])
            logging.info(f"Using device sample rate: {self.device_sample_rate}")
            
            self.audio_buffer = []
            logging.info(f"Starting InputStream with device {device_id}, sample rate {self.device_sample_rate}")
            
            with sd.InputStream(
                device=device_id, 
                channels=1,
                samplerate=self.device_sample_rate,  # Use device's native rate
                blocksize=int(self.device_sample_rate * 0.5),  # 0.5 second chunks
                callback=self.audio_callback
            ) as stream:
                logging.info("Audio stream started successfully")
                while self.recording:
                    sd.sleep(100)
                    
        except Exception as e:
            error_msg = str(e)  # Capture error message
            self.root.after(0, lambda: self.show_error(f"Recording error: {error_msg}"))

    def audio_callback(self, indata, frames, time, status):
        """Process recorded audio chunks."""
        if status:
            logging.warning(f"Audio callback status: {status}")
            
        if self.recording:
            try:
                # Only store the audio data, don't process
                self.audio_buffer.append(indata.copy())
                    
            except Exception as e:
                logging.error(f"Error in audio callback: {e}")
                self.recording = False

    def process_audio(self, audio_data):
        """Process audio data and update transcription."""
        try:
            logging.info("Starting audio processing")
            self.status_label.config(text="Processing audio...")
            
            # Ensure audio data is in the right format
            audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            logging.info(f"Processing audio of shape: {audio_data.shape}")
            
            # Convert audio to features
            input_features = self.processor(
                audio_data, 
                sampling_rate=self.target_sample_rate,
                return_tensors="pt"
            ).input_features.to(self.model.device)
            
            # Generate transcription with Icelandic language
            predicted_ids = self.model.generate(
                input_features,
                language="<|is|>",
                task="transcribe"
            )
            
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            logging.info(f"Transcription result: {transcription}")
            
            # Update UI in main thread
            if transcription.strip():
                self.root.after(0, lambda: self.update_transcription(transcription))
            
            self.root.after(0, lambda: self.status_label.config(text="Ready"))
            
        except Exception as e:
            logging.error(f"Error in process_audio: {e}", exc_info=True)
            self.root.after(0, lambda: self.status_label.config(text="Error processing audio"))
            self.root.after(0, lambda: self.show_error(f"Processing error: {str(e)}"))

    def process_text_formatting(self, text):
        """Process text to add proper capitalization and periods."""
        # Initial cleanup
        text = text.strip()
        if not text:
            return text
        
        # Common Icelandic abbreviations to preserve
        abbreviations = ['hr.', 'dr.', 'pr.', 'sr.', 't.d.', 'o.s.frv.', 'þ.e.', 'þ.e.a.s.']
        
        # Split into potential sentences (keeping abbreviations intact)
        raw_sentences = []
        current = []
        words = text.split()
        
        for i, word in enumerate(words):
            current.append(word)
            
            # Check if this word ends a sentence
            ends_sentence = False
            
            # Check if word is not an abbreviation
            if word not in abbreviations:
                # Check for sentence endings
                if (i == len(words) - 1 or  # Last word
                    word.endswith(('.', '!', '?')) or  # Has ending punctuation
                    (len(word) > 1 and not word[-1].isalnum() and not word.endswith('.')) or  # Non-period punctuation
                    # Next word starts with capital (unless it's "I" or "É")
                    (i < len(words) - 1 and words[i + 1][0].isupper() and 
                     not words[i + 1].startswith(('I ', 'É'))) or
                    # Long pause indicated by multiple spaces or punctuation
                    (i < len(words) - 1 and (word.endswith(',') or word.endswith(';')))
                ):
                    ends_sentence = True
            
            if ends_sentence:
                sentence = ' '.join(current)
                # Add period if no ending punctuation
                if not sentence[-1] in '.!?':
                    sentence += '.'
                raw_sentences.append(sentence)
                current = []
        
        # Handle any remaining words
        if current:
            sentence = ' '.join(current)
            if not sentence[-1] in '.!?':
                sentence += '.'
            raw_sentences.append(sentence)
        
        # Process each sentence
        processed_sentences = []
        for sentence in raw_sentences:
            # Capitalize first letter if it's a letter
            if sentence and sentence[0].isalpha():
                sentence = sentence[0].upper() + sentence[1:]
            
            # Fix spacing around punctuation
            sentence = sentence.replace(' ,', ',')
            sentence = sentence.replace(' .', '.')
            sentence = sentence.replace(' !', '!')
            sentence = sentence.replace(' ?', '?')
            sentence = sentence.replace(' :', ':')
            sentence = sentence.replace(' ;', ';')
            
            # Add proper spacing after punctuation
            for punct in ['.', ',', '!', '?', ':', ';']:
                sentence = sentence.replace(f"{punct}", f"{punct} ")
            
            # Clean up multiple spaces
            sentence = ' '.join(sentence.split())
            
            processed_sentences.append(sentence)
        
        # Join sentences with proper spacing
        final_text = ' '.join(processed_sentences)
        
        # Final cleanup of any double spaces
        final_text = ' '.join(final_text.split())
        
        return final_text

    def update_transcription(self, text):
        """Update the transcription display smoothly."""
        logging.info(f"Updating transcription with: {text}")
        
        # Process text formatting
        text = self.process_text_formatting(text)
        
        # Clean up the text
        text = text.strip()
        
        # If this is a continuation of previous text, append it
        if text.lower().startswith(self.current_text.lower()):
            new_part = text[len(self.current_text):].strip()
            if new_part:
                self.transcription_text.insert(tk.END, new_part + " ")
                self.current_text = text
        else:
            # If it's new text, add it on a new line
            if self.current_text:  # If there was previous text
                self.transcription_text.insert(tk.END, "\n")
            self.transcription_text.insert(tk.END, text + " ")
            self.current_text = text
        
        self.transcription_text.see(tk.END)

    def choose_file(self):
        """Open file chooser dialog."""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.m4a *.flac"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path):
        """Process an audio file."""
        if not self.model:
            self.show_error("Please wait for the model to load")
            return
            
        self.file_label.config(text=os.path.basename(file_path))
        self.status_label.config(text="Processing audio file...")
        self.progress.pack(pady=5)
        self.progress.start()
        
        def process():
            try:
                # Load and process audio file
                audio_data, sr = sf.read(file_path)
                if sr != 16000:
                    # Resample if needed
                    audio_data = self.resample_audio(audio_data, sr, 16000)
                
                # Process audio
                input_features = self.processor(
                    audio_data, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(self.model.device)  # Move to same device as model
                
                # Generate transcription with Icelandic language
                predicted_ids = self.model.generate(
                    input_features,
                    language="<|is|>",  # Specify Icelandic
                    task="transcribe"
                )
                transcription = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
                
                # Update UI in main thread
                self.root.after(0, lambda: self.file_processed(transcription))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.show_error(f"Failed to process file: {error_msg}"))
        
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()

    def file_processed(self, transcription):
        """Called when file processing is complete."""
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.config(text="File processed")
        
        # Process text formatting
        formatted_text = self.process_text_formatting(transcription)
        
        # Update text widget
        self.upload_text.delete(1.0, tk.END)
        self.upload_text.insert(tk.END, formatted_text)
        self.current_text = ""  # Reset current text

    def resample_audio(self, audio_data, orig_sr, target_sr):
        """Resample audio to target sample rate."""
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        output_length = int(len(audio_data) * ratio)
        
        # Use scipy's resample function
        return signal.resample(audio_data, output_length)

    def show_error(self, message):
        """Show error message."""
        messagebox.showerror("Error", message)
        self.status_label.config(text="Error occurred")
        self.progress.stop()
        self.progress.pack_forget()

    def on_device_change(self, *args):
        """Handle device selection changes."""
        logging.info(f"Device changed to: {self.device_var.get()}")
        logging.info(f"Current devices: {self.devices}")

    def refresh_devices(self):
        """Refresh the list of audio devices."""
        try:
            # Get updated devices
            self.devices, new_default = self.get_audio_devices()
            
            if self.device_menu is None:
                logging.warning("Device menu not initialized")
                return
                
            # Update the OptionMenu
            menu = self.device_menu['menu']
            menu.delete(0, 'end')
            
            for device in self.devices.keys():
                menu.add_command(
                    label=device,
                    command=lambda d=device: self.device_var.set(d)
                )
            
            # Set to default if current selection is not available
            current = self.device_var.get()
            if current not in self.devices:
                self.device_var.set(new_default)
                
            logging.info(f"Devices refreshed: {self.devices}")
            
        except Exception as e:
            logging.error(f"Error refreshing devices: {e}", exc_info=True)

    def toggle_playback(self):
        """Toggle playback of last recording."""
        if not self.last_recording:
            return
            
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """Start playing the last recording."""
        if not self.last_recording:
            return
            
        try:
            self.is_playing = True
            self.playback_btn.config(
                text="Stop Playback",
                bg="#4CAF50",  # Green
                activebackground="#45a049"
            )
            
            # Start playback in a separate thread
            self.playback_thread = threading.Thread(target=self.play_audio)
            self.playback_thread.daemon = True
            self.playback_thread.start()
            
        except Exception as e:
            logging.error(f"Error starting playback: {e}")
            self.show_error("Failed to start playback")
            self.stop_playback()

    def stop_playback(self):
        """Stop the audio playback."""
        self.is_playing = False
        self.playback_btn.config(
            text="Play Last Recording",
            bg="#f0f0f0",
            activebackground="#e0e0e0"
        )

    def play_audio(self):
        """Play the audio data."""
        try:
            # Create a stream for playback
            with sd.OutputStream(
                channels=1,
                samplerate=self.last_recording['sample_rate'],
                callback=self.playback_callback
            ) as stream:
                while self.is_playing:
                    sd.sleep(100)
                    
        except Exception as e:
            logging.error(f"Playback error: {e}")
            self.root.after(0, self.stop_playback)

    def playback_callback(self, outdata, frames, time, status):
        """Callback for audio playback."""
        if status:
            logging.warning(f"Playback status: {status}")
            
        try:
            if self.is_playing and self.last_recording:
                # Get the data to play
                data = self.last_recording['data']
                if len(data) > 0:
                    # Copy data to output buffer
                    if len(data) >= len(outdata):
                        outdata[:] = data[:len(outdata)]
                        self.last_recording['data'] = data[len(outdata):]
                    else:
                        outdata[:len(data)] = data
                        outdata[len(data):] = 0
                        self.last_recording['data'] = np.array([])
                        # Stop when we're done
                        self.root.after(0, self.stop_playback)
        except Exception as e:
            logging.error(f"Error in playback callback: {e}")
            self.root.after(0, self.stop_playback)

    def process_youtube(self):
        """Process YouTube video URL."""
        url = self.url_var.get().strip()
        if not url:
            self.show_error("Please enter a YouTube URL")
            return
            
        if not self.model:
            self.show_error("Please wait for the model to load")
            return
        
        # Disable button while processing
        self.youtube_btn.config(state="disabled")
        self.youtube_status.config(text="Starting download...")
        self.download_progress['value'] = 0
        self.processing_progress.stop()
        
        # Start processing in background
        thread = threading.Thread(target=self.download_and_process_youtube)
        thread.daemon = True
        thread.start()

    def download_and_process_youtube(self):
        """Download and process YouTube video."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }],
                    'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                    'progress_hooks': [self.youtube_progress_hook],
                }
                
                try:
                    # Reset progress bars and title
                    self.download_progress['value'] = 0
                    self.processing_progress['value'] = 0
                    self.processing_progress['mode'] = 'determinate'
                    self.video_title_label.config(text="")
                    
                    # Download video
                    self.root.after(0, lambda: self.youtube_status.config(text="Downloading..."))
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(self.url_var.get(), download=True)
                        filename = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".wav"
                        
                        # Show video title
                        video_title = info.get('title', 'Unknown Title')
                        self.root.after(0, lambda: self.video_title_label.config(text=video_title))
                    
                    # Process audio
                    self.root.after(0, lambda: self.youtube_status.config(text="Processing audio..."))
                    
                    # Load audio in chunks
                    chunk_length = 30  # seconds
                    full_transcription = []
                    
                    with sf.SoundFile(filename) as audio_file:
                        total_duration = len(audio_file) / audio_file.samplerate
                        total_chunks = int(np.ceil(total_duration / chunk_length))
                        
                        for chunk_idx in range(total_chunks):
                            # Update progress
                            progress = (chunk_idx / total_chunks) * 100
                            self.root.after(0, lambda p=progress: self.processing_progress.config(value=p))
                            self.root.after(0, lambda i=chunk_idx, t=total_chunks: 
                                          self.youtube_status.config(text=f"Processing segment {i+1} of {t}..."))
                            
                            # Calculate chunk size
                            chunk_samples = int(chunk_length * audio_file.samplerate)
                            audio_file.seek(chunk_idx * chunk_samples)
                            audio_data = audio_file.read(chunk_samples)
                            
                            if len(audio_data) == 0:
                                break
                            
                            # Convert to mono if stereo
                            if len(audio_data.shape) > 1:
                                audio_data = audio_data.mean(axis=1)
                            
                            # Process chunk
                            audio_data = audio_data.astype(np.float32)
                            
                            # Normalize
                            if np.abs(audio_data).max() > 1.0:
                                audio_data = audio_data / np.abs(audio_data).max()
                            
                            # Resample if needed
                            if audio_file.samplerate != self.target_sample_rate:
                                audio_data = self.resample_audio(
                                    audio_data, 
                                    audio_file.samplerate, 
                                    self.target_sample_rate
                                )
                            
                            # Clear GPU memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Process chunk
                            input_features = self.processor(
                                audio_data, 
                                sampling_rate=self.target_sample_rate,
                                return_tensors="pt"
                            ).input_features.to(self.model.device)
                            
                            with torch.no_grad():
                                predicted_ids = self.model.generate(
                                    input_features,
                                    language="<|is|>",
                                    task="transcribe"
                                )
                            
                            chunk_text = self.processor.batch_decode(
                                predicted_ids, 
                                skip_special_tokens=True
                            )[0]
                            
                            # Add to transcription
                            if chunk_text.strip():
                                full_transcription.append(chunk_text)
                                # Update UI with current progress
                                current_text = " ".join(full_transcription)
                                self.root.after(0, lambda t=current_text: self.update_youtube_text(t))
                    
                    # Final update
                    final_text = " ".join(full_transcription)
                    self.root.after(0, lambda: self.update_youtube_text(final_text))
                    self.root.after(0, lambda: self.youtube_status.config(text="Processing complete"))
                    
                    # Hide progress bars after delay
                    self.root.after(1000, lambda: self.download_progress.config(value=0))
                    self.root.after(1000, lambda: self.processing_progress.config(value=0))
                    
                except yt_dlp.utils.DownloadError as e:
                    error_msg = str(e)
                    if "ffmpeg not found" in error_msg.lower():
                        error_msg = "FFmpeg is required but not installed. Please install FFmpeg first."
                    self.root.after(0, lambda: self.show_error(error_msg))
                    logging.error(f"YouTube download error: {error_msg}")
                    self.root.after(0, lambda: self.video_title_label.config(text=""))
                    
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda: self.show_error(f"Processing error: {error_msg}"))
                    logging.error(f"Processing error: {e}", exc_info=True)
                    self.root.after(0, lambda: self.video_title_label.config(text=""))
                    
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.show_error(f"Failed to process video: {error_msg}"))
            logging.error(f"YouTube processing error: {e}", exc_info=True)
            self.root.after(0, lambda: self.video_title_label.config(text=""))
            
        finally:
            # Reset UI with a delay
            self.root.after(1000, lambda: self.download_progress.config(value=0))
            self.root.after(1000, lambda: self.processing_progress.config(value=0))
            self.root.after(0, lambda: self.youtube_btn.config(state="normal"))
            self.root.after(0, lambda: self.youtube_status.config(text="Ready"))

    def youtube_progress_hook(self, d):
        """Update download progress."""
        if d['status'] == 'downloading':
            # Calculate progress
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            if total > 0:
                downloaded = d.get('downloaded_bytes', 0)
                progress = (downloaded / total) * 100
                self.root.after(0, lambda: self.download_progress.config(value=progress))
                
                # Update status with speed and ETA
                speed = d.get('speed', 0)
                eta = d.get('eta', 0)
                if speed and eta:
                    status = f"Downloading: {speed/1024/1024:.1f} MB/s, ETA: {eta} seconds"
                    self.root.after(0, lambda: self.youtube_status.config(text=status))

    def update_youtube_text(self, text):
        """Update the YouTube transcription text."""
        # Process text formatting
        formatted_text = self.process_text_formatting(text)
        
        # Update text widget
        self.youtube_text.delete(1.0, tk.END)
        self.youtube_text.insert(tk.END, formatted_text)
        self.youtube_status.config(text="Processing complete")

    def check_server(self):
        """Check if the server has started and open the browser."""
        if self.process and self.process.poll() is None:
            # Check if the process is running and give it a moment to start up
            time.sleep(2)  # Wait for server to initialize
            if self.is_streamlit_running():
                self.status_label.config(text="Application running")
                self.progress.stop()
                self.progress.pack_forget()
                
                # Open browser
                webbrowser.open("http://localhost:8501")
                
                # Minimize launcher
                self.root.iconify()
            else:
                # Wait a bit longer and check again
                self.root.after(2000, self.check_server)
        else:
            messagebox.showerror("Error", "Failed to start server")
            self.progress.stop()
            self.progress.pack_forget()
            self.status_label.config(text="Failed to start")

    def run(self):
        """Start the application."""
        self.root.mainloop()

    def update_memory_usage(self):
        """Update memory usage information."""
        try:
            # System RAM usage
            ram = psutil.virtual_memory()
            ram_usage = f"RAM: {ram.percent}%"
            
            # GPU memory usage if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_usage = f" | GPU: {gpu_memory:.1f}/{gpu_memory_total:.1f}GB"
            else:
                gpu_usage = ""
            
            self.memory_label.config(text=f"📊 {ram_usage}{gpu_usage}")
            
        except Exception as e:
            logging.error(f"Error updating memory usage: {e}")
        
        # Update every second
        self.root.after(1000, self.update_memory_usage)

    def setup_tts_controls(self, parent_frame):
        """Setup TTS controls in a given frame."""
        controls_frame = ttk.Frame(parent_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Voice selection (only show Icelandic voices)
        voice_frame = ttk.Frame(controls_frame)
        voice_frame.pack(fill=tk.X, pady=2)
        ttk.Label(voice_frame, text="Voice:").pack(side=tk.LEFT, padx=5)
        
        # Filter for Icelandic voices
        voices = self.tts_engine.getProperty('voices')
        icelandic_voices = [
            voice for voice in voices 
            if any(identifier in voice.name.lower() 
                  for identifier in ['icelandic', 'íslenska', 'is-is', 'islensku'])
        ]
        
        # If no Icelandic voices found, show all voices
        voice_list = icelandic_voices if icelandic_voices else voices
        voice_names = [voice.name for voice in voice_list]
        
        self.voice_var = tk.StringVar(value=voice_names[0] if voice_names else "Default")
        
        voice_menu = ttk.Combobox(
            voice_frame, 
            textvariable=self.voice_var,
            values=voice_names,
            state="readonly",
            width=30
        )
        voice_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        voice_menu.bind('<<ComboboxSelected>>', self.update_voice)
        
        # Add warning if no Icelandic voices found
        if not icelandic_voices:
            warning_label = ttk.Label(
                voice_frame,
                text="No Icelandic voices found",
                foreground="red"
            )
            warning_label.pack(side=tk.LEFT, padx=5)
        
        # Speed control
        speed_frame = ttk.Frame(controls_frame)
        speed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        
        self.speed_var = tk.IntVar(value=150)
        speed_scale = ttk.Scale(
            speed_frame,
            from_=50,
            to=300,
            variable=self.speed_var,
            command=self.update_speed,
            orient=tk.HORIZONTAL
        )
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Volume control
        volume_frame = ttk.Frame(controls_frame)
        volume_frame.pack(fill=tk.X, pady=2)
        ttk.Label(volume_frame, text="Volume:").pack(side=tk.LEFT, padx=5)
        
        self.volume_var = tk.DoubleVar(value=1.0)
        volume_scale = ttk.Scale(
            volume_frame,
            from_=0.0,
            to=1.0,
            variable=self.volume_var,
            command=self.update_volume,
            orient=tk.HORIZONTAL
        )
        volume_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        return controls_frame

    def update_voice(self, event=None):
        """Update TTS voice."""
        try:
            voices = self.tts_engine.getProperty('voices')
            selected_voice = next(
                voice for voice in voices 
                if voice.name == self.voice_var.get()
            )
            self.tts_engine.setProperty('voice', selected_voice.id)
        except Exception as e:
            logging.error(f"Error updating voice: {e}")

    def update_speed(self, event=None):
        """Update TTS speed."""
        try:
            self.tts_engine.setProperty('rate', self.speed_var.get())
        except Exception as e:
            logging.error(f"Error updating speed: {e}")

    def update_volume(self, event=None):
        """Update TTS volume."""
        try:
            self.tts_engine.setProperty('volume', self.volume_var.get())
        except Exception as e:
            logging.error(f"Error updating volume: {e}")

    def speak_text(self, source='record'):
        """Convert text to speech."""
        try:
            # Get text based on source
            if source == 'youtube':
                text = self.youtube_text.get(1.0, tk.END).strip()
            else:
                text = self.transcription_text.get(1.0, tk.END).strip()
            
            if not text:
                self.show_error("No text to speak")
                return
            
            # Disable button while processing
            btn = self.youtube_tts_btn if source == 'youtube' else self.tts_btn
            btn.config(state="disabled")
            self.status_label.config(text="Speaking...")
            
            def speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.root.after(0, lambda: self.status_label.config(text="Ready"))
                    self.root.after(0, lambda: btn.config(state="normal"))
                except Exception as e:
                    self.root.after(0, lambda: self.show_error(f"TTS Error: {str(e)}"))
                    self.root.after(0, lambda: btn.config(state="normal"))
            
            # Run in background thread
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.show_error(f"TTS Error: {str(e)}")
            btn.config(state="normal")

if __name__ == "__main__":
    launcher = WhisperSSTLauncher()
    launcher.run()
