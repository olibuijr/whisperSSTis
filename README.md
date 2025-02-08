# 🎙️ WhisperSST.is

Real-time Icelandic Speech Recognition powered by Whisper AI

## 🌟 Overview

WhisperSST.is is a 100% local web application that provides real-time Icelandic speech recognition using a fine-tuned version of OpenAI's Whisper model. This tool runs entirely on your machine - no cloud services or internet connection required for processing (only needed for initial model download). Your audio data never leaves your computer, ensuring complete privacy and security.

## ✨ Features

- 🎤 Real-time audio recording and transcription
- 🔒 100% local processing - no cloud or internet needed
- 🚀 Fast, efficient transcription
- 🔊 Instant audio playback
- 📱 User-friendly interface
- 🇮🇸 Specialized for Icelandic language
- 💻 Runs on your hardware (CPU/GPU)
- 📝 Timestamped transcriptions
- 💾 Export to TXT and SRT formats
- 🎵 Supports M4A audio files

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works too)
- Microphone access
- Internet connection (only for initial model download)
- ~4GB disk space for models

### Privacy & Security
- 🔒 100% local processing - your audio never leaves your computer
- 🚫 No cloud services or API calls
- 💻 All transcription happens on your machine
- 🔐 No internet needed after model download
- 🎯 No external dependencies for core functionality

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
```

#### macOS
```bash
brew install portaudio
```

#### Windows
The required libraries are typically included with Python packages.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Magnussmari/whisperSSTis.git
cd whisperSSTis
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Start the application:
```bash
python launcher.py
```

### Development Setup

For developers who want to contribute or modify the application:

1. Set up your development environment:
```bash
# Clone the repository
git clone https://github.com/Magnussmari/whisperSSTis.git
cd whisperSSTis

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. Project Structure:
- `app.py`: Main Streamlit application
- `launcher.py`: GUI launcher for the application
- `whisperSSTis/`: Core module containing audio and transcription logic
- `setup_dependencies.sh/bat`: System dependency installation scripts
- `TODO.md`: Current development tasks and future plans

3. Running in Development Mode:
```bash
# Run with launcher GUI
python launcher.py

# Run Streamlit directly
streamlit run app.py
```

4. Development Guidelines:
- Follow PEP 8 style guidelines
- Add docstrings for new functions
- Update TODO.md for new features/fixes
- Test changes with different audio inputs

### Running Tests

To run the unit tests:
```bash
# Install test dependencies
pip install pytest pytest-mock

# Run tests from project root
pytest
```

### Troubleshooting

#### Common Issues

- **Application won't start**: 
  - Run the setup script for your platform
  - Make sure you have extracted all files from the downloaded package
  - Try running as administrator
  - Check your antivirus isn't blocking the application

- **No audio input**: 
  - Run the setup script to install audio dependencies
  - Check your microphone is properly connected
  - Allow microphone access in your system settings
  - Select the correct input device in the application

- **Slow transcription**:
  - A GPU is recommended but not required
  - First launch may be slow while loading the model
  - Try adjusting chunk size for better performance
  - Models are cached locally for faster subsequent runs

- **PortAudio Error**: 
  - Run `setup_dependencies.sh` (macOS/Linux) or `setup_dependencies.bat` (Windows)
  - Windows: Install Visual C++ Redistributable if prompted
  - Linux: Run `sudo apt-get install portaudio19-dev python3-pyaudio`
  - macOS: Run `brew install portaudio`

- **Missing Dependencies**:
  - Run the setup script for your platform
  - Check the error message for specific missing packages
  - For Windows, ensure Visual C++ Redistributable is installed
  - For Linux, install required system packages using your package manager

For more help, check the [issues page](https://github.com/Magnussmari/whisperSSTis/issues) or create a new issue.

## 💻 Technical Details

- **Frontend**: Streamlit (local web interface)
- **Speech Recognition**: Fine-tuned Whisper model (runs locally)
- **Audio Processing**: PortAudio, PyAudio
- **ML Framework**: PyTorch, Transformers
- **Privacy**: All processing done locally on your machine

## 👥 Credits

### Developer
- **Magnus Smari Smarason**

### Model Credits
- **Original Whisper Model**: [OpenAI](https://github.com/openai/whisper)
- **Icelandic Fine-tuned Model**: [Carlos Daniel Hernandez Mena](https://huggingface.co/carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h)

### Technologies
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Magnussmari/whisperSSTis/issues).

---
<p align="center">
Developed with ❤️ for the Icelandic language community
</p>
