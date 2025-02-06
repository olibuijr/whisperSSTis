# 🎙️ WhisperSST.is

Real-time Icelandic Speech Recognition powered by Whisper AI

## 🌟 Overview

WhisperSST.is is a web application that provides real-time Icelandic speech recognition using a fine-tuned version of OpenAI's Whisper model. This tool allows users to record their voice and receive instant, accurate Icelandic transcriptions.

## ✨ Features

- 🎤 Real-time audio recording
- 🚀 Fast transcription processing
- 🔊 Instant audio playback
- 📱 User-friendly interface
- 🇮🇸 Specialized for Icelandic language

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Microphone access
- Internet connection (for model download)

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

### Installation Steps

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

### Running the Application

1. Start the Streamlit server:
```bash
streamlit run sst_is_test.py
```

2. Open your web browser and navigate to:
```
http://localhost:8501
```

### Troubleshooting

- **PortAudio Error**: Make sure you've installed the system dependencies
- **CUDA Error**: Check your GPU drivers and PyTorch installation
- **Microphone Error**: Ensure your microphone is properly connected and has necessary permissions

## 💻 Technical Details

- **Frontend**: Streamlit
- **Speech Recognition**: Fine-tuned Whisper model
- **Audio Processing**: PortAudio, PyAudio
- **ML Framework**: PyTorch, Transformers

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
