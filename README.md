# 🎯 YOLO Object Detection with VLM Assistant

A real-time object detection application combining YOLOv11 with Vision Language Model (VLM) capabilities, optimized for NVIDIA Jetson Orin NX development boards.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![JetPack](https://img.shields.io/badge/JetPack-6.0-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)


## 📹 Demo

Here’s a demo video of the project in action:

👉 ![Output Demo](output.mp4)



## 🚀 Features

- **Real-time Object Detection**: YOLOv11 powered person detection with live webcam streaming
- **VLM Integration**: Local Ollama-based Vision Language Model for intelligent scene analysis
- **GPU Toggle**: Dynamic switching between CUDA and CPU inference
- **Web Interface**: Professional Flask-based dashboard with real-time controls
- **WebSocket Support**: Live status updates and real-time communication
- **Optimized for Jetson**: Specifically designed for NVIDIA Jetson Orin NX performance

## 🏗️ System Requirements

### Hardware
- **NVIDIA Jetson Orin NX** (8GB recommended)
- **USB Webcam** or CSI camera
- **Storage**: Minimum 8GB free space
- **RAM**: 16GB memory

### Software
- **JetPack 6.0** (Ubuntu 22.04 base)
- **Python 3.10** with virtual environment
- **CUDA 12.x** (included with JetPack 6.0)

## 🛠️ Installation Guide

### 1. System Preparation

First, ensure your Jetson Orin NX is running JetPack 6.0:

```
# Check JetPack version
sudo apt show nvidia-jetpack
```

### 2. Clone Repository

```
git clone https://github.com/12boopathi/Vision-Assistant-flask-vlm--jetson-orin.git
cd Vision-Assistant-flask-vlm--jetson-orin
```

### 3. Python 3.10 Virtual Environment Setup

Create and activate a Python 3.10 virtual environment 

```
# Install Python 3.10 venv if not available
sudo apt update
sudo apt install python3.10-venv python3.10-dev

# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 4. Install Dependencies

```
# Install PyTorch with CUDA support for Jetson
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install application requirements
pip install -r requirements.txt
```

### 5. Install and Configure Ollama

```
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull Gemma VLM model (choose based on your memory)
ollama pull gemma3:4b
# Alternative: ollama pull llava:7b
```

## 📁 Project Structure

```
yolo-vlm-detection/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Web interface
```

## 🚀 Quick Start

### 1. Activate Environment

```
cd Vision-Assistant-flask-vlm--jetson-orin
source venv/bin/activate
```

### 2. Start Ollama Service

```
# Start Ollama in background
ollama serve &
```

### 3. Run Application

```
python app.py
```

### 4. Access Web Interface

Open your browser and navigate to:
- **Local access**: `http://localhost:5000`
- **Network access**: `http://YOUR_JETSON_IP:5000`

## 💡 Usage Guide

### Web Interface Controls

1. **Video Feed**: Live webcam stream with optional YOLO detection overlay
2. **Detection Toggle**: Enable/disable object detection in real-time
3. **GPU Toggle**: Switch between CUDA and CPU inference
4. **VLM Chat**: Interact with the vision language model
5. **Status Panel**: Monitor system performance and detection statistics

### VLM Chat Examples

Try these prompts with the VLM assistant:

- `"What do you see in the current frame?"`
- `"How many people are visible?"`
- `"Describe the scene in detail"`
- `"What objects can you identify?"`
- `"Is anyone wearing bright colors?"`

### Performance Optimization

#### For Maximum Performance (GPU):
```
# In app.py, ensure these settings:
detection_system.gpu_enabled = True
detection_system.model.to("cuda")
```

#### For Power Efficiency (CPU):
```
# Toggle to CPU mode via web interface or:
detection_system.gpu_enabled = False
detection_system.model.to("cpu")
```

## 🔧 Configuration

### Camera Settings

Modify camera parameters in `app.py`:

```
# Camera resolution and FPS
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
self.cap.set(cv2.CAP_PROP_FPS, 30)
```

### YOLO Model Configuration

Change YOLO model variant:

```
# Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
self.model = YOLO("yolo11n.pt")  # Nano (fastest)
# self.model = YOLO("yolo11s.pt")  # Small (balanced)
```

### Ollama Model Configuration

Switch VLM models:

```
# In app.py
self.vlm_model = "gemma2-3b-vlm"    # Default
# self.vlm_model = "llava:7b"       # Alternative
# self.vlm_model = "llava:13b"      # More capable (requires more RAM)
```

## 🔬 Technical Details

### Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │◄───┤   Flask Server   │◄───┤  YOLO Detection │
│                 │    │                  │    │                 │
│  - Video Stream │    │ - WebSocket API  │    │ - CUDA/CPU      │
│  - Control UI   │    │ - REST Endpoints │    │ - YOLOv11       │
│  - VLM Chat     │    │ - Template Render│    │ - Person Count  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Ollama VLM API  │
                       │                 │
                       │ - Gemma Model   │
                       │ - Image Analysis│
                       │ - Local Inference│
                       └─────────────────┘
```

## 🙏 Acknowledgments

- **NVIDIA**: For Jetson platform and CUDA support
- **Ultralytics**: For YOLOv11 implementation
- **Ollama**: For local LLM infrastructure
- **Google**: For Gemma VLM models

```
