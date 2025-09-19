import cv2
import numpy as np
import base64
import io
import json
import threading
import time
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import requests
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

class ObjectDetectionSystem:
    def __init__(self):
        self.model = None
        self.cap = None
        self.detection_enabled = False
        self.gpu_enabled = True
        self.running = False
        self.person_count = 0
        self.last_frame = None
        self.detection_thread = None
        self.ollama_url = "http://localhost:11434"
        self.vlm_model = "gemma3:4b"  # Adjust model name as needed
        
        # Initialize YOLO model
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize YOLO model with GPU support"""
        try:
            self.model = YOLO("yolo11n.pt")
            if self.gpu_enabled:
                self.model.to("cuda")
                logger.info("✅ YOLO model loaded on GPU")
            else:
                self.model.to("cpu")
                logger.info("✅ YOLO model loaded on CPU")
        except Exception as e:
            logger.error(f"❌ Error loading YOLO model: {e}")
            
    def initialize_camera(self):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("❌ Error: Could not open webcam")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("✅ Camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error initializing camera: {e}")
            return False
            
    def toggle_detection(self):
        """Toggle object detection on/off"""
        self.detection_enabled = not self.detection_enabled
        logger.info(f"Detection {'enabled' if self.detection_enabled else 'disabled'}")
        return self.detection_enabled
        
    def toggle_gpu(self):
        """Toggle GPU usage for inference"""
        self.gpu_enabled = not self.gpu_enabled
        self.initialize_model()  # Reinitialize model with new device
        logger.info(f"GPU {'enabled' if self.gpu_enabled else 'disabled'}")
        return self.gpu_enabled
        
    def detect_objects(self, frame):
        """Run YOLO detection on frame"""
        if not self.detection_enabled or self.model is None:
            return frame, 0
            
        try:
            device = "cuda" if self.gpu_enabled else "cpu"
            results = self.model(frame, device=device, verbose=False)
            
            # Count persons (class 0 in COCO dataset)
            person_count = 0
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0:  # Person class
                        person_count += 1
            
            # Plot results on frame
            annotated_frame = results[0].plot()
            
            return annotated_frame, person_count
            
        except Exception as e:
            logger.error(f"❌ Error in object detection: {e}")
            return frame, 0
            
    def generate_frames(self):
        """Generate video frames for streaming"""
        if not self.initialize_camera():
            return
            
        self.running = True
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("❌ Failed to grab frame")
                    break
                
                # Apply object detection if enabled
                processed_frame, person_count = self.detect_objects(frame)
                self.person_count = person_count
                self.last_frame = processed_frame.copy()
                
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Emit detection data via SocketIO
                socketio.emit('detection_data', {
                    'person_count': person_count,
                    'detection_enabled': self.detection_enabled,
                    'gpu_enabled': self.gpu_enabled
                })
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"❌ Error in frame generation: {e}")
                break
                
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        self.running = False
        logger.info("✅ Resources cleaned up")
        
    def query_vlm(self, prompt, include_image=False):
        """Query Ollama VLM with optional image context"""
        try:
            url = f"{self.ollama_url}/api/generate"
            
            data = {
                "model": self.vlm_model,
                "prompt": prompt,
                "stream": False
            }
            
            # Include current frame if requested and available
            if include_image and self.last_frame is not None:
                # Convert frame to base64
                ret, buffer = cv2.imencode('.jpg', self.last_frame)
                if ret:
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    data["images"] = [img_base64]
            
            response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response received')
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            logger.error(f"❌ Error querying VLM: {e}")
            return f"Error querying VLM: {str(e)}"

# Initialize detection system
detection_system = ObjectDetectionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(detection_system.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle object detection"""
    enabled = detection_system.toggle_detection()
    return jsonify({'detection_enabled': enabled})

@app.route('/toggle_gpu', methods=['POST'])
def toggle_gpu():
    """Toggle GPU usage"""
    enabled = detection_system.toggle_gpu()
    return jsonify({'gpu_enabled': enabled})

@app.route('/query_vlm', methods=['POST'])
def query_vlm():
    """Query VLM assistant"""
    data = request.get_json()
    prompt = data.get('prompt', '')
    include_image = data.get('include_image', False)
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Enhance prompt with detection context
    context = f"Current person count: {detection_system.person_count}. "
    if detection_system.detection_enabled:
        context += "Object detection is currently active. "
    else:
        context += "Object detection is currently disabled. "
    
    enhanced_prompt = context + prompt
    
    response = detection_system.query_vlm(enhanced_prompt, include_image)
    
    return jsonify({
        'response': response,
        'person_count': detection_system.person_count,
        'detection_enabled': detection_system.detection_enabled
    })

@app.route('/status')
def status():
    """Get current system status"""
    return jsonify({
        'detection_enabled': detection_system.detection_enabled,
        'gpu_enabled': detection_system.gpu_enabled,
        'person_count': detection_system.person_count,
        'camera_active': detection_system.running
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('status', {
        'detection_enabled': detection_system.detection_enabled,
        'gpu_enabled': detection_system.gpu_enabled,
        'person_count': detection_system.person_count
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

if __name__ == '__main__':
    try:
        # Start the detection system in a separate thread
        detection_thread = threading.Thread(target=detection_system.generate_frames)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Run Flask-SocketIO app
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        detection_system.cleanup()

