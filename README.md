# eyeOfProvidence
Real-Time Person Detection and Alert System

This project provides a real-time surveillance system that detects people in a video stream from an IP camera using YOLO (You Only Look Once) object detection. When the number of people detected in the frame reaches or exceeds a specified threshold, an audio alert is played.

## Features

- **Real-Time Person Detection**: Utilizes YOLOv3 model to detect persons in a video stream.
- **Audio Alerts**: Plays an audio file when the number of detected persons meets or exceeds a predefined threshold.
- **Adjustable Parameters**: Users can easily modify parameters such as the frame width, skip frames, detection thresholds, and more.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- **Python 3.6+** installed on your system.
- **OpenCV**, **NumPy**, and **pygame** Python libraries installed.
- **YOLOv3 weights and configuration files**: These files should be placed in the same directory as the script or their paths should be correctly specified in the script.
- An **RTSP stream URL** from an IP camera.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. **Install required Python packages**:
   ```bash
   pip install numpy opencv-python pygame
   ```

3. **Download YOLOv3 weights and configuration**:
   - You can download these files from the official YOLO website or via running:
     ```bash
     wget https://pjreddie.com/media/files/yolov3.weights
     wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
     ```

4. **Place your RTSP stream URL**:
   - Open `stream.txt` and replace the existing URL with your RTSP stream URL.

5. **Audio File**:
   - Place your `alert.mp3` file in the same directory as the script or update the `sound_file` path in the script.

## Usage

Run the script using Python:

```bash
python3 console3.py
```

When the script is running, it will display the video stream in a window. Detected persons will be highlighted with green rectangles. If the number of detected persons in a frame meets or exceeds the threshold, an audio alert will play.

## Configuration

You can adjust several parameters in the script to suit your specific requirements:
- `frame_width`: Width of the video frame for processing.
- `skip_frames`: Number of frames to skip between processing to reduce computational load.
- `confidence_threshold`: Confidence threshold for detecting persons.
- `nms_threshold`: Threshold for non-maximum suppression to refine the bounding boxes.
- `people_threshold`: Number of people required to trigger the audio alert.

## Troubleshooting

If you encounter issues with the detection accuracy or performance, consider adjusting the `confidence_threshold`, `nms_threshold`, or processing fewer frames by increasing `skip_frames`.

* This currently only works in well illuminated areas, the system struggles to detect human with grayscale visuals. It also struggles with reaction time. These are issues that limits practical use.
