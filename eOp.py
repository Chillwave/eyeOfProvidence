import cv2
import imutils
import numpy as np
import pygame

# eyeOfProvidence got your back (rev 1.3)
# Configuration Variables
frame_width = 480
skip_frames = 9
confidence_threshold = 0.5
nms_threshold = 0.4  # Non-maximum suppression threshold
model_weights = 'yolov3.weights'
model_cfg = 'yolov3.cfg'
input_size = 288
people_threshold = 1
sound_file = 'alert.mp3'

# Load RTSP stream URL from file
with open('stream.txt', 'r') as file:
    stream_url = file.read().strip()

# Initialize Pygame Mixer
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(sound_file)

# Load YOLO
net = cv2.dnn.readNet(model_weights, model_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_persons(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (input_size, input_size), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    final_boxes = []
    if indices is not None and len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])

    return len(final_boxes), final_boxes

cap = cv2.VideoCapture(stream_url)
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received")
            break

        if frame_count % skip_frames == 0:
            frame = imutils.resize(frame, width=frame_width)
            number_of_persons, boxes = detect_persons(frame)
            print(f"Frame {frame_count}: Detected {number_of_persons} persons")
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('Video Stream', frame)

            if number_of_persons >= people_threshold:
                print("DETECTION TRIGGER")
                alert_sound.play()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

except KeyboardInterrupt:
    print("Terminated by the user")

finally:
    cap.release()
    cv2.destroyAllWindows()
