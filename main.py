from ultralytics import YOLO
import cv2
from playsound import playsound
import threading
from time import sleep, time

video = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

is_playing = False
last_played_time = 0
cooldown_period = 0.2  # Cooldown period in seconds

def play_audio():
    global is_playing, last_played_time
    is_playing = True
    playsound('audio.mp3')
    sleep(2)  # Wait for a short delay after playback to prevent overlapping
    is_playing = False
    last_played_time = time()  # Update last played time after audio finishes

while True:
    ret, frame = video.read()
    results = model.track(frame)
    frame_result = results[0].plot()

    # Check if 'cell phone' is detected in the frame
    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = box.cls
            # Check if 'cell phone' is detected and enough time has passed since last play
            if model.names[int(c)] == 'cell phone' and not is_playing and (time() - last_played_time > cooldown_period):
                # Start a new audio thread
                audio_thread = threading.Thread(target=play_audio)
                audio_thread.start()

    cv2.imshow("Phone Detector", frame_result)

    # Break loop on pressing 'Esc'
    if cv2.waitKey(1) == 27:
        break



