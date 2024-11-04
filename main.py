from ultralytics import YOLO
import cv2
from time import sleep
from playsound import playsound

video = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

while True:
    ret, frame = video.read()
    results = model.track(frame)
    frame_result = results[0].plot()

    for r in results:
        
        boxes = r.boxes
        for box in boxes:
            
            c = box.cls
            if model.names[int(c)] == 'cell phone':
                print("Go back and study")
                playsound('audio.mp3')
                sleep(2)

    cv2.imshow("Phone Detector",frame_result)

    if cv2.waitKey(1) == 27:
        break



