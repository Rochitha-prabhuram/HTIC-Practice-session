"""
import cv2
img = cv2.imread(r"C:\Image_processing\eye img.jpeg")  
cv2.imshow('Image', img)
cv2.waitKey(0)         
cv2.destroyAllWindows()
"""

import cv2

cap = cv2.VideoCapture(r"C:\Image_processing\WhatsApp Video 2026-07-10 at 12.04.15 PM.mp4")   # or 0 for default webcam

if not cap.isOpened():
    raise IOError("Cannot open video source")

while True:
    ret, frame = cap.read()   # ret: bool success flag, frame: image array
    if not ret:
        break                  # end of video or read failure

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):  # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()