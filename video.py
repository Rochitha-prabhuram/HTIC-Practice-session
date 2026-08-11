import cv2
cap = cv2.VideoCapture(r"C:\Image_processing\WhatsApp Video 2026-07-10 at 12.04.15 PM.mp4")   # or 0 for default webcam

if not cap.isOpened():
    raise IOError("Cannot open video source")

while True:
    ret, frame = cap.read()   
    if not ret:
        break                  
    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()