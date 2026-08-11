import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Image_processing\eye img.jpeg")  # loaded as BGR

colors = ('b', 'g', 'r')  # matches OpenCV's channel order

plt.figure(figsize=(8, 5))
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()