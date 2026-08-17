import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Image_processing\Screenshot 2026-08-17 110118.png")

crop = image[15:65, 55:130]

cv2.imwrite("cropped_7.png", crop)

plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()