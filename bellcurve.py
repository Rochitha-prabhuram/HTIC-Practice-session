import numpy as np
import cv2


# Step 1: Create Gaussian Kernel
def create_gaussian_kernel(size=3, sigma=1.0):

    kernel = np.zeros((size, size), dtype=np.float32)
    k = size // 2

    for x in range(-k, k + 1):
        for y in range(-k, k + 1):

            exponent = -(x**2 + y**2) / (2 * sigma**2)
            kernel[x + k, y + k] = np.exp(exponent)

    # Normalize
    kernel = kernel / np.sum(kernel)

    return kernel


# Step 2: Apply Gaussian Filter
def apply_gaussian_blur(image, size=3, sigma=1.0):

    h, w = image.shape
    k = size // 2

    # Create kernel
    kernel = create_gaussian_kernel(size, sigma)

    print("Gaussian Kernel:")
    print(kernel)

    # Padding
    padded_image = np.pad(image, k, mode='edge')

    # Output image
    output = np.zeros((h, w), dtype=np.float32)

    # Convolution
    for i in range(h):
        for j in range(w):

            region = padded_image[
                i:i + size,
                j:j + size
            ]

            output[i, j] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


# Step 3: Read Image
image = cv2.imread(r"C:\Image_processing\eye img.jpeg")

if image is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Step 4: Apply Gaussian Filter
custom_blurred = apply_gaussian_blur(
    gray_image,
    size=3,
    sigma=5.0
)


# Step 5: Display
cv2.imshow("Original Image", gray_image)
cv2.imshow("Gaussian Filtered Image", custom_blurred)

cv2.waitKey(0)
cv2.destroyAllWindows()