import numpy as np
import cv2


# ============================================================
# STEP 1: Apply Median Filter
# ============================================================

def apply_median_filter_from_scratch(image, size=3):

    """Apply median filtering manually"""

    h, w = image.shape

    # Find padding size
    k = size // 2

    # Add padding around image
    padded_image = np.pad(
        image,
        k,
        mode='edge'
    )

    # Create output image
    output = np.zeros(
        (h, w),
        dtype=np.float32
    )

    # ========================================================
    # STEP 2: Slide Window Across Image
    # ========================================================

    for i in range(h):

        for j in range(w):

            # Take current neighborhood
            region = padded_image[
                i:i + size,
                j:j + size
            ]

            # Convert 2D region into 1D array
            pixels = region.flatten()

            # Sort pixel values
            pixels.sort()

            # Find middle value
            middle = len(pixels) // 2

            # Store median as new pixel
            output[i, j] = pixels[middle]

    return output.astype(np.uint8)


# ============================================================
# STEP 3: Read Image
# ============================================================

image = cv2.imread(r"C:\Image_processing\eye img.jpeg")

if image is None:

    print("Error: Image not found!")
    exit()


# Convert image to grayscale
gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# STEP 4: Apply Median Filter
# ============================================================

custom_median = apply_median_filter_from_scratch(
    gray_image,
    size=3
)


# ============================================================
# STEP 5: Display Images
# ============================================================

cv2.imshow(
    "Original Image",
    gray_image
)

cv2.imshow(
    "Median Filtered Image",
    custom_median
)

cv2.waitKey(0)
cv2.destroyAllWindows()