
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Kernel generation
# ---------------------------------------------------------------------

def gaussian_kernel(size, sigma):
    
    ax = np.arange(size) - size // 2          # e.g. size=5 -> [-2,-1,0,1,2]
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)           # normalize -> sums to 1
    return kernel


def gaussian_kernel_1d(size, sigma):
    """Generate a 1D Gaussian kernel (used for the separable version)."""
    ax = np.arange(size) - size // 2
    kernel = np.exp(-(ax**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


# ---------------------------------------------------------------------
# 2. Brute-force 2D convolution (single channel)
# ---------------------------------------------------------------------

def convolve2d(image, kernel, border_mode='constant'):
    
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=border_mode)

    out_h, out_w = image.shape
    output = np.zeros((out_h, out_w), dtype=np.float32)

    kernel = np.flipud(np.fliplr(kernel))

    for i in range(out_h):
        for j in range(out_w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


# ---------------------------------------------------------------------
# 3. Faster separable 1D convolution (horizontal then vertical pass)
# ---------------------------------------------------------------------

def convolve1d(image, kernel, axis, border_mode='constant'):
   
    k = len(kernel)
    pad = k // 2

    if axis == 0:  # vertical pass
        padded = np.pad(image, ((pad, pad), (0, 0)), mode=border_mode)
        out = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            out[i, :] = np.sum(padded[i:i + k, :] * kernel[:, None], axis=0)
    else:  # horizontal pass
        padded = np.pad(image, ((0, 0), (pad, pad)), mode=border_mode)
        out = np.zeros_like(image, dtype=np.float32)
        for j in range(image.shape[1]):
            out[:, j] = np.sum(padded[:, j:j + k] * kernel[None, :], axis=1)

    return out


def gaussian_blur_separable(image, size, sigma, border_mode='constant'):
    """Apply Gaussian blur as two 1D passes (much faster than 2D)."""
    k1d = gaussian_kernel_1d(size, sigma)
    temp = convolve1d(image, k1d, axis=0, border_mode=border_mode)    # vertical
    result = convolve1d(temp, k1d, axis=1, border_mode=border_mode)   # horizontal
    return result


# ---------------------------------------------------------------------
# 4. Wrappers for grayscale / color images
# ---------------------------------------------------------------------

def gaussian_blur_gray(image, size=5, sigma=1.0, method='separable'):
    
    image = image.astype(np.float32)

    if method == '2d':
        kernel = gaussian_kernel(size, sigma)
        blurred = convolve2d(image, kernel)
    elif method == 'separable':
        blurred = gaussian_blur_separable(image, size, sigma)
    else:
        raise ValueError("method must be '2d' or 'separable'")

    return np.clip(blurred, 0, 255).astype(np.uint8)


def gaussian_blur_color(image, size=5, sigma=1.0, method='separable'):
   
    image = image.astype(np.float32)
    blurred = np.zeros_like(image)

    for c in range(image.shape[2]):  # loop over B, G, R
        if method == '2d':
            kernel = gaussian_kernel(size, sigma)
            blurred[:, :, c] = convolve2d(image[:, :, c], kernel)
        else:
            blurred[:, :, c] = gaussian_blur_separable(image[:, :, c], size, sigma)

    return np.clip(blurred, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# 5. Demo / sanity check
# ---------------------------------------------------------------------

def main(image_path):
    # ---- Load image ----
    img_color = cv2.imread("C:\Image_processing\eye img.jpeg")
    if img_color is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    size, sigma = 5, 1.5

    # ---- Grayscale: brute-force 2D vs separable vs OpenCV ----
    blurred_2d = gaussian_blur_gray(img_gray, size, sigma, method='2d')
    blurred_sep = gaussian_blur_gray(img_gray, size, sigma, method='separable')
    blurred_cv2 = cv2.GaussianBlur(img_gray, (size, size), sigmaX=sigma)

    diff_2d_sep = np.abs(blurred_2d.astype(np.float32) - blurred_sep.astype(np.float32))
    diff_ours_cv2 = np.abs(blurred_sep.astype(np.float32) - blurred_cv2.astype(np.float32))

    print(f"Max diff (2D vs separable, should be ~0): {diff_2d_sep.max():.4f}")
    print(f"Max diff (ours vs OpenCV, small is expected): {diff_ours_cv2.max():.4f}")
    print("  (nonzero mainly due to different border-handling: we use zero-padding,")
    print("   OpenCV defaults to BORDER_REFLECT_101)")

    # ---- Color version ----
    blurred_color = gaussian_blur_color(img_color, size, sigma, method='separable')

    # ---- Visualize ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title('Original (gray)')

    axes[0, 1].imshow(blurred_2d, cmap='gray')
    axes[0, 1].set_title(f'From-scratch 2D conv\n(size={size}, sigma={sigma})')

    axes[0, 2].imshow(blurred_sep, cmap='gray')
    axes[0, 2].set_title('From-scratch separable\n(faster, same result)')

    axes[1, 0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Original (color)')

    axes[1, 1].imshow(cv2.cvtColor(blurred_color, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('From-scratch blur (color)')

    axes[1, 2].imshow(blurred_cv2, cmap='gray')
    axes[1, 2].set_title('cv2.GaussianBlur\n(reference)')

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('gaussian_filter_comparison.png', dpi=150)
    print("Saved comparison figure to 'gaussian_filter_comparison.png'")
    plt.show()


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'image.jpg'
    main(path)