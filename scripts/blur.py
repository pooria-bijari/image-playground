"""
blur.py — Blur an image using convolution.

Convolution = slide a small kernel over the image and combine neighbors.
This is the foundation of CNNs, Photoshop filters, and almost all CV.
"""

import numpy as np
from PIL import Image
from pathlib import Path

# ---------- Configuration ----------
INPUT_FILE = Path("images/photo.jpg")
OUTPUT_DIR = Path("output")


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a 2D convolution to a grayscale image.

    For each pixel:
        1. Take the surrounding block (same size as kernel)
        2. Multiply each value by the corresponding kernel value
        3. Sum it all up → that's the new pixel value
    """
    kh, kw = kernel.shape
    h, w = image.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad the image so edges don't get cut off
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    # Slide the kernel over every pixel
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


def blur_kernel(size: int = 5) -> np.ndarray:
    """Create a box blur kernel of the given size (must be odd)."""
    return np.ones((size, size), dtype=np.float32) / (size * size)


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Create a Gaussian blur kernel — softer, more natural than box blur."""
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading: {INPUT_FILE}")
    original = load_image(INPUT_FILE)
    print(f"Shape:   {original.shape}")

    # We'll blur the grayscale version (simpler, faster)
    print("\nConverting to grayscale ...")
    gray = to_grayscale(original)

    print("\nApplying box blur (5x5 kernel) ...")
    box = blur_kernel(5)
    blurred_box = convolve2d(gray, box)
    save_image(blurred_box, OUTPUT_DIR / "blur_box.jpg")
    print("Saved:   output/blur_box.jpg")

    print("\nApplying Gaussian blur (5x5 kernel) ...")
    gauss = gaussian_kernel(5, sigma=1.5)
    blurred_gauss = convolve2d(gray, gauss)
    save_image(blurred_gauss, OUTPUT_DIR / "blur_gaussian.jpg")
    print("Saved:   output/blur_gaussian.jpg")

    print("\n✅ Done! Compare the two blurred images.")