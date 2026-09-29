"""
edges.py — Detect edges in an image using gradients.

An edge = a place where pixel values change sharply.
We compute the gradient (difference) between neighboring pixels.
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


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale using luminance weights."""
    # These weights match how human eyes perceive brightness
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)


def detect_edges(gray: np.ndarray) -> np.ndarray:
    """Find edges by computing gradients in x and y directions."""
    # Convert to int16 so differences don't wrap around
    g = gray.astype(np.int16)

    # Gradient in X direction: difference between right and left neighbor
    grad_x = np.zeros_like(g)
    grad_x[:, 1:-1] = g[:, 2:] - g[:, :-2]

    # Gradient in Y direction: difference between bottom and top neighbor
    grad_y = np.zeros_like(g)
    grad_y[1:-1, :] = g[2:, :] - g[:-2, :]

    # Combine: magnitude of the gradient vector
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Scale to 0–255 for display
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude


def detect_edges_simple(gray: np.ndarray) -> np.ndarray:
    """Simpler version: horizontal + vertical differences only."""
    g = gray.astype(np.int16)
    grad_x = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    grad_y = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    combined = np.clip(grad_x + grad_y, 0, 255).astype(np.uint8)
    return combined


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading: {INPUT_FILE}")
    original = load_image(INPUT_FILE)
    print(f"Shape:   {original.shape}")

    print("\nConverting to grayscale ...")
    gray = to_grayscale(original)
    save_image(gray, OUTPUT_DIR / "grayscale.jpg")
    print(f"Saved:   output/grayscale.jpg")

    print("\nDetecting edges (gradient method) ...")
    edges = detect_edges(gray)
    save_image(edges, OUTPUT_DIR / "edges.jpg")
    print(f"Saved:   output/edges.jpg")

    print("\nDetecting edges (simple method) ...")
    edges_simple = detect_edges_simple(gray)
    save_image(edges_simple, OUTPUT_DIR / "edges_simple.jpg")
    print(f"Saved:   output/edges_simple.jpg")

    print("\n✅ Done! Compare the two edge images.")