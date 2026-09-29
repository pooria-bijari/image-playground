"""
edit.py — Brighten and darken an image by adding/subtracting from every pixel.

This is your first real image edit. The magic is embarrassingly simple:
    new_pixel = old_pixel + amount
"""

import numpy as np
from PIL import Image
from pathlib import Path

# ---------- Configuration ----------
INPUT_FILE  = Path("images/photo.jpg")   # ← change if needed
OUTPUT_DIR  = Path("output")
AMOUNT      = 50                          # how much to add/subtract


def load_image(path: Path) -> np.ndarray:
    """Load an image file and return it as a NumPy array."""
    return np.array(Image.open(path))


def save_image(arr: np.ndarray, path: Path) -> None:
    """Save a NumPy array as an image file."""
    Image.fromarray(arr).save(path)


def brighten(arr: np.ndarray, amount: int) -> np.ndarray:
    """Add 'amount' to every pixel value, clipping to 0–255."""
    brightened = arr + amount
    clipped = np.clip(brightened, 0, 255).astype(np.uint8)
    return clipped


def darken(arr: np.ndarray, amount: int) -> np.ndarray:
    """Subtract 'amount' from every pixel value, clipping to 0–255."""
    darkened = arr - amount
    clipped = np.clip(darkened, 0, 255).astype(np.uint8)
    return clipped


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading: {INPUT_FILE}")
    original = load_image(INPUT_FILE)
    print(f"Shape:   {original.shape}")

    print(f"\nBrightening by +{AMOUNT} ...")
    brighter = brighten(original, AMOUNT)
    save_image(brighter, OUTPUT_DIR / "brighter.jpg")
    print(f"Saved:   output/brighter.jpg")

    print(f"\nDarkening by -{AMOUNT} ...")
    darker = darken(original, AMOUNT)
    save_image(darker, OUTPUT_DIR / "darker.jpg")
    print(f"Saved:   output/darker.jpg")

    print("\n✅ Done! Open the output/ folder to see the results.")