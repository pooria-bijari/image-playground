"""
redder.py — Increase the red channel to make an image look redder.

Demonstrates editing ONE channel instead of all three.
"""

import numpy as np
from PIL import Image
from pathlib import Path

# ---------- Configuration ----------
INPUT_FILE = Path("images/photo.jpg")
OUTPUT_DIR = Path("output")
RED_AMOUNT = 80   # how much red to add (try 30, 80, 150)


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def make_redder(arr: np.ndarray, amount: int) -> np.ndarray:
    """Add 'amount' to the red channel only."""
    result = arr.copy()            # never edit the original in-place
    result[:, :, 0] = np.clip(
        result[:, :, 0].astype(np.int16) + amount, 0, 255
    ).astype(np.uint8)
    return result


def make_greener(arr: np.ndarray, amount: int) -> np.ndarray:
    """Add 'amount' to the green channel only."""
    result = arr.copy()
    result[:, :, 1] = np.clip(
        result[:, :, 1].astype(np.int16) + amount, 0, 255
    ).astype(np.uint8)
    return result


def make_bluer(arr: np.ndarray, amount: int) -> np.ndarray:
    """Add 'amount' to the blue channel only."""
    result = arr.copy()
    result[:, :, 2] = np.clip(
        result[:, :, 2].astype(np.int16) + amount, 0, 255
    ).astype(np.uint8)
    return result


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading: {INPUT_FILE}")
    original = load_image(INPUT_FILE)
    print(f"Shape:   {original.shape}")

    print(f"\nMaking it redder (+{RED_AMOUNT} on R channel) ...")
    redder = make_redder(original, RED_AMOUNT)
    save_image(redder, OUTPUT_DIR / "redder.jpg")

    print(f"Making it greener (+{RED_AMOUNT} on G channel) ...")
    greener = make_greener(original, RED_AMOUNT)
    save_image(greener, OUTPUT_DIR / "greener.jpg")

    print(f"Making it bluer (+{RED_AMOUNT} on B channel) ...")
    bluer = make_bluer(original, RED_AMOUNT)
    save_image(bluer, OUTPUT_DIR / "bluer.jpg")

    print("\n✅ Done! Compare all three in the output/ folder.")