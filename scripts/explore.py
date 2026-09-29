"""
explore.py — Load an image and inspect what it really is.

An image is just a NumPy array of numbers. Let's prove it.
"""

import numpy as np
from PIL import Image
from pathlib import Path

# ---------- Configuration ----------
IMAGE_PATH = Path("images/photo.png")   # ← change this if your file has a different name

# ---------- Load the image ----------
print(f"Loading: {IMAGE_PATH}")
img = Image.open(IMAGE_PATH)

print(f"\n--- PIL Image Info ---")
print(f"Format:  {img.format}")
print(f"Mode:    {img.mode}")
print(f"Size:    {img.size}   (width, height)")

# ---------- Convert to NumPy array ----------
arr = np.array(img)

print(f"\n--- NumPy Array Info ---")
print(f"Shape:   {arr.shape}")
print(f"Dtype:   {arr.dtype}")
print(f"Min:     {arr.min()}")
print(f"Max:     {arr.max()}")

# ---------- Peek at some pixels ----------
print(f"\n--- Pixel Samples ---")
print(f"Top-left pixel (0, 0):        {arr[0, 0]}")
print(f"Center pixel:                 {arr[arr.shape[0] // 2, arr.shape[1] // 2]}")
print(f"Bottom-right pixel (-1, -1):  {arr[-1, -1]}")

# ---------- Interpret the shape ----------
h, w = arr.shape[0], arr.shape[1]
print(f"\n--- Interpretation ---")
print(f"This image is {h} pixels tall and {w} pixels wide.")
print(f"Each pixel has {arr.shape[2]} color channels (Red, Green, Blue).")
print(f"Total numbers in this file: {arr.size:,}")