"""
generate_demo_images.py
-----------------------
Creates 10 synthetic satellite-like images in data/processed/
so the Streamlit dashboard can demo without a real dataset.
Run from the project root:
    python data/generate_demo_images.py
"""
import os
import numpy as np
from PIL import Image, ImageFilter

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIZE = 512
N    = 10

rng = np.random.default_rng(42)

def make_satellite_image(seed: int) -> Image.Image:
    rng_local = np.random.default_rng(seed)
    # Base terrain: brownish / greenish noise
    base = rng_local.integers(40, 120, (SIZE, SIZE, 3), dtype=np.uint8)
    # Terrain color tint (earthy)
    base[:, :, 0] = np.clip(base[:, :, 0].astype(int) + 30, 0, 200)   # R
    base[:, :, 1] = np.clip(base[:, :, 1].astype(int) + 15, 0, 180)   # G
    base[:, :, 2] = np.clip(base[:, :, 2].astype(int) - 10, 0, 140)   # B

    img = Image.fromarray(base, mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    arr = np.array(img)

    # Add rectangular "ruin" patches (faint gray stone-like areas)
    num_patches = rng_local.integers(2, 6)
    for _ in range(num_patches):
        x1 = rng_local.integers(0, SIZE - 30)
        y1 = rng_local.integers(0, SIZE - 30)
        w  = rng_local.integers(15, 50)
        h  = rng_local.integers(15, 50)
        x2, y2 = min(x1 + w, SIZE), min(y1 + h, SIZE)
        color = rng_local.integers(120, 200, 3)
        arr[y1:y2, x1:x2] = color

    # Add green vegetation blobs
    for _ in range(rng_local.integers(3, 8)):
        cx = rng_local.integers(20, SIZE - 20)
        cy = rng_local.integers(20, SIZE - 20)
        r  = rng_local.integers(8, 25)
        Y, X = np.ogrid[:SIZE, :SIZE]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
        arr[mask, 0] = np.clip(arr[mask, 0].astype(int) - 30, 0, 255)
        arr[mask, 1] = np.clip(arr[mask, 1].astype(int) + 60, 0, 255)
        arr[mask, 2] = np.clip(arr[mask, 2].astype(int) - 20, 0, 255)

    # Slight final blur for realism
    result = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    result = result.filter(ImageFilter.GaussianBlur(radius=0.8))
    return result

print(f"[Demo] Generating {N} synthetic satellite images → {OUTPUT_DIR}")
for i in range(N):
    img = make_satellite_image(seed=i * 7 + 13)
    path = os.path.join(OUTPUT_DIR, f"demo_site_{i+1:02d}.jpg")
    img.save(path, quality=92)
    print(f"  Saved: {os.path.basename(path)}")

print(f"[Demo] Done! {N} images ready for the dashboard.")
