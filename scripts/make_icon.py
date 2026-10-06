#!/usr/bin/env python3
"""Draw the FringeLab icon: an "FL" monogram over a band of interference fringes.

Writes assets/fringelab_icon.png (512 px, used as the browser-tab icon by app.py)
and assets/fringelab_icon_64.png. The fringe pattern is a chirped cosine whose
spacing widens toward the critical wavelength, like the real spectra.

Usage: python scripts/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
SIZE = 512
NAVY, PALE, INK = (0x14, 0x2A, 0x4E), (0xDD, 0xE7, 0xF5), (0xFF, 0xFF, 0xFF)


def fringes(width: int, height: int) -> Image.Image:
    """Vertical bands whose spacing grows toward a turning point at ~62 % of the width."""
    x = np.linspace(0, 1, width)
    phase = 26 * np.pi * np.sign(x - 0.62) * np.abs(x - 0.62) ** 1.6   # chirp: wide near 0.62
    t = 0.5 + 0.5 * np.cos(phase)
    rgb = np.array(NAVY)[None, :] + (np.array(PALE) - np.array(NAVY))[None, :] * t[:, None]
    row = rgb.astype(np.uint8)
    return Image.fromarray(np.repeat(row[None, :, :], height, axis=0), "RGB")


def rounded_mask(size: int, radius: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return m


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def make(size: int) -> Image.Image:
    img = fringes(size, size)
    # a centred plate so the monogram reads on every stripe
    plate = (int(size * 0.17), int(size * 0.29), int(size * 0.83), int(size * 0.75))
    shade = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shade).rounded_rectangle(plate, radius=int(size * 0.08), fill=(*NAVY, 215))
    img = Image.alpha_composite(img.convert("RGBA"), shade)
    d = ImageDraw.Draw(img)
    f = font(int(size * 0.40))
    # centre the glyphs' ink box (not the font's line box) inside the plate
    left, top, right, bottom = d.textbbox((0, 0), "FL", font=f)
    cx, cy = (plate[0] + plate[2]) / 2, (plate[1] + plate[3]) / 2
    d.text((cx - (left + right) / 2, cy - (top + bottom) / 2), "FL", font=f, fill=INK)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), rounded_mask(size, int(size * 0.2)))
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    icon = make(SIZE)
    icon.save(OUT / "fringelab_icon.png")
    icon.resize((64, 64), Image.LANCZOS).save(OUT / "fringelab_icon_64.png")
    print(f"written: {OUT / 'fringelab_icon.png'}, {OUT / 'fringelab_icon_64.png'}")


if __name__ == "__main__":
    main()
