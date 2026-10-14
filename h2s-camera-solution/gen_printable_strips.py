"""
Generate print-ready H2S lead acetate strip reference images from 5 to 20 ppm.
Colors tuned to real lead acetate chemistry (warm tan -> deep brown) with
contrast boost to survive printer ink washout.
NO changes to main codebase.
"""
import cv2
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "test", "printable_strips")
os.makedirs(OUT_DIR, exist_ok=True)

PRINT_DARKEN = 20  # compensate for printer lightening

# Anchors: (dose_ppm_h, R, G, B)  — real lead acetate warm tan -> deep brown
ANCHORS = [
    (0.0,  210, 195, 155),
    (3.0,  198, 178, 138),
    (5.0,  182, 158, 112),
    (7.0,  168, 140,  90),
    (10.0, 150, 118,  68),
    (13.0, 132,  98,  50),
    (15.0, 115,  82,  40),
    (18.0,  95,  65,  30),
    (20.0,  80,  52,  22),
    (30.0,  58,  38,  15),
]


def interp_color(dose: float):
    """Return (B, G, R) OpenCV color for given dose, with print darkening."""
    for i in range(len(ANCHORS) - 1):
        d0, r0, g0, b0 = ANCHORS[i]
        d1, r1, g1, b1 = ANCHORS[i + 1]
        if d0 <= dose <= d1:
            t = (dose - d0) / (d1 - d0)
            R = int(r0 + t * (r1 - r0)) - PRINT_DARKEN
            G = int(g0 + t * (g1 - g0)) - PRINT_DARKEN
            B = int(b0 + t * (b1 - b0)) - PRINT_DARKEN
            return (max(5, B), max(8, G), max(10, R))   # BGR
    # beyond last anchor
    _, r, g, b = ANCHORS[-1]
    return (max(5, b - PRINT_DARKEN), max(8, g - PRINT_DARKEN), max(10, r - PRINT_DARKEN))


def make_strip(dose: float, w=110, h=380) -> np.ndarray:
    B, G, R = interp_color(dose)
    strip = np.zeros((h, w, 3), dtype=np.float32)
    strip[:, :, 0] = B
    strip[:, :, 1] = G
    strip[:, :, 2] = R

    # Subtle edge darkening (paper curl)
    grad = np.linspace(0.92, 1.0, w // 2)
    grad = np.concatenate([grad, grad[::-1]])[:w]
    strip *= grad[np.newaxis, :, np.newaxis]

    # Paper grain noise
    strip += np.random.normal(0, 3.0, (h, w, 3))
    strip = np.clip(strip, 0, 255).astype(np.uint8)

    cv2.rectangle(strip, (1, 1), (w - 2, h - 2), (25, 25, 25), 2)
    return strip


def make_card(dose: float) -> np.ndarray:
    cw, ch = 200, 520
    card = np.ones((ch, cw, 3), dtype=np.uint8) * 248

    strip = make_strip(dose)
    sx = (cw - 110) // 2
    card[60:440, sx:sx + 110] = strip

    # Dose label
    label = f"{dose:.0f} ppm"
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, _), _ = cv2.getTextSize(label, font, 0.75, 2)
    cv2.putText(card, label, ((cw - tw) // 2, 42), font, 0.75, (15, 15, 15), 2, cv2.LINE_AA)

    # Risk zone
    if dose <= 8:
        zone, zc = "SAFE",    (0, 140, 0)
    elif dose <= 14:
        zone, zc = "WARNING", (0, 120, 210)
    else:
        zone, zc = "DANGER",  (0, 0, 200)
    (zw, _), _ = cv2.getTextSize(zone, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(card, zone, ((cw - zw) // 2, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.44, zc, 1, cv2.LINE_AA)

    # Color swatch
    B, G, R = interp_color(dose)
    cv2.rectangle(card, (sx, 452), (sx + 110, 478), (B, G, R), -1)
    cv2.rectangle(card, (sx, 452), (sx + 110, 478), (50, 50, 50), 1)
    cv2.putText(card, f"R{R} G{G} B{B}", (sx, 498),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1, cv2.LINE_AA)
    return card


# ── Generate individual strips ───────────────────────────────────────────────
doses = list(range(5, 21))   # 5 to 20 inclusive
cards = []

for d in doses:
    B, G, R = interp_color(float(d))
    card = make_card(float(d))
    path = os.path.join(OUT_DIR, f"strip_{d:02d}ppm.png")
    cv2.imwrite(path, card)
    cards.append(card)
    print(f"  strip_{d:02d}ppm.png  |  RGB=({R},{G},{B})  BGR=({B},{G},{R})")

# ── Combined reference sheet (4 cols x 4 rows) ──────────────────────────────
cols = 4
rows = (len(doses) + cols - 1) // cols
cw, ch = cards[0].shape[1], cards[0].shape[0]
gap = 14
sheet_w = cols * cw + (cols + 1) * gap
sheet_h = rows * ch + (rows + 1) * gap + 62

sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 238
cv2.putText(sheet,
    "H2S LEAD ACETATE STRIP REFERENCE  |  5-20 ppm  |  PRINT AT 300 DPI",
    (gap, 38), cv2.FONT_HERSHEY_DUPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
cv2.line(sheet, (gap, 48), (sheet_w - gap, 48), (150, 150, 150), 1)

for i, card in enumerate(cards):
    row, col = divmod(i, cols)
    y = 62 + gap + row * (ch + gap)
    x = gap + col * (cw + gap)
    sheet[y:y + ch, x:x + cw] = card

sheet_path = os.path.join(OUT_DIR, "REFERENCE_SHEET_5to20ppm.png")
cv2.imwrite(sheet_path, sheet)
print(f"\nReference sheet: {sheet_path}  ({sheet_w}x{sheet_h}px)")
print(f"Print tip: 300 DPI, no scaling, matte white paper 80gsm+")
