"""
UI per assegnare i nomi ai giocatori dopo il tracking.

Mostra un mosaico OpenCV con i crop dei 4 giocatori e chiede i nomi
nel terminale. La finestra rimane aperta finché tutti i nomi sono stati inseriti.
"""

import cv2
import numpy as np
import os


# Dimensioni thumbnail nella finestra
THUMB_W = 200
THUMB_H = 360
PADDING  = 20
HEADER_H = 60

# Colori per ogni slot player (BGR)
PLAYER_COLORS_BGR = [
    (80,  80,  255),   # rosso
    (255, 160, 100),   # blu
    (40,  160, 255),   # arancio
    (255, 100, 220),   # viola
]


def _fit_crop(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Ridimensiona l'immagine mantenendo aspect ratio, centrata su sfondo nero."""
    ih, iw = img.shape[:2]
    scale = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    dy = (h - nh) // 2
    dx = (w - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas


def name_players(crop_paths: dict, player_ids: list) -> dict:
    """
    Mostra i crop dei giocatori e legge i nomi dal terminale.

    crop_paths : {player_id: path_png}  (può mancare qualche player)
    player_ids : lista ordinata di player_id (es. [1, 2, 3, 4])

    Ritorna: {player_id: nome_stringa}
    """
    n = len(player_ids)
    canvas_w = n * THUMB_W + (n + 1) * PADDING
    canvas_h = THUMB_H + HEADER_H + PADDING * 2

    canvas = np.full((canvas_h, canvas_w, 3), 28, dtype=np.uint8)

    # Titolo
    cv2.putText(
        canvas,
        "PadelVision — Guarda i crop e assegna i nomi nel terminale",
        (PADDING, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    for i, pid in enumerate(player_ids):
        x = PADDING + i * (THUMB_W + PADDING)
        y = HEADER_H
        color = PLAYER_COLORS_BGR[i % len(PLAYER_COLORS_BGR)]

        # Crop o placeholder
        crop_path = crop_paths.get(pid)
        if crop_path and os.path.exists(crop_path):
            img = cv2.imread(crop_path)
            if img is not None:
                thumb = _fit_crop(img, THUMB_W, THUMB_H)
                canvas[y:y + THUMB_H, x:x + THUMB_W] = thumb
            else:
                _draw_placeholder(canvas, x, y, THUMB_W, THUMB_H)
        else:
            _draw_placeholder(canvas, x, y, THUMB_W, THUMB_H)

        # Bordo colorato attorno al crop
        cv2.rectangle(canvas, (x - 2, y - 2), (x + THUMB_W + 2, y + THUMB_H + 2), color, 2)

        # Label "Player N" sopra il crop
        label = f"Player {pid}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tx = x + (THUMB_W - tw) // 2
        ty = y - 10
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2, cv2.LINE_AA)

    # Mostra finestra
    win = "PadelVision — Nomi Giocatori"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, canvas)
    cv2.waitKey(200)   # primo refresh per rendere la finestra visibile su macOS

    # Leggi i nomi dal terminale (la finestra rimane aperta)
    names = {}
    print()
    print("─" * 56)
    print("  ASSEGNA I NOMI AI GIOCATORI")
    print("  (la finestra mostra i crop — premi INVIO per default)")
    print("─" * 56)

    for i, pid in enumerate(player_ids):
        cv2.waitKey(1)   # mantieni finestra reattiva tra un input e l'altro
        team = "A" if i < 2 else "B"
        try:
            name = input(f"  Player {pid}  [Team {team}]  → ").strip()
        except EOFError:
            name = ""
        names[pid] = name if name else f"Player {pid}"

    print("─" * 56)

    # Chiudi finestra in modo affidabile (come calibrazione)
    cv2.destroyWindow(win)
    for _ in range(10):
        cv2.waitKey(30)

    return names


def _draw_placeholder(canvas, x, y, w, h):
    """Disegna un rettangolo grigio con '?' per player senza crop."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.putText(
        canvas, "?",
        (x + w // 2 - 12, y + h // 2 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5, (100, 100, 100), 4,
    )
