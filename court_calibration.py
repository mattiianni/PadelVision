"""
Court calibration — 4 punti SEMPRE visibili.

Per qualsiasi angolazione di ripresa padel, questi 4 punti sono sempre
nel frame anche quando il fondo lontano è tagliato:

  1. Angolo vicino-Sinistra  → (0,  20)
  2. Angolo vicino-Destra    → (10, 20)
  3. Rete lato Sinistro      → (0,  10)   dove la rete tocca la parete
  4. Rete lato Destro        → (10, 10)

Opzionali (se visibili, migliorano la precisione):
  5. Linea servizio vicino-Sx → (0,  13)
  6. Linea servizio vicino-Dx → (10, 13)

Campo padel standard: 10m x 20m
"""

import cv2
import numpy as np
import json
import os

COURT_WIDTH_M  = 10.0
COURT_LENGTH_M = 20.0

# Punti nell'ordine di click — (label display, (x_reale_m, y_reale_m))
#
# La camera è tipicamente sulla parete di fondo (y=20).
# I 4 OBBLIGATORI sono sempre visibili anche con i corner vicini tagliati:
#   - rete sx/dx (y=10) : centro campo, sempre nel frame
#   - linea servizio vicino sx/dx (y=13) : in vista anche senza i corner
# Gli OPZIONALI migliorano l'accuratezza se visibili.
#
CALIB_POINTS = [
    ("Rete — lato Sinistro",          ( 0.0, 10.0)),   # sempre visibile
    ("Rete — lato Destro",            (10.0, 10.0)),   # sempre visibile
    ("Serv. vicino — Sinistra",       ( 0.0, 13.0)),   # sempre visibile
    ("Serv. vicino — Destra",         (10.0, 13.0)),   # sempre visibile
    ("Angolo lontano — Sx [opt]",     ( 0.0,  0.0)),   # fondo opposto
    ("Angolo lontano — Dx [opt]",     (10.0,  0.0)),   # fondo opposto
    ("Angolo vicino — Sx [opt]",      ( 0.0, 20.0)),   # potrebbe essere tagliato
    ("Angolo vicino — Dx [opt]",      (10.0, 20.0)),   # potrebbe essere tagliato
]

COLORS = [
    (0,  255,  80),   # verde       — ang sx
    (0,  200, 255),   # ciano       — ang dx
    (255, 140,  0),   # arancio     — rete sx
    (255,  60,  60),  # rosso       — rete dx
    (200, 100, 255),  # viola       — serv sx
    (255, 220,   0),  # giallo      — serv dx
]

# Mini schema del campo (coordinate reali) usato come overlay
_COURT_LINES = [
    # Bordo esterno
    ((0,0),   (10,0)),  ((10,0),  (10,20)),
    ((10,20), (0,20)),  ((0,20),  (0,0)),
    # Rete
    ((0,10),  (10,10)),
    # Linee servizio
    ((0,7),   (10,7)),  ((0,13),  (10,13)),
    # Centro servizio
    ((5,7),   (5,13)),
]


class CourtCalibrator:
    def __init__(self, video_path: str, calibration_path: str = None):
        self.video_path = video_path
        if calibration_path is None:
            name = os.path.splitext(os.path.basename(video_path))[0]
            calibration_path = f"calibration_{name}.json"
        self.calibration_path = calibration_path
        self.H = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def calibrate(self, force: bool = False):
        if not force and os.path.exists(self.calibration_path):
            self._load()
            print(f"  Calibrazione caricata da {self.calibration_path}")
            return

        frame = self._read_frame_at_1min()
        self._interactive_click(frame)
        self._save()
        print("  Calibrazione salvata.")

    # ------------------------------------------------------------------
    # UI interattiva
    # ------------------------------------------------------------------

    def _interactive_click(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        base = max(w, h) / 1080.0
        fs   = round(0.72 * base, 2)
        fs_s = round(0.54 * base, 2)

        src_pts = []   # pixel cliccati
        dst_pts = []   # coordinate reali corrispondenti
        current = [0]  # indice punto corrente

        img = [frame.copy()]

        def refresh():
            canvas = img[0].copy()
            ci = current[0]

            # --- Istruzione principale ---
            if ci < len(CALIB_POINTS):
                label, _ = CALIB_POINTS[ci]
                required = "(obbligatorio)" if ci < 4 else "(opzionale — premi INVIO per saltare)"
                self._txt(canvas, f"Punto {ci+1}: {label}  {required}",
                          (16, 44), fs, COLORS[ci % len(COLORS)])
            else:
                self._txt(canvas, "Tutti i punti cliccati!",
                          (16, 44), fs, (0, 255, 120))

            self._txt(canvas,
                      "INVIO = conferma/prossimo   Z = annulla ultimo   ESC = esci",
                      (16, 44 + int(48 * base)), fs_s, (170, 170, 170))

            # --- Punti già cliccati ---
            for i, (px, py) in enumerate(src_pts):
                r = max(8, int(10 * base))
                cv2.circle(canvas, (int(px), int(py)), r, COLORS[i], -1)
                cv2.circle(canvas, (int(px), int(py)), r + 3, (255, 255, 255), 2)
                lbl = CALIB_POINTS[i][0].split(" [")[0]   # rimuovi "[opt]"
                self._txt(canvas, lbl,
                          (int(px) + r + 5, int(py) + 5), fs_s, COLORS[i], 1)

            # --- Mini schema campo ---
            self._draw_mini_court(canvas, src_pts, dst_pts, ci)

            cv2.imshow("PadelVision — Calibrazione", canvas)

        def on_click(event, x, y, flags, param):
            ci = current[0]
            if event == cv2.EVENT_LBUTTONDOWN and ci < len(CALIB_POINTS):
                _, real = CALIB_POINTS[ci]
                src_pts.append((x, y))
                dst_pts.append(real)
                current[0] += 1
                refresh()

        cv2.namedWindow("PadelVision — Calibrazione", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PadelVision — Calibrazione", min(w, 1440), min(h, 900))
        cv2.setMouseCallback("PadelVision — Calibrazione", on_click)
        refresh()

        done = [False]
        while not done[0]:
            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # INVIO — conferma e chiudi
                if len(src_pts) >= 4:
                    done[0] = True
                else:
                    warn = img[0].copy()
                    self._txt(warn,
                              f"Clicca almeno 4 punti! ({len(src_pts)}/4)",
                              (16, 44), fs, (0, 60, 255))
                    cv2.imshow("PadelVision — Calibrazione", warn)

            elif key == ord('z') or key == ord('Z'):  # Undo
                if src_pts:
                    src_pts.pop()
                    dst_pts.pop()
                    current[0] = max(0, current[0] - 1)
                    refresh()

            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                raise RuntimeError("Calibrazione annullata.")

        cv2.destroyAllWindows()

        # Calcola omografia
        src = np.float32(src_pts)
        dst = np.float32(dst_pts)
        self.H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if self.H is None:
            raise RuntimeError("Omografia non calcolabile. Riprova.")

    # ------------------------------------------------------------------
    # Mini schema campo (overlay in basso a destra)
    # ------------------------------------------------------------------

    def _draw_mini_court(self, img, src_pts, dst_pts, current_idx):
        h, w = img.shape[:2]
        cw, ch = 150, 260
        m = 14
        x0, y0 = w - cw - m, h - ch - m

        # Sfondo
        ov = img.copy()
        cv2.rectangle(ov, (x0-4, y0-4), (x0+cw+4, y0+ch+4), (18, 18, 18), -1)
        cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)

        def r2p(rx, ry):  # real metres → mini court pixel
            return (int(x0 + rx / COURT_WIDTH_M * cw),
                    int(y0 + ry / COURT_LENGTH_M * ch))

        for (a, b) in _COURT_LINES:
            cv2.line(img, r2p(*a), r2p(*b), (130, 130, 130), 1)

        # Punti di riferimento
        for i, (label, (rx, ry)) in enumerate(CALIB_POINTS):
            px, py = r2p(rx, ry)
            if i < len(src_pts):
                cv2.circle(img, (px, py), 6, COLORS[i], -1)
            elif i == current_idx:
                cv2.circle(img, (px, py), 7, (255, 255, 255), 2)
            else:
                cv2.circle(img, (px, py), 4, (80, 80, 80), 1)

    # ------------------------------------------------------------------
    # Lettura frame
    # ------------------------------------------------------------------

    def _read_frame_at_1min(self) -> np.ndarray:
        cap   = cv2.VideoCapture(self.video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for t in [int(60 * fps), int(total * 0.10), int(total * 0.02), 0]:
            t = max(0, min(t, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, f = cap.read()
            if ret and np.mean(f) > 20:
                cap.release()
                return f
        cap.release()
        raise ValueError(f"Impossibile leggere il video: {self.video_path}")

    # ------------------------------------------------------------------
    # Testo con sfondo
    # ------------------------------------------------------------------

    @staticmethod
    def _txt(img, text, pos, fs, color, thick=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(text, font, fs, thick)
        x, y = pos
        pad = 5
        ov = img.copy()
        cv2.rectangle(ov, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        cv2.putText(img, text, (x, y), font, fs, color, thick, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Trasformazione
    # ------------------------------------------------------------------

    def transform_point(self, px: float, py: float) -> tuple[float, float]:
        pt  = np.float32([[[px, py]]])
        out = cv2.perspectiveTransform(pt, self.H)
        cx  = float(np.clip(out[0][0][0], 0.0, COURT_WIDTH_M))
        cy  = float(np.clip(out[0][0][1], 0.0, COURT_LENGTH_M))
        return cx, cy

    def transform_points_batch(self, points: np.ndarray) -> np.ndarray:
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        out[:, 0] = np.clip(out[:, 0], 0.0, COURT_WIDTH_M)
        out[:, 1] = np.clip(out[:, 1], 0.0, COURT_LENGTH_M)
        return out

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    def _save(self):
        with open(self.calibration_path, "w") as f:
            json.dump({"H": self.H.tolist()}, f, indent=2)

    def _load(self):
        with open(self.calibration_path, "r") as f:
            self.H = np.array(json.load(f)["H"], dtype=np.float64)
