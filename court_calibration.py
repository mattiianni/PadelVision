"""
Court calibration: mappa pixel del video -> coordinate reali del campo (metri).
Campo padel standard: 10m x 20m.

La calibrazione è sequenziale: ogni click corrisponde a un punto noto del campo
(angoli, rete, linee servizio). INVIO conferma quando hai cliccato almeno 4 punti.

Uso:
  calibrator = CourtCalibrator("video.mp4")
  calibrator.calibrate()
  cx, cy = calibrator.transform_point(pixel_x, pixel_y)
"""

import cv2
import numpy as np
import json
import os

COURT_WIDTH_M  = 10.0
COURT_LENGTH_M = 20.0

# 8 punti di riferimento noti del campo padel
# (label, (real_x_m, real_y_m))
CALIB_POINTS = [
    ("Angolo Alto-Sinistra",          ( 0.0,  0.0)),
    ("Angolo Alto-Destra",            (10.0,  0.0)),
    ("Angolo Basso-Destra",           (10.0, 20.0)),
    ("Angolo Basso-Sinistra",         ( 0.0, 20.0)),
    ("Rete - lato Sinistro",          ( 0.0, 10.0)),
    ("Rete - lato Destro",            (10.0, 10.0)),
    ("Linea servizio - Alto-Sinistra",( 0.0,  7.0)),
    ("Linea servizio - Alto-Destra",  (10.0,  7.0)),
]

DOT_COLORS = [
    (0, 255, 80),    # verde
    (0, 200, 255),   # ciano
    (255, 160, 0),   # arancio
    (255, 80, 255),  # magenta
    (80, 255, 80),   # verde chiaro
    (80, 200, 255),  # azzurro
    (255, 220, 0),   # giallo
    (200, 80, 255),  # viola
]


class CourtCalibrator:
    def __init__(self, video_path: str, calibration_path: str = None):
        self.video_path = video_path
        if calibration_path is None:
            name = os.path.splitext(os.path.basename(video_path))[0]
            calibration_path = f"calibration_{name}.json"
        self.calibration_path = calibration_path
        self.src_points = []   # pixel cliccati
        self.dst_points = []   # coordinate reali corrispondenti
        self.H = None

    # ------------------------------------------------------------------
    # Calibrazione
    # ------------------------------------------------------------------

    def calibrate(self, force: bool = False):
        if not force and os.path.exists(self.calibration_path):
            self._load()
            print(f"  Calibrazione caricata da {self.calibration_path}")
            return

        frame = self._read_frame_at_1min()
        self._interactive_click(frame)
        self._compute_homography()
        self._save()
        print(f"  Calibrazione salvata: {len(self.src_points)} punti")

    # ------------------------------------------------------------------
    # Lettura frame
    # ------------------------------------------------------------------

    def _read_frame_at_1min(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        candidates = [
            int(60 * fps),        # ~1 minuto
            int(total * 0.10),
            int(total * 0.02),
            0,
        ]
        frame = None
        for t in candidates:
            t = max(0, min(t, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, f = cap.read()
            if ret and np.mean(f) > 20:
                frame = f
                break
        cap.release()
        if frame is None:
            raise ValueError(f"Impossibile leggere il video: {self.video_path}")
        return frame

    # ------------------------------------------------------------------
    # UI calibrazione
    # ------------------------------------------------------------------

    @staticmethod
    def _text_bg(img, text, pos, fs, color, thick=2):
        """Testo con sfondo nero semi-trasparente."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(text, font, fs, thick)
        x, y = pos
        pad = 5
        overlay = img.copy()
        cv2.rectangle(overlay, (x - pad, y - th - pad),
                      (x + tw + pad, y + bl + pad), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        cv2.putText(img, text, (x, y), font, fs, color, thick, cv2.LINE_AA)

    def _draw_mini_court(self, img, current_idx: int):
        """Disegna un mini-schema del campo in basso a destra con i punti già cliccati."""
        h, w = img.shape[:2]
        cw, ch = 160, 280
        margin = 14
        x0, y0 = w - cw - margin, h - ch - margin

        # Sfondo
        overlay = img.copy()
        cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x0 + cw + 4, y0 + ch + 4),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Linee campo
        def cp(rx, ry):  # real -> pixel mini-court
            return (int(x0 + rx / COURT_WIDTH_M * cw),
                    int(y0 + ry / COURT_LENGTH_M * ch))

        cv2.rectangle(img, cp(0, 0), cp(10, 20), (180, 180, 180), 1)
        cv2.line(img, cp(0, 10), cp(10, 10), (255, 255, 255), 2)     # rete
        cv2.line(img, cp(0, 7),  cp(10, 7),  (140, 140, 140), 1)     # serv. A
        cv2.line(img, cp(0, 13), cp(10, 13), (140, 140, 140), 1)     # serv. B
        cv2.line(img, cp(5, 7),  cp(5, 13),  (140, 140, 140), 1)     # centro

        # Punti di riferimento numerati
        for i, (label, (rx, ry)) in enumerate(CALIB_POINTS):
            px, py = cp(rx, ry)
            if i < len(self.src_points):
                cv2.circle(img, (px, py), 6, DOT_COLORS[i], -1)
            elif i == current_idx:
                cv2.circle(img, (px, py), 6, (255, 255, 255), 2)
            else:
                cv2.circle(img, (px, py), 4, (100, 100, 100), 1)
            cv2.putText(img, str(i + 1), (px + 4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

    def _interactive_click(self, frame: np.ndarray):
        self.src_points = []
        self.dst_points = []

        h, w = frame.shape[:2]
        # Font scale calibrata sulla risoluzione (via di mezzo)
        base = max(w, h) / 1080.0
        fs_title = round(0.75 * base, 2)   # istruzione principale
        fs_small = round(0.55 * base, 2)   # info secondarie

        current_idx = [0]
        done = [False]
        img = [frame.copy()]

        def refresh():
            canvas = img[0].copy()
            ci = current_idx[0]

            # --- Barra superiore ---
            if ci < len(CALIB_POINTS):
                label, _ = CALIB_POINTS[ci]
                title = f"Punto {ci+1}/{len(CALIB_POINTS)}:  {label}"
                color_title = DOT_COLORS[ci % len(DOT_COLORS)]
            else:
                title = "Tutti i punti cliccati!"
                color_title = (0, 255, 120)

            self._text_bg(canvas, title, (16, 42), fs_title, color_title, thick=2)

            hint = (f"INVIO = conferma ({len(self.src_points)} punti"
                    f"{' — ok!' if len(self.src_points) >= 4 else ', min 4'})"
                    f"   |   Z = annulla ultimo   |   ESC = esci")
            self._text_bg(canvas, hint, (16, 42 + int(50 * base)),
                          fs_small, (200, 200, 200), thick=1)

            # Punti già cliccati
            for i, (px, py) in enumerate(self.src_points):
                r = max(8, int(10 * base))
                cv2.circle(canvas, (int(px), int(py)), r, DOT_COLORS[i], -1)
                cv2.circle(canvas, (int(px), int(py)), r + 3, (255, 255, 255), 2)
                lbl, _ = CALIB_POINTS[i]
                short = lbl.split(" - ")[-1] if " - " in lbl else lbl
                self._text_bg(canvas, f"{i+1}: {short}",
                              (int(px) + r + 5, int(py) + 5),
                              fs_small, DOT_COLORS[i], thick=1)
                if i > 0:
                    cv2.line(canvas,
                             (int(self.src_points[i-1][0]), int(self.src_points[i-1][1])),
                             (int(px), int(py)), (180, 180, 180), 1)

            self._draw_mini_court(canvas, ci)
            cv2.imshow("Calibrazione campo — PadelVision", canvas)

        def on_click(event, x, y, flags, param):
            ci = current_idx[0]
            if event == cv2.EVENT_LBUTTONDOWN and ci < len(CALIB_POINTS):
                _, real = CALIB_POINTS[ci]
                self.src_points.append((x, y))
                self.dst_points.append(real)
                current_idx[0] += 1
                refresh()

        cv2.namedWindow("Calibrazione campo — PadelVision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibrazione campo — PadelVision",
                         min(w, 1440), min(h + 40, 900))
        cv2.setMouseCallback("Calibrazione campo — PadelVision", on_click)
        refresh()

        while not done[0]:
            key = cv2.waitKey(20) & 0xFF

            if key == 13:   # INVIO
                if len(self.src_points) >= 4:
                    done[0] = True
                else:
                    # Blink warning
                    warn = img[0].copy()
                    self._text_bg(warn, f"  Clicca almeno 4 punti! ({len(self.src_points)} finora)  ",
                                  (16, 90), fs_title, (0, 60, 255), thick=2)
                    cv2.imshow("Calibrazione campo — PadelVision", warn)

            elif key == ord('z') or key == ord('Z'):   # Undo
                if self.src_points:
                    self.src_points.pop()
                    self.dst_points.pop()
                    current_idx[0] = max(0, current_idx[0] - 1)
                    refresh()

            elif key == 27:   # ESC
                cv2.destroyAllWindows()
                raise RuntimeError("Calibrazione annullata.")

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Omografia
    # ------------------------------------------------------------------

    def _compute_homography(self):
        src = np.float32(self.src_points)
        dst = np.float32(self.dst_points)
        self.H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if self.H is None:
            raise RuntimeError("Omografia non calcolabile. Riprova la calibrazione.")

    # ------------------------------------------------------------------
    # Trasformazione
    # ------------------------------------------------------------------

    def transform_point(self, px: float, py: float) -> tuple[float, float]:
        pt = np.float32([[[px, py]]])
        out = cv2.perspectiveTransform(pt, self.H)
        cx = float(np.clip(out[0][0][0], 0.0, COURT_WIDTH_M))
        cy = float(np.clip(out[0][0][1], 0.0, COURT_LENGTH_M))
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
        data = {
            "src_points": self.src_points,
            "dst_points": self.dst_points,
            "H": self.H.tolist(),
        }
        with open(self.calibration_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        with open(self.calibration_path, "r") as f:
            data = json.load(f)
        self.src_points = [tuple(p) for p in data["src_points"]]
        self.dst_points = [tuple(p) for p in data["dst_points"]]
        self.H = np.array(data["H"], dtype=np.float64)
