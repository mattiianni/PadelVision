"""
Court calibration: mappa pixel del video -> coordinate reali del campo (metri).
Il campo padel standard è 10m x 20m.

Uso:
  calibrator = CourtCalibrator("video.mp4")
  calibrator.calibrate()   # apre finestra interattiva se non c'è già il json
  cx, cy = calibrator.transform_point(pixel_x, pixel_y)
"""

import cv2
import numpy as np
import json
import os

COURT_WIDTH_M  = 10.0   # larghezza campo (m)
COURT_LENGTH_M = 20.0   # lunghezza campo (m)


class CourtCalibrator:
    def __init__(self, video_path: str, calibration_path: str = None):
        self.video_path = video_path
        if calibration_path is None:
            name = os.path.splitext(os.path.basename(video_path))[0]
            calibration_path = f"calibration_{name}.json"
        self.calibration_path = calibration_path
        self.points = []   # 4 punti cliccati nel frame
        self.H = None       # matrice di omografia (pixel -> metri)

    # ------------------------------------------------------------------
    # Calibrazione interattiva
    # ------------------------------------------------------------------

    def calibrate(self, force: bool = False):
        """
        Se esiste già il file JSON di calibrazione, lo carica.
        Altrimenti apre una finestra OpenCV per cliccare i 4 angoli del campo.
        Con force=True ricalibra anche se il file esiste.
        """
        if not force and os.path.exists(self.calibration_path):
            self._load()
            print(f"  Calibrazione caricata da {self.calibration_path}")
            return

        frame = self._read_first_frame()
        self._interactive_click(frame)
        self._compute_homography()
        self._save()
        print(f"  Calibrazione salvata in {self.calibration_path}")

    def _read_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Impossibile leggere il video: {self.video_path}")
        return frame

    def _interactive_click(self, frame):
        """Finestra interattiva: clicca i 4 angoli del campo."""
        self.points = []
        clone = frame.copy()

        labels = ["1: Alto-Sx", "2: Alto-Dx", "3: Basso-Dx", "4: Basso-Sx"]
        instructions = [
            "Clicca i 4 angoli del campo nell'ordine indicato:",
            "1. Alto-Sinistra   2. Alto-Destra",
            "3. Basso-Destra    4. Basso-Sinistra",
            "Poi premi INVIO per confermare (ESC = annulla)"
        ]

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
                self.points.append((x, y))
                cv2.circle(clone, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(clone, labels[len(self.points) - 1],
                            (x + 12, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                if len(self.points) > 1:
                    cv2.line(clone, self.points[-2], self.points[-1],
                             (0, 255, 0), 2)
                if len(self.points) == 4:
                    cv2.line(clone, self.points[-1], self.points[0],
                             (0, 255, 0), 2)
                cv2.imshow("Calibrazione campo", clone)

        cv2.namedWindow("Calibrazione campo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibrazione campo", 1280, 720)
        cv2.setMouseCallback("Calibrazione campo", on_click)

        for i, txt in enumerate(instructions):
            cv2.putText(clone, txt, (10, 30 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

        cv2.imshow("Calibrazione campo", clone)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13 and len(self.points) == 4:   # ENTER
                break
            elif key == 27:                             # ESC
                cv2.destroyAllWindows()
                raise RuntimeError("Calibrazione annullata dall'utente.")

        cv2.destroyAllWindows()

    def _compute_homography(self):
        if len(self.points) != 4:
            raise RuntimeError("Servono esattamente 4 punti per la calibrazione.")

        src = np.float32(self.points)
        # Ordine atteso: TL, TR, BR, BL  →  coordinate reali in metri
        dst = np.float32([
            [0.0,             0.0],
            [COURT_WIDTH_M,   0.0],
            [COURT_WIDTH_M,   COURT_LENGTH_M],
            [0.0,             COURT_LENGTH_M],
        ])
        self.H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if self.H is None:
            raise RuntimeError("Omografia non calcolabile. Riprova la calibrazione.")

    # ------------------------------------------------------------------
    # Trasformazione punto
    # ------------------------------------------------------------------

    def transform_point(self, px: float, py: float) -> tuple[float, float]:
        """Trasforma coordinate pixel (px, py) in coordinate campo (m)."""
        pt = np.float32([[[px, py]]])
        out = cv2.perspectiveTransform(pt, self.H)
        cx, cy = float(out[0][0][0]), float(out[0][0][1])
        cx = max(0.0, min(COURT_WIDTH_M,  cx))
        cy = max(0.0, min(COURT_LENGTH_M, cy))
        return cx, cy

    def transform_points_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Trasformazione batch.
        points: shape (N, 2)  →  output: shape (N, 2) in metri
        """
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        out[:, 0] = np.clip(out[:, 0], 0.0, COURT_WIDTH_M)
        out[:, 1] = np.clip(out[:, 1], 0.0, COURT_LENGTH_M)
        return out

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    def _save(self):
        data = {"points": self.points, "H": self.H.tolist()}
        with open(self.calibration_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        with open(self.calibration_path, "r") as f:
            data = json.load(f)
        self.points = [tuple(p) for p in data["points"]]
        self.H = np.array(data["H"], dtype=np.float64)
