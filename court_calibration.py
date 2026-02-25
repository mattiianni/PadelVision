"""
Court calibration — basata su colore del campo.

Flusso:
  1. L'utente clicca 3 punti sulla superficie del campo
     → il sistema campiona il colore HSV e segmenta il campo
     → rileva automaticamente i 4 angoli
  2. L'utente clicca 2 punti sulla rete (sx e dx)
     → determina quale metà è "sopra" (y=0) e quale è "sotto" (y=20)
  3. Si calcola l'omografia pixel → metri

Campo padel: 10m x 20m
"""

import cv2
import numpy as np
import json
import os

COURT_WIDTH_M  = 10.0
COURT_LENGTH_M = 20.0


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
        corners = None

        while corners is None:
            # Fase 1: campiona colore campo
            lower, upper = self._phase1_color_sampling(frame)
            # Fase 2: rileva contorno automaticamente
            corners = self._detect_court_corners(frame, lower, upper)
            if corners is None:
                print("  Rilevamento fallito — riprova i click sul campo")

        # Fase 3: due click sulla rete
        net_pts = self._phase2_net_click(frame, corners)

        # Calcola omografia
        self._compute_homography(corners, net_pts)
        self._save()
        print("  Calibrazione completata e salvata.")

    # ------------------------------------------------------------------
    # Fase 1 — campionamento colore campo (3 click)
    # ------------------------------------------------------------------

    def _phase1_color_sampling(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        samples = []
        clone = frame.copy()
        fs = max(w, h) / 1080.0

        self._put_text(clone,
            "FASE 1 — Clicca 3 punti sulla SUPERFICIE del campo",
            (16, 44), fs * 0.8, (0, 220, 255))
        self._put_text(clone,
            "Evita le linee bianche. Poi premi INVIO per rilevare il campo.",
            (16, 44 + int(44 * fs)), fs * 0.65, (180, 180, 180))

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(samples) < 3:
                color_hsv = hsv_frame[y, x]
                samples.append(color_hsv)
                r = max(10, int(12 * fs))
                cv2.circle(clone, (x, y), r, (0, 255, 80), -1)
                cv2.circle(clone, (x, y), r + 3, (255, 255, 255), 2)
                self._put_text(clone, f"Campione {len(samples)}/3",
                               (x + r + 6, y + 6), fs * 0.6, (0, 255, 80))
                cv2.imshow("PadelVision — Calibrazione", clone)

        cv2.namedWindow("PadelVision — Calibrazione", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PadelVision — Calibrazione", min(w, 1440), min(h, 900))
        cv2.setMouseCallback("PadelVision — Calibrazione", on_click)
        cv2.imshow("PadelVision — Calibrazione", clone)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13 and len(samples) >= 1:   # INVIO con almeno 1 campione
                break
            elif key == 27:
                cv2.destroyAllWindows()
                raise RuntimeError("Calibrazione annullata.")

        # Calcola range HSV dai campioni (tolleranza ±25 su H, ±60 su S/V)
        arr = np.array(samples, dtype=np.float32)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0) if len(samples) > 1 else np.array([0, 0, 0])
        tol  = np.array([25, 60, 60], dtype=np.float32)
        lower = np.clip(mean - std - tol, [0,   0,   0  ], [179, 255, 255]).astype(np.uint8)
        upper = np.clip(mean + std + tol, [0,   0,   0  ], [179, 255, 255]).astype(np.uint8)

        return lower, upper

    # ------------------------------------------------------------------
    # Rilevamento automatico contorno campo
    # ------------------------------------------------------------------

    def _detect_court_corners(self, frame, lower, upper):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Pulizia morfologica
        k_big  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        k_med  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  k_big, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   k_med, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k_med, iterations=1)

        # Contorno più grande = campo
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        court = max(contours, key=cv2.contourArea)

        # Deve coprire almeno il 10% dell'immagine
        if cv2.contourArea(court) < 0.10 * w * h:
            return None

        # Approssima a poligono → cerca 4 angoli
        peri  = cv2.arcLength(court, True)
        approx = cv2.approxPolyDP(court, 0.02 * peri, True)

        # Se ha più di 4 punti, prendi i 4 angoli tramite convex hull estremi
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) < 4:
            # Fallback: bounding box ruotato
            rect = cv2.minAreaRect(court)
            pts  = cv2.boxPoints(rect).astype(np.float32)

        corners = self._order_corners(pts)

        # Mostra il risultato per conferma
        preview = frame.copy()
        cv2.drawContours(preview, [corners.astype(np.int32)], -1, (0, 255, 0), 3)
        for i, pt in enumerate(corners):
            cv2.circle(preview, tuple(pt.astype(int)), 10, (0, 200, 255), -1)
            labels = ["TL", "TR", "BR", "BL"]
            self._put_text(preview, labels[i],
                           (int(pt[0]) + 12, int(pt[1]) - 6),
                           max(w, h) / 1080.0 * 0.8, (0, 200, 255))

        fs = max(w, h) / 1080.0
        self._put_text(preview,
            "Campo rilevato! INVIO = OK    R = riprova    ESC = annulla",
            (16, 44), fs * 0.8, (0, 255, 120))

        cv2.imshow("PadelVision — Calibrazione", preview)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13:       # OK
                return corners
            elif key == ord('r') or key == ord('R'):
                return None     # segnale di retry
            elif key == 27:
                cv2.destroyAllWindows()
                raise RuntimeError("Calibrazione annullata.")

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """
        Ordina N punti in: [TL, TR, BR, BL] basandosi su somma e differenza
        delle coordinate (metodo classico per rettangoli in prospettiva).
        Se ci sono più di 4 punti, prende i 4 angoli più estremi.
        """
        if len(pts) > 4:
            # Hull convex → prendi i 4 punti più lontani dal centroide
            hull  = cv2.convexHull(pts.astype(np.float32))
            pts   = hull.reshape(-1, 2)

        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    # ------------------------------------------------------------------
    # Fase 2 — 2 click sulla rete
    # ------------------------------------------------------------------

    def _phase2_net_click(self, frame, corners):
        h, w = frame.shape[:2]
        fs = max(w, h) / 1080.0
        net_pts = []
        clone = frame.copy()

        # Disegna il campo rilevato come sfondo
        cv2.drawContours(clone, [corners.astype(np.int32)], -1, (0, 200, 80), 2)

        self._put_text(clone,
            "FASE 2 — Clicca 2 punti sulla RETE (bordo sinistro e destro)",
            (16, 44), fs * 0.8, (0, 200, 255))

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(net_pts) < 2:
                net_pts.append((x, y))
                r = max(10, int(12 * fs))
                cv2.circle(clone, (x, y), r, (255, 80, 0), -1)
                cv2.circle(clone, (x, y), r + 3, (255, 255, 255), 2)
                lbl = "Rete-Sx" if len(net_pts) == 1 else "Rete-Dx"
                self._put_text(clone, lbl,
                               (x + r + 6, y + 6), fs * 0.65, (255, 180, 0))
                if len(net_pts) == 2:
                    cv2.line(clone, net_pts[0], net_pts[1], (255, 140, 0), 3)
                cv2.imshow("PadelVision — Calibrazione", clone)

        cv2.setMouseCallback("PadelVision — Calibrazione", on_click)
        cv2.imshow("PadelVision — Calibrazione", clone)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13 and len(net_pts) == 2:
                break
            elif key == 27:
                cv2.destroyAllWindows()
                raise RuntimeError("Calibrazione annullata.")

        cv2.destroyAllWindows()
        return net_pts

    # ------------------------------------------------------------------
    # Calcolo omografia
    # ------------------------------------------------------------------

    def _compute_homography(self, corners, net_pts):
        """
        corners  : [TL, TR, BR, BL] in pixel (ordinati in image space)
        net_pts  : [sx, dx] della rete in pixel

        Usa il punto medio della rete per capire quale coppia di corner
        è "vicina alla rete" (y=10m) e quale è "lontana" (y=0 o y=20).
        Poi assegna le coordinate reali di conseguenza.
        """
        tl, tr, br, bl = corners

        # Centroide campo (pixel)
        court_center_y = np.mean([tl[1], tr[1], br[1], bl[1]])
        net_mid_y = (net_pts[0][1] + net_pts[1][1]) / 2.0

        # I due corner "in alto" (y pixel minore) e "in basso"
        top_pair    = sorted([tl, tr], key=lambda p: p[1])  # minore y = più in alto
        bottom_pair = sorted([br, bl], key=lambda p: p[1], reverse=True)

        # Se la rete è nella metà superiore dell'immagine rispetto al campo:
        # → i corner vicini alla rete sono quelli in BASSO nell'immagine (BL, BR)
        # → TL, TR sono il fondo (y=0 o y=20 del campo reale)
        # Altrimenti viceversa.
        # In pratica: assegna TL→(0,0), TR→(10,0), BR→(10,20), BL→(0,20)
        # e poi controlla con la rete se dobbiamo ruotare di 180°.

        # I corner "vicini alla rete" sono quelli la cui y_pixel è più vicina a net_mid_y
        dists = [abs(pt[1] - net_mid_y) for pt in [tl, tr, br, bl]]
        near_net_idxs = np.argsort(dists)[:2]   # i 2 più vicini alla rete

        corner_labels  = {0: "TL", 1: "TR", 2: "BR", 3: "BL"}
        corner_arr     = [tl, tr, br, bl]

        near_net  = [corner_arr[i] for i in near_net_idxs]
        far_net   = [corner_arr[i] for i in range(4) if i not in near_net_idxs]

        # I corner vicini alla rete → y_real = 10 (rete)
        # I corner lontani dalla rete → y_real = 0 oppure 20
        # Decidiamo: i far che hanno y_pixel < net → y_real = 0 (sopra rete)
        #            i far che hanno y_pixel > net → y_real = 20 (sotto rete)
        far_above = [p for p in far_net if p[1] < net_mid_y]
        far_below = [p for p in far_net if p[1] >= net_mid_y]

        # Se tutti i far sono sopra o tutti sotto, usa la distanza
        if not far_above:
            far_above = sorted(far_net, key=lambda p: p[1])[:1]
            far_below = sorted(far_net, key=lambda p: p[1])[1:]
        if not far_below:
            far_below = sorted(far_net, key=lambda p: p[1], reverse=True)[:1]
            far_above = sorted(far_net, key=lambda p: p[1], reverse=True)[1:]

        # Ordina sinistra/destra per ogni gruppo (x minore = sinistro)
        def sort_lr(pts):
            return sorted(pts, key=lambda p: p[0])

        above_l, above_r = sort_lr(far_above) if len(far_above) >= 2 else (far_above[0], far_above[0])
        below_l, below_r = sort_lr(far_below) if len(far_below) >= 2 else (far_below[0], far_below[0])
        net_l,   net_r   = sort_lr(near_net)  if len(near_net)  >= 2 else (near_net[0],  near_net[0])

        # Costruisci src e dst — usiamo 6 punti per un'omografia più robusta
        src = np.float32([
            above_l, above_r,       # y=0  (fondo sopra)
            net_l,   net_r,         # y=10 (rete)
            below_l, below_r,       # y=20 (fondo sotto)
        ])
        dst = np.float32([
            [0.0,           0.0],
            [COURT_WIDTH_M, 0.0],
            [0.0,           COURT_LENGTH_M / 2],
            [COURT_WIDTH_M, COURT_LENGTH_M / 2],
            [0.0,           COURT_LENGTH_M],
            [COURT_WIDTH_M, COURT_LENGTH_M],
        ])

        self.H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if self.H is None:
            raise RuntimeError("Omografia non calcolabile. Riprova la calibrazione.")

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
    def _put_text(img, text, pos, fs, color, thick=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(text, font, fs, thick)
        x, y = pos
        pad = 5
        overlay = img.copy()
        cv2.rectangle(overlay, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
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
