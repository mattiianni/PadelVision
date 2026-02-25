"""
Player tracking con YOLOv8 — versione parallelizzata.

Il video viene diviso in N chunk (uno per core CPU).
Ogni worker carica il proprio modello YOLO e processa il suo chunk.
I risultati vengono uniti e le posizioni assegnate per squadra
in base alla metà campo (non serve ID consistente tra chunk).
"""

import cv2
import numpy as np
import multiprocessing as mp
from ultralytics import YOLO
import torch
import os


# ------------------------------------------------------------------
# Worker (eseguito in ogni sottoprocesso)
# ------------------------------------------------------------------

def _worker(args):
    """
    Ogni worker processa un chunk del video:
      - Carica il proprio modello YOLO (non serializzabile tra processi)
      - Scorre i frame nel range [start_frame, end_frame)
      - Trasforma i piedi dei giocatori in coordinate campo
      - Ritorna lista di (cx, cy, frame_num)
    """
    (video_path, start_frame, end_frame, sample_every,
     H_list, court_w, court_h, conf, model_name, worker_id) = args

    # Sopprime output YOLO nei worker
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["YOLO_VERBOSE"] = "False"

    H = np.array(H_list, dtype=np.float64)
    model = YOLO(model_name)
    model.overrides["verbose"] = False

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    positions = []   # (cx, cy, frame_num)
    frame_num = start_frame

    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_num - start_frame) % sample_every == 0:
            results = model.predict(
                frame,
                classes=[0],
                conf=conf,
                verbose=False,
                imgsz=640,
            )
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    # Piede = bottom-center bbox
                    feet = np.column_stack([
                        (boxes[:, 0] + boxes[:, 2]) / 2,
                        boxes[:, 3],
                    ])
                    pts = feet.reshape(-1, 1, 2).astype(np.float32)
                    court_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                    court_pts[:, 0] = np.clip(court_pts[:, 0], 0.0, court_w)
                    court_pts[:, 1] = np.clip(court_pts[:, 1], 0.0, court_h)

                    for (cx, cy) in court_pts:
                        positions.append((float(cx), float(cy), frame_num))

        frame_num += 1

    cap.release()
    return positions


# ------------------------------------------------------------------
# Classe pubblica
# ------------------------------------------------------------------

class PlayerTracker:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35):
        self.model_name = model_name
        self.conf = conf
        # Usa core fisici (non logici) per evitare contesa
        self.n_workers = max(1, mp.cpu_count() // 2)

    def track_video(
        self,
        video_path: str,
        calibrator,
        sample_every: int = 2,
        progress_callback=None,
    ) -> tuple[dict, float, int]:
        """
        Traccia i giocatori usando tutti i core disponibili.

        Ritorna:
          tracks       — dict {player_id: [(cx, cy, frame_num), ...]}
          fps          — frame rate del video
          total_frames — totale frame
        """
        cap = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        n = min(self.n_workers, total_frames)
        chunk_size = total_frames // n
        chunks = []
        for i in range(n):
            start = i * chunk_size
            end   = (i + 1) * chunk_size if i < n - 1 else total_frames
            chunks.append((start, end))

        args_list = [
            (
                video_path,
                start, end,
                sample_every,
                calibrator.H.tolist(),
                10.0,   # COURT_WIDTH_M
                20.0,   # COURT_LENGTH_M
                self.conf,
                self.model_name,
                i,
            )
            for i, (start, end) in enumerate(chunks)
        ]

        print(f"\n  Avvio {n} worker paralleli su {mp.cpu_count()} core logici...")

        ctx = mp.get_context("spawn")
        results_all = []

        with ctx.Pool(processes=n) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker, args_list)):
                results_all.extend(result)
                if progress_callback:
                    progress_callback(int((i + 1) / n * total_frames), total_frames)

        # Assegna ID giocatore in base alla metà campo
        tracks = self._assign_player_ids(results_all)

        return tracks, fps, total_frames

    # ------------------------------------------------------------------
    # Assegnazione ID per posizione (senza ByteTrack cross-chunk)
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_player_ids(
        all_positions: list,
        court_length: float = 20.0,
        n_bins: int = 8,
    ) -> dict:
        """
        Raggruppa le detection in 'giocatori' usando k-means sul campo 2D.
        Funziona senza ID tracking cross-chunk.
        """
        if not all_positions:
            return {}

        pts = np.array([(cx, cy) for cx, cy, _ in all_positions], dtype=np.float32)

        # k-means con k=4 (4 giocatori padel)
        k = 4
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
        _, labels, centers = cv2.kmeans(
            pts, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        tracks = {}
        for pid in range(k):
            mask = (labels.flatten() == pid)
            pts_cluster = [all_positions[i] for i in range(len(all_positions)) if mask[i]]
            tracks[pid + 1] = pts_cluster

        return tracks

    @staticmethod
    def filter_players(
        tracks: dict,
        min_frames: int = 30,
        max_players: int = 4,
    ) -> dict:
        filtered = {
            pid: pos for pid, pos in tracks.items()
            if len(pos) >= min_frames
        }
        return dict(
            sorted(filtered.items(), key=lambda x: len(x[1]), reverse=True)[:max_players]
        )
